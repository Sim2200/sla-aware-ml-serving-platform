# SLA-Aware Multi-Variant ML Serving Platform

> **Team course project (MSML605, University of Maryland, Spring 2026, team of 2).** This is my copy of
> the team repository, with its full commit history. **My part:** both model variants (the high-accuracy
> model and the optimized fast model, including the ONNX quantization work), the FastAPI inference
> services, the SLA-aware routing logic, and the Locust load testing and performance analysis. My
> teammate owned the Kubernetes deployment, autoscaling, Prometheus/Grafana and the canary controller.
>
> Result from our load tests: under high load the SLA-aware router cut p95 latency from 30+ s to about
> 5 s and the error rate from 5–10% to 0–2%, compared with sending everything to the high-accuracy model.
>
> I later rebuilt this idea from scratch as a solo project with measured, reproducible results:
> [sla-inference-gateway](https://github.com/Sim2200/sla-inference-gateway).

The rest of this README is the team's original run guide.

---

This README explains how to run the full project from scratch on Windows using Docker Desktop Kubernetes.

The project demonstrates:

- FastAPI model serving
- Model A: high-accuracy model
- Model B: optimized fast model
- SLA-aware router with web UI
- Kubernetes deployment
- HPA autoscaling
- Prometheus monitoring
- Grafana visualization

The main idea is simple: Model A is preferred because it is more accurate. If Model A is busy or unhealthy, the SLA router temporarily sends new requests to Model B to keep latency low.

---

## 1. Prerequisites

Install these first:

- Docker Desktop for Windows
- Kubernetes enabled inside Docker Desktop
- kubectl
- PowerShell

Enable Kubernetes:

```text
Docker Desktop → Settings → Kubernetes → Enable Kubernetes → Apply & Restart
```

Verify Kubernetes:

```powershell
kubectl get nodes
```

Expected:

```text
docker-desktop   Ready
```

---

## 2. Go to the Project Folder

```powershell
cd C:\Users\shash\sla-ml-serving
```

Expected structure:

```text
sla-ml-serving/
├── high_model/
├── fast_model/
├── router/
├── k8s/
│   ├── high-model.yaml
│   ├── fast-model.yaml
│   ├── router.yaml
│   └── hpa.yaml
└── monitoring/
    ├── prometheus-config.yaml
    ├── prometheus.yaml
    └── grafana.yaml
```

If your folder names are slightly different, use your actual local folder names.

---

## 3. Build Docker Images

Build Model A:

```powershell
docker build -t high-model:latest ./high_model
```

Build Model B:

```powershell
docker build -t fast-model:latest ./fast_model
```

Build SLA Router:

```powershell
docker build -t sla-router:latest ./router
```

Verify images:

```powershell
docker images
```

You should see:

```text
high-model
fast-model
sla-router
```

---

## 4. Create Namespace

```powershell
kubectl create namespace sla-serving
```

If it already exists, continue.

Check namespaces:

```powershell
kubectl get namespaces
```

---

## 5. Deploy Model A, Model B, and Router

```powershell
kubectl apply -f k8s/high-model.yaml -n sla-serving
kubectl apply -f k8s/fast-model.yaml -n sla-serving
kubectl apply -f k8s/router.yaml -n sla-serving
```

Check pods:

```powershell
kubectl get pods -n sla-serving
```

Expected:

```text
high-model-xxxxx     1/1   Running
fast-model-xxxxx     1/1   Running
sla-router-xxxxx     1/1   Running
```

Check services:

```powershell
kubectl get svc -n sla-serving
```

Expected services:

```text
high-model-service
fast-model-service
sla-router-service
```

---

## 6. Run the UI on Port 8080

Start port-forwarding:

```powershell
kubectl port-forward svc/sla-router-service 8080:8000 -n sla-serving
```

Keep this PowerShell window open.

Open the UI:

```text
http://localhost:8080/ui
```

Test health:

```powershell
curl.exe http://localhost:8080/health
```

Expected:

```json
{
  "status": "healthy",
  "service": "SLA-Aware Router"
}
```

---

## 7. Test SLA-Aware Routing

Open two browser tabs:

```text
http://localhost:8080/ui
```

Demo steps:

1. In Tab 1, upload a batch of images.
2. While Tab 1 is processing, upload another batch in Tab 2.
3. The router should detect Model A is busy.
4. The second request should route to Model B.

Expected behavior:

```text
Model A free → route_to_high_accuracy_model
Model A busy → route_to_fast_model
Model A stable again → route_to_high_accuracy_model
```

The UI dashboard shows:

```text
Current Decision
Model A Currently Busy
In-Flight Requests
p95 Latency
CPU Usage
Error Rate
Total Requests
Routing Reasons
```

---

## 8. Install Metrics Server for HPA

HPA needs Metrics Server.

Install Metrics Server:

```powershell
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

For Docker Desktop, patch Metrics Server:

```powershell
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/args/-",
    "value": "--kubelet-insecure-tls"
  }
]'
```

Restart Metrics Server:

```powershell
kubectl rollout restart deployment metrics-server -n kube-system
kubectl rollout status deployment metrics-server -n kube-system
```

Verify metrics:

```powershell
kubectl top pods -n sla-serving
```

If CPU and memory values appear, Metrics Server is working.

---

## 9. Enable HPA Autoscaling

Apply HPA:

```powershell
kubectl apply -f k8s/hpa.yaml
```

Check HPA:

```powershell
kubectl get hpa -n sla-serving
```

Expected:

```text
high-model-hpa
fast-model-hpa
```

Watch HPA:

```powershell
kubectl get hpa -n sla-serving -w
```

In another PowerShell window, watch pods:

```powershell
kubectl get pods -n sla-serving -w
```

Generate load from the UI. If CPU crosses the HPA threshold, Kubernetes should scale pods from 1 up to 3 replicas.

Useful HPA commands:

```powershell
kubectl get hpa -n sla-serving
kubectl describe hpa high-model-hpa -n sla-serving
kubectl describe hpa fast-model-hpa -n sla-serving
kubectl top pods -n sla-serving
```

---

## 10. Deploy Prometheus

Apply Prometheus files:

```powershell
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/prometheus.yaml
```

Check Prometheus:

```powershell
kubectl get pods -n sla-serving
```

Expected:

```text
prometheus-xxxxx   1/1   Running
```

Port-forward Prometheus:

```powershell
kubectl port-forward svc/prometheus-service 9090:9090 -n sla-serving
```

Open:

```text
http://localhost:9090
```

Go to:

```text
Status → Targets
```

The following should be `UP`:

```text
sla-router
high-model
fast-model
```

---

## 11. Top 6 Prometheus Queries

Use these on the Prometheus Query page.

### 1. Fallback Traffic Percentage

```promql
100 * increase(router_fallbacks_total[5m]) / increase(router_requests_total[5m])
```

Shows what percentage of recent traffic was shifted from Model A to Model B.

### 2. Model A Busy State

```promql
router_observed_high_model_in_flight
```

Shows whether Model A is occupied.

```text
0 = free
1 = busy
```

### 3. Router p95 Latency

```promql
histogram_quantile(0.95, rate(router_latency_seconds_bucket[5m]))
```

Shows p95 latency at the SLA router.

### 4. Model B Usage in Last 5 Minutes

```promql
increase(router_fast_model_requests_total[5m])
```

Shows how many recent requests were routed to Model B.

### 5. Model A Usage in Last 5 Minutes

```promql
increase(router_high_model_requests_total[5m])
```

Shows how many recent requests were handled by Model A.

### 6. Request Traffic Rate

```promql
rate(router_requests_total[1m]) * 60
```

Shows request traffic in requests per minute.

---

## 12. Deploy Grafana

Apply Grafana:

```powershell
kubectl apply -f monitoring/grafana.yaml
```

Check Grafana:

```powershell
kubectl get pods -n sla-serving
```

Port-forward Grafana:

```powershell
kubectl port-forward svc/grafana-service 3000:3000 -n sla-serving
```

Open:

```text
http://localhost:3000
```

Default login:

```text
Username: admin
Password: admin
```

Add Prometheus as a data source:

```text
Connections → Data Sources → Add data source → Prometheus
```

Use this URL:

```text
http://prometheus-service:9090
```

Click:

```text
Save & Test
```

---

## 13. Recommended Presentation Demo Flow

1. Show running pods:

```powershell
kubectl get pods -n sla-serving
```

2. Show services:

```powershell
kubectl get svc -n sla-serving
```

3. Show HPA:

```powershell
kubectl get hpa -n sla-serving
```

4. Open UI:

```text
http://localhost:8080/ui
```

5. Upload images in one tab to make Model A busy.
6. Upload images in another tab to trigger fallback to Model B.
7. Show Prometheus query:

```promql
router_observed_high_model_in_flight
```

8. Show fallback percentage:

```promql
100 * increase(router_fallbacks_total[5m]) / increase(router_requests_total[5m])
```

9. Show HPA scaling:

```powershell
kubectl get hpa -n sla-serving
kubectl get pods -n sla-serving
```

---

## 14. Clean Restart

If the system becomes messy, restart the namespace.

Warning: this deletes all project pods and services.

```powershell
kubectl delete namespace sla-serving
kubectl create namespace sla-serving
```

Redeploy:

```powershell
kubectl apply -f k8s/high-model.yaml -n sla-serving
kubectl apply -f k8s/fast-model.yaml -n sla-serving
kubectl apply -f k8s/router.yaml -n sla-serving
kubectl apply -f k8s/hpa.yaml

kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/prometheus.yaml
kubectl apply -f monitoring/grafana.yaml
```

Check:

```powershell
kubectl get pods -n sla-serving
kubectl get svc -n sla-serving
kubectl get hpa -n sla-serving
```

Start UI again:

```powershell
kubectl port-forward svc/sla-router-service 8080:8000 -n sla-serving
```

Open:

```text
http://localhost:8080/ui
```

---

## 15. Troubleshooting

### UI shows Not Found

Use:

```text
http://localhost:8080/ui
```

The UI is served at `/ui`.

### Metrics API not available

Patch Metrics Server:

```powershell
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/args/-",
    "value": "--kubelet-insecure-tls"
  }
]'
```

Then restart:

```powershell
kubectl rollout restart deployment metrics-server -n kube-system
kubectl rollout status deployment metrics-server -n kube-system
```

### HPA shows cpu unknown

Wait 30 to 60 seconds and run:

```powershell
kubectl get hpa -n sla-serving
kubectl top pods -n sla-serving
```

### Prometheus does not open

Check pod and service:

```powershell
kubectl get pods -n sla-serving
kubectl get svc -n sla-serving
```

Then port-forward:

```powershell
kubectl port-forward svc/prometheus-service 9090:9090 -n sla-serving
```

### Upload failed because batch is too large

Avoid uploading hundreds of images on Docker Desktop.

Recommended demo size:

```text
5 to 20 images for normal demo
30 to 50 images for stress demo
```

Very large uploads can cause memory pressure or pod restarts.

---

## 16. Project Summary

This project demonstrates a production-style MLOps serving system.

```text
Model A = higher accuracy, slower inference
Model B = faster fallback model
SLA Router = decides which model receives traffic
HPA = adds pods when CPU load increases
Prometheus = collects metrics
Grafana = visualizes metrics
```

The core behavior is:

```text
If Model A is free and healthy:
    route request to Model A

If Model A is busy or unhealthy:
    route request to Model B

If CPU load increases:
    HPA automatically scales model pods

Prometheus and Grafana monitor routing, latency, fallback events, and traffic load
```
