import os
import time
from typing import List

import requests
from fastapi import FastAPI, File, UploadFile
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, HTMLResponse, RedirectResponse


HIGH_MODEL_URL = os.getenv("HIGH_MODEL_URL", "http://high-model-service:8001")
FAST_MODEL_URL = os.getenv("FAST_MODEL_URL", "http://fast-model-service:8002")

P95_LATENCY_THRESHOLD = float(os.getenv("P95_LATENCY_THRESHOLD", "8.0"))
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "75.0"))
ERROR_RATE_THRESHOLD = float(os.getenv("ERROR_RATE_THRESHOLD", "0.05"))
IN_FLIGHT_THRESHOLD = int(os.getenv("IN_FLIGHT_THRESHOLD", "1"))

RECOVERY_COOLDOWN_SECONDS = float(os.getenv("RECOVERY_COOLDOWN_SECONDS", "5.0"))

router_state = {
    "currently_routing_to_fast": False,
    "last_overloaded_time": 0.0,
    "last_decision": "route_to_high_accuracy_model",
    "last_reasons": ["within_sla"]
}

app = FastAPI(title="SLA-Aware Router")


ROUTER_REQUESTS = Counter(
    "router_requests_total",
    "Total requests received by router"
)

ROUTER_HIGH_MODEL_REQUESTS = Counter(
    "router_high_model_requests_total",
    "Requests routed to high-accuracy model"
)

ROUTER_FAST_MODEL_REQUESTS = Counter(
    "router_fast_model_requests_total",
    "Requests routed to optimized fast model"
)

ROUTER_FALLBACKS = Counter(
    "router_fallbacks_total",
    "Total fallback decisions made by router"
)

ROUTER_LATENCY = Histogram(
    "router_latency_seconds",
    "Router-level request latency"
)

HIGH_MODEL_P95 = Gauge(
    "router_observed_high_model_p95_latency_seconds",
    "Observed p95 latency from high-accuracy model"
)

HIGH_MODEL_CPU = Gauge(
    "router_observed_high_model_cpu_percent",
    "Observed CPU usage from high-accuracy model"
)

HIGH_MODEL_IN_FLIGHT = Gauge(
    "router_observed_high_model_in_flight",
    "Observed in-flight requests from high-accuracy model"
)


def get_model_status(model_url: str):
    response = requests.get(f"{model_url}/status", timeout=2)
    response.raise_for_status()
    return response.json()


def is_high_model_overloaded(status: dict):
    """
    Demo-friendly overload logic.

    Model A is considered overloaded only if it is currently busy,
    unhealthy, high CPU, or high error rate.

    Old p95 latency is still displayed on the dashboard, but it does NOT
    force Model A to remain overloaded after the task is finished.
    """

    p95_latency = status.get("p95_latency_seconds", 0)
    cpu_usage = status.get("cpu_usage_percent", 0)
    error_rate = status.get("error_rate", 0)
    in_flight = status.get("in_flight_requests", 0)
    healthy = status.get("healthy", False)

    HIGH_MODEL_P95.set(p95_latency)
    HIGH_MODEL_CPU.set(cpu_usage)
    HIGH_MODEL_IN_FLIGHT.set(in_flight)

    reasons = []

    if not healthy:
        reasons.append("high_accuracy_model_unhealthy")

    # Main automatic recovery rule:
    # If Model A is not processing anything, this becomes false automatically.
    if in_flight >= IN_FLIGHT_THRESHOLD:
        reasons.append("high_accuracy_model_busy")

    # Keep these as extra safety checks.
    if cpu_usage > CPU_THRESHOLD:
        reasons.append("cpu_usage_above_threshold")

    if error_rate > ERROR_RATE_THRESHOLD:
        reasons.append("error_rate_above_threshold")

    return len(reasons) > 0, reasons

def get_stable_routing_decision(status: dict):
    """
    Prevents flickering between Model A and Model B.

    If Model A becomes busy, route to Model B immediately.
    After Model A becomes free, wait for RECOVERY_COOLDOWN_SECONDS
    before switching back to Model A.
    """

    now = time.time()
    overloaded, reasons = is_high_model_overloaded(status)

    if overloaded:
        router_state["currently_routing_to_fast"] = True
        router_state["last_overloaded_time"] = now
        router_state["last_decision"] = "route_to_fast_model"
        router_state["last_reasons"] = reasons
        return True, reasons, "route_to_fast_model"

    # If Model A is now free, do not immediately switch back.
    # Wait for cooldown to avoid flickering.
    time_since_overload = now - router_state["last_overloaded_time"]

    if router_state["currently_routing_to_fast"] and time_since_overload < RECOVERY_COOLDOWN_SECONDS:
        remaining = round(RECOVERY_COOLDOWN_SECONDS - time_since_overload, 2)
        reasons = [f"recovery_cooldown_active_{remaining}s"]

        router_state["last_decision"] = "route_to_fast_model"
        router_state["last_reasons"] = reasons

        return True, reasons, "route_to_fast_model"

    # Model A has been stable long enough. Route back to Model A.
    router_state["currently_routing_to_fast"] = False
    router_state["last_decision"] = "route_to_high_accuracy_model"
    router_state["last_reasons"] = ["high_accuracy_model_available"]

    return False, ["high_accuracy_model_available"], "route_to_high_accuracy_model"


def forward_batch(model_url: str, files: List[UploadFile]):
    multipart_files = []

    for file in files:
        file.file.seek(0)
        content = file.file.read()
        multipart_files.append(
            ("files", (file.filename, content, file.content_type))
        )

    response = requests.post(
        f"{model_url}/predict-batch",
        files=multipart_files,
        timeout=600
    )

    response.raise_for_status()
    return response.json()


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "SLA-Aware Router",
        "high_model_url": HIGH_MODEL_URL,
        "fast_model_url": FAST_MODEL_URL,
        "p95_latency_threshold": P95_LATENCY_THRESHOLD,
        "cpu_threshold": CPU_THRESHOLD,
        "error_rate_threshold": ERROR_RATE_THRESHOLD,
        "in_flight_threshold": IN_FLIGHT_THRESHOLD
    }


@app.get("/routing-state")
def routing_state():
    try:
        high_status = get_model_status(HIGH_MODEL_URL)
        overloaded, reasons, decision = get_stable_routing_decision(high_status)

        return {
            "high_model_status": high_status,
            "high_model_overloaded": overloaded,
            "routing_reasons": reasons,
            "current_decision": decision,
            "recovery_cooldown_seconds": RECOVERY_COOLDOWN_SECONDS
        }

    except Exception as e:
        return {
            "high_model_overloaded": True,
            "routing_reasons": ["unable_to_read_high_model_status"],
            "current_decision": "route_to_fast_model",
            "error": str(e)
        }


@app.post("/predict-batch")
def predict_batch(files: List[UploadFile] = File(...)):
    ROUTER_REQUESTS.inc()
    start_time = time.time()

    try:
        high_status = get_model_status(HIGH_MODEL_URL)
        overloaded, routing_reason, decision = get_stable_routing_decision(high_status)

        if overloaded:
            selected_model = "Model B - Optimized Fast Model"
            ROUTER_FAST_MODEL_REQUESTS.inc()
            ROUTER_FALLBACKS.inc()

            user_notice = (
                "The high-accuracy model is currently busy, so your images "
                "were processed using the faster optimized model. Results may be slightly "
                "less accurate. If you need the highest accuracy, please try again shortly."
            )

            result = forward_batch(FAST_MODEL_URL, files)

        else:
            selected_model = "Model A - High Accuracy Model"
            ROUTER_HIGH_MODEL_REQUESTS.inc()
            user_notice = None
            routing_reason = ["high_accuracy_model_available"]
            result = forward_batch(HIGH_MODEL_URL, files)

        router_latency = time.time() - start_time
        ROUTER_LATENCY.observe(router_latency)

        return {
            "status": "success",
            "selected_model": selected_model,
            "routing_reason": routing_reason,
            "user_notice": user_notice,
            "router_latency_seconds": round(router_latency, 4),
            "result": result
        }

    except Exception as e:
        ROUTER_FAST_MODEL_REQUESTS.inc()
        ROUTER_FALLBACKS.inc()

        user_notice = (
            "The high-accuracy model is unavailable or busy, so your images "
            "were processed using the faster optimized model."
        )

        result = forward_batch(FAST_MODEL_URL, files)

        return {
            "status": "success",
            "selected_model": "Model B - Optimized Fast Model",
            "routing_reason": ["fallback_due_to_high_model_error"],
            "user_notice": user_notice,
            "result": result,
            "original_error": str(e)
        }


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SLA-Aware ML Serving Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 0;
        }

        .header {
            background: #111827;
            color: white;
            padding: 22px;
            text-align: center;
        }

        .header h1 {
            margin-bottom: 8px;
        }

        .container {
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 20px;
            padding: 20px;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }

        h2 {
            margin-top: 0;
            color: #111827;
        }

        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 1px dashed #9ca3af;
            border-radius: 8px;
            background: #f9fafb;
        }

        button {
            margin-top: 15px;
            padding: 12px 18px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-size: 15px;
            cursor: pointer;
        }

        button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            padding: 9px 0;
            border-bottom: 1px solid #e5e7eb;
            font-size: 15px;
        }

        .metric span:first-child {
            font-weight: bold;
            color: #374151;
        }

        .metric span:last-child {
            text-align: right;
            word-break: break-word;
        }

        .ok {
            color: #059669;
            font-weight: bold;
        }

        .bad {
            color: #dc2626;
            font-weight: bold;
        }

        .notice {
            background: #fff7ed;
            color: #9a3412;
            border-left: 5px solid #f97316;
            padding: 12px;
            border-radius: 8px;
            margin-top: 15px;
        }

        pre {
            background: #111827;
            color: #e5e7eb;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            max-height: 450px;
            white-space: pre-wrap;
        }

        .small {
            color: #6b7280;
            font-size: 13px;
        }

        .chart-block {
            margin-top: 18px;
        }

        canvas {
            width: 100%;
            height: 120px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #f9fafb;
            margin-top: 8px;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
    <div class="header">
        <h1>SLA-Aware Multi-Variant ML Serving Demo</h1>
        <p>Upload images and watch the router switch between Model A and Model B based on live system load.</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>Image Batch Upload</h2>
            <p class="small">
                To demo fallback: start a large upload in one tab, then quickly upload again in another tab.
            </p>

            <input id="files" type="file" multiple accept="image/*">

            <br>
            <button id="uploadBtn" onclick="uploadImages()">Upload Images</button>

            <div id="uploadStatus"></div>

            <h2 style="margin-top: 25px;">Prediction Response</h2>
            <div id="notice"></div>
            <pre id="result">No request sent yet.</pre>
        </div>

        <div class="card">
            <h2>Live System SLA Dashboard</h2>

            <div class="metric">
                <span>Current Decision</span>
                <span id="decision">Loading...</span>
            </div>

            <div class="metric">
                <span>Model A Currently Busy</span>
                <span id="overloaded">Loading...</span>
            </div>

            <div class="metric">
                <span>In-Flight Requests</span>
                <span id="inflight">Loading...</span>
            </div>

            <div class="metric">
                <span>p95 Latency</span>
                <span id="p95">Loading...</span>
            </div>

            <div class="metric">
                <span>CPU Usage</span>
                <span id="cpu">Loading...</span>
            </div>

            <div class="metric">
                <span>Error Rate</span>
                <span id="errorRate">Loading...</span>
            </div>

            <div class="metric">
                <span>Total Requests</span>
                <span id="totalRequests">Loading...</span>
            </div>

            <div class="metric">
                <span>Routing Reasons</span>
                <span id="reasons">Loading...</span>
            </div>

            <h3 style="margin-top: 25px;">Live Graphs</h3>

            <div class="chart-block">
                <b>p95 Latency Over Time</b>
                <canvas id="p95Chart" width="360" height="120"></canvas>
            </div>

            <div class="chart-block">
                <b>CPU Usage Over Time</b>
                <canvas id="cpuChart" width="360" height="120"></canvas>
            </div>

            <div class="chart-block">
                <b>In-Flight Requests Over Time</b>
                <canvas id="inflightChart" width="360" height="120"></canvas>
            </div>

            <p class="small" style="margin-top: 20px;">
                Dashboard refreshes every 1 second using the router's /routing-state endpoint.
            </p>
        </div>
    </div>

<script>
const maxPoints = 40;

const p95History = [];
const cpuHistory = [];
const inflightHistory = [];

function pushMetric(history, value) {
    history.push(Number(value) || 0);

    if (history.length > maxPoints) {
        history.shift();
    }
}

function drawChart(canvasId, data, maxValue, labelSuffix) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#f9fafb";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(35, 10);
    ctx.lineTo(35, height - 25);
    ctx.lineTo(width - 10, height - 25);
    ctx.stroke();

    if (data.length === 0) {
        return;
    }

    const actualMax = Math.max(maxValue, ...data, 1);
    const plotWidth = width - 50;
    const plotHeight = height - 40;

    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((value, index) => {
        const x = 35 + (index / Math.max(data.length - 1, 1)) * plotWidth;
        const y = height - 25 - (value / actualMax) * plotHeight;

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    const latest = data[data.length - 1];

    ctx.fillStyle = "#111827";
    ctx.font = "12px Arial";
    ctx.fillText("Latest: " + latest.toFixed(2) + " " + labelSuffix, 45, 20);
}

async function refreshDashboard() {
    try {
        const response = await fetch("/routing-state");
        const data = await response.json();

        const status = data.high_model_status || {};

        document.getElementById("decision").innerText = data.current_decision || "unknown";

        const overloaded = data.high_model_overloaded;
        const overloadedEl = document.getElementById("overloaded");
        overloadedEl.innerText = overloaded ? "YES" : "NO";
        overloadedEl.className = overloaded ? "bad" : "ok";

        const inflightValue = status.in_flight_requests ?? 0;
        const p95Value = status.p95_latency_seconds ?? 0;
        const cpuValue = status.cpu_usage_percent ?? 0;
        const errorValue = status.error_rate ?? 0;
        const totalRequests = status.total_requests ?? 0;

        document.getElementById("inflight").innerText = inflightValue;
        document.getElementById("p95").innerText = p95Value + " sec";
        document.getElementById("cpu").innerText = cpuValue + " %";
        document.getElementById("errorRate").innerText = errorValue;
        document.getElementById("totalRequests").innerText = totalRequests;
        document.getElementById("reasons").innerText =
            (data.routing_reasons || []).join(", ") || "within_sla";

        pushMetric(p95History, p95Value);
        pushMetric(cpuHistory, cpuValue);
        pushMetric(inflightHistory, inflightValue);

        drawChart("p95Chart", p95History, 10, "sec");
        drawChart("cpuChart", cpuHistory, 100, "%");
        drawChart("inflightChart", inflightHistory, 5, "req");

    } catch (err) {
        document.getElementById("decision").innerText = "dashboard error";
        document.getElementById("reasons").innerText = err.toString();
    }
}

async function uploadImages() {
    const fileInput = document.getElementById("files");
    const files = fileInput.files;

    if (!files.length) {
        alert("Please select at least one image.");
        return;
    }

    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }

    const uploadBtn = document.getElementById("uploadBtn");
    const uploadStatus = document.getElementById("uploadStatus");
    const resultBox = document.getElementById("result");
    const noticeBox = document.getElementById("notice");

    uploadBtn.disabled = true;
    uploadStatus.innerHTML = "<p>Uploading " + files.length + " image(s)...</p>";
    resultBox.innerText = "Waiting for prediction...";
    noticeBox.innerHTML = "";

    try {
        const response = await fetch("/predict-batch", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.user_notice) {
            noticeBox.innerHTML =
                '<div class="notice"><b>Notice:</b> ' + data.user_notice + '</div>';
        }

        resultBox.innerText = JSON.stringify(data, null, 2);

    } catch (err) {
        resultBox.innerText = "Upload failed: " + err.toString();
    } finally {
        uploadBtn.disabled = false;
        uploadStatus.innerHTML = "<p>Upload complete.</p>";
        refreshDashboard();
    }
}

setInterval(refreshDashboard, 1000);
refreshDashboard();
</script>
</body>
</html>
    """


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)