import os
import time
import threading
from collections import deque
from typing import List

import psutil
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from torchvision import models


MODEL_ARCH = os.getenv("MODEL_ARCH", "resnet50")
MODEL_LABEL = os.getenv("MODEL_LABEL", "Model B - Optimized")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))

app = FastAPI(title=MODEL_LABEL)

lock = threading.Lock()
in_flight_requests = 0
latency_window = deque(maxlen=100)
total_requests = 0
total_errors = 0

REQUEST_COUNT = Counter("model_requests_total", "Total model requests", ["model"])
ERROR_COUNT = Counter("model_errors_total", "Total model errors", ["model"])
REQUEST_LATENCY = Histogram("model_latency_seconds", "Model inference latency", ["model"])
IN_FLIGHT = Gauge("model_in_flight_requests", "Current in-flight requests", ["model"])
CPU_USAGE = Gauge("model_cpu_usage_percent", "CPU usage percentage", ["model"])


def load_model():
    if MODEL_ARCH == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights)
    else:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


model, preprocess, categories = load_model()


def calculate_p95(values):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(0.95 * len(sorted_values)) - 1
    index = max(index, 0)
    return sorted_values[index]


def classify_image(image_file: UploadFile):
    image = Image.open(image_file.file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        confidence, class_id = torch.max(probabilities, dim=0)

    return {
        "filename": image_file.filename,
        "prediction": categories[class_id.item()],
        "confidence": round(confidence.item(), 4)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_label": MODEL_LABEL,
        "model_arch": MODEL_ARCH
    }


@app.get("/status")
def status():
    cpu = psutil.cpu_percent(interval=0.1)
    CPU_USAGE.labels(model=MODEL_ARCH).set(cpu)

    with lock:
        p95_latency = calculate_p95(list(latency_window))
        error_rate = total_errors / total_requests if total_requests > 0 else 0.0

        return {
            "model_label": MODEL_LABEL,
            "model_arch": MODEL_ARCH,
            "healthy": True,
            "in_flight_requests": in_flight_requests,
            "p95_latency_seconds": round(p95_latency, 4),
            "cpu_usage_percent": round(cpu, 2),
            "error_rate": round(error_rate, 4),
            "total_requests": total_requests,
            "total_errors": total_errors
        }


@app.post("/predict-batch")
def predict_batch(files: List[UploadFile] = File(...)):
    global in_flight_requests, total_requests, total_errors

    if len(files) > MAX_BATCH_SIZE:
        return {
            "status": "error",
            "message": f"Batch too large. Max allowed batch size is {MAX_BATCH_SIZE} images."
        }

    REQUEST_COUNT.labels(model=MODEL_ARCH).inc()

    with lock:
        in_flight_requests += 1
        total_requests += 1
        IN_FLIGHT.labels(model=MODEL_ARCH).set(in_flight_requests)

    start_time = time.time()

    try:
        predictions = []

        for image_file in files:
            predictions.append(classify_image(image_file))

        latency = time.time() - start_time

        with lock:
            latency_window.append(latency)

        REQUEST_LATENCY.labels(model=MODEL_ARCH).observe(latency)

        return {
            "status": "success",
            "model_used": MODEL_LABEL,
            "model_arch": MODEL_ARCH,
            "batch_size": len(files),
            "latency_seconds": round(latency, 4),
            "predictions": predictions
        }

    except Exception as e:
        ERROR_COUNT.labels(model=MODEL_ARCH).inc()

        with lock:
            total_errors += 1

        return {
            "status": "error",
            "model_used": MODEL_LABEL,
            "message": str(e)
        }

    finally:
        with lock:
            in_flight_requests -= 1
            IN_FLIGHT.labels(model=MODEL_ARCH).set(in_flight_requests)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)