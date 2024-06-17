from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import Response
from prometheus_client import start_http_server, Counter, Histogram, Summary, Gauge, generate_latest, CONTENT_TYPE_LATEST
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import io
import time
import json

# Initialize FastAPI app
app = FastAPI()

# Load the pretrained model and feature extractor
model = ViTForImageClassification.from_pretrained('vit_model')
feature_extractor = ViTFeatureExtractor.from_pretrained('vit_feature_extractor')

# Load id2label mapping from the JSON configuration file
with open('vit_model/config.json', 'r') as f:
    config = json.load(f)
    id2label = config["id2label"]

# Define Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total number of requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")

INFERENCE_TIME = Summary("inference_time_seconds", "Time spent on inference")
# ACCURACY = Gauge("model_accuracy", "Model accuracy over time")
PREDICTED_CLASSES = Counter("predicted_classes", "Distribution of predicted classes", ["class"])


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    REQUEST_COUNT.inc()  # Increment the request count
    start_time = time.time()  # Start time for latency measurement
    response = await call_next(request)
    latency = time.time() - start_time  # Calculate latency
    REQUEST_LATENCY.observe(latency)  # Record latency
    return response


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Get predictions
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)

    inference_time = time.time() - start_time
    INFERENCE_TIME.observe(inference_time)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Get the label from the index
    predicted_label = id2label.get(str(predicted_class_idx), "Unknown")

    # Increment the predicted class counter
    PREDICTED_CLASSES.labels(predicted_label).inc()

    # Return the prediction
    return {"predicted_class_idx": predicted_class_idx, "predicted_label": predicted_label}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Vision Transformer model API for image classification!"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    start_http_server(8001)  # Start the Prometheus metrics server on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8000)