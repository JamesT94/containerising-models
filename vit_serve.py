from fastapi import FastAPI, Request, File, UploadFile
from prometheus_client import start_http_server, Counter, Histogram, generate_latest
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import io
import time

# Initialize FastAPI app
app = FastAPI()

# Load the pretrained model and feature extractor
model = ViTForImageClassification.from_pretrained('vit_model')
feature_extractor = ViTFeatureExtractor.from_pretrained('vit_feature_extractor')

# # Define Prometheus metrics
# REQUEST_COUNT = Counter('request_count', 'Total number of requests')
# REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')

# @app.middleware("http")
# async def prometheus_middleware(request: Request, call_next):
#     REQUEST_COUNT.inc()
#     start_time = time.time()
#     response = await call_next(request)
#     latency = time.time() - start_time
#     REQUEST_LATENCY.observe(latency)
#     return response

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Return the prediction
    return {"predicted_class_idx": predicted_class_idx}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vision Transformer model API for image classification!"}

# @app.get("/metrics")
# async def metrics():
#     return generate_latest()

if __name__ == "__main__":
    import uvicorn
    # start_http_server(8001)  # Start the Prometheus metrics server on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8000)
