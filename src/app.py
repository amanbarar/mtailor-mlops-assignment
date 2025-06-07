from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
from typing import Dict, Any

from model import load_model

app = FastAPI(title="Image Classification API")
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        model = load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Make prediction on uploaded image.
    
    Args:
        file (UploadFile): Image file to classify
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        start_time = time.time()
        result = model.predict(image)
        inference_time = time.time() - start_time
        
        # Add inference time to response
        result["inference_time"] = inference_time
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"} 