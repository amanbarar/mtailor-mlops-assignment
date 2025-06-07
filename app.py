from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from model import ONNXModel
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Classification API")

MODEL_PATH = "model.onnx"
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model = ONNXModel(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make prediction on uploaded image
    Args:
        file (UploadFile): Image file to classify
    Returns:
        JSONResponse: Prediction results
    """
    try:
        temp_path = f"temp_{int(time.time())}.jpg"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        start_time = time.time()
        class_id, confidence = model.predict(temp_path)
        end_time = time.time()
        
        response_time = end_time - start_time

        os.remove(temp_path)
        
        return JSONResponse({
            "class_id": int(class_id),
            "confidence": float(confidence),
            "response_time": float(response_time)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Healthcheck"""
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 