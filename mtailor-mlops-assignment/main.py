from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from model import ONNXModel
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Image Classification API")

# Initialize model
MODEL_PATH = "model.onnx"
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
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
    Make prediction on uploaded image.
    
    Args:
        file (UploadFile): Image file to classify
        
    Returns:
        JSONResponse: Prediction results
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{int(time.time())}.jpg"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Make prediction
        start_time = time.time()
        class_id, confidence = model.predict(temp_path)
        end_time = time.time()
        
        # Calculate response time
        response_time = end_time - start_time
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Return prediction
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
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# To deploy your app, run:
# cerebrium deploy
