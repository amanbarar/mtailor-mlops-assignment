import onnxruntime
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Any

class ONNXModel:
    """Class for loading and running ONNX model with preprocessing."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize ONNX model.
        
        Args:
            model_path (str): Path to ONNX model file. If None, uses default path.
        """
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "model.onnx")
        
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 224x224
        image = image.resize((224, 224), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize using mean and std
        img_array = (img_array - self.mean) / self.std
        
        # Transpose to NCHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Ensure float32 dtype
        img_array = img_array.astype(np.float32)
        
        return img_array
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Make prediction on input image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            Dict[str, Any]: Prediction results including class_id and confidence
        """
        if image is None:
            raise ValueError("Input image cannot be None")
        if not hasattr(image, 'size') or image.size[0] <= 0 or image.size[1] <= 0:
            raise ValueError("Invalid image size")
        
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: img_array}
        )
        
        # Get predictions
        logits = outputs[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        class_id = np.argmax(probabilities)
        confidence = float(probabilities[0, class_id])
        
        return {
            "class_id": int(class_id),
            "confidence": confidence,
            "predictions": probabilities[0].tolist()
        }

def load_model() -> ONNXModel:
    """
    Load ONNX model.
    
    Returns:
        ONNXModel: Loaded model instance
    """
    return ONNXModel() 