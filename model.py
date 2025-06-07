import onnxruntime as ort
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Class for handling image preprocessing"""
    
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = (224, 224)
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image for model inference
        Args:
            image_path (str): Path to the input image
        Returns:
            np.ndarray: Preprocessed image tensor in float32 format
        """
        try:

            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.target_size, Image.BILINEAR)

            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = (image_array - self.mean) / self.std
            
            image_array = np.transpose(image_array, (2, 0, 1))
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

class ONNXModel:
    """Class for ONNX model inference"""
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX model
        Args:
            model_path (str): Path to the ONNX model file
        """
        try:
            self.session = ort.InferenceSession(model_path)
            self.preprocessor = ImagePreprocessor()
            logger.info(f"Successfully loaded ONNX model from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax function to array
        Args:
            x (np.ndarray): Input array
        Returns:
            np.ndarray: Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict(self, image_path: str) -> Tuple[int, float]:
        """
        Make prediction on an image
        Args:
            image_path (str): Path to the input image
        Returns:
            Tuple[int, float]: Predicted class ID and confidence score
        """
        try:
            input_tensor = self.preprocessor.preprocess(image_path)
            
            if input_tensor.dtype != np.float32:
                input_tensor = input_tensor.astype(np.float32)
            
            outputs = self.session.run(None, {'input': input_tensor})
            logits = outputs[0][0]
            
            probabilities = self._softmax(logits)
            
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise 