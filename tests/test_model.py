import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
import pytest
from PIL import Image
import numpy as np

from model import ONNXModel

@pytest.fixture
def model():
    """Create model fixture."""
    return ONNXModel()

@pytest.fixture
def test_image():
    """Load test image."""
    image_path = Path(__file__).parent.parent / "assets" / "n01440764_tench.jpeg"
    return Image.open(image_path)

def test_model_initialization(model):
    """Test model initialization."""
    assert model is not None
    assert model.session is not None
    assert model.input_name is not None
    assert model.output_name is not None

def test_preprocess_image(model, test_image):
    """Test image preprocessing."""
    processed = model.preprocess_image(test_image)
    
    # Check shape
    assert processed.shape == (1, 3, 224, 224)
    
    # Check value range
    assert processed.min() >= -3.0  # After normalization
    assert processed.max() <= 3.0   # After normalization

def test_predict(model, test_image):
    """Test model prediction."""
    result = model.predict(test_image)
    
    # Check result structure
    assert "class_id" in result
    assert "confidence" in result
    assert "predictions" in result
    
    # Check types
    assert isinstance(result["class_id"], int)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["predictions"], list)
    
    # Check prediction length
    assert len(result["predictions"]) == 1000  # ImageNet classes
    
    # Check confidence range
    assert 0 <= result["confidence"] <= 1

def test_invalid_image(model):
    """Test handling of invalid image."""
    import pytest
    # Passing None
    with pytest.raises(Exception):
        model.predict(None)
    # Passing a non-image type
    with pytest.raises(Exception):
        model.predict("not an image") 