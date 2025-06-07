import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path to import pytorch_model
sys.path.append(str(Path(__file__).parent.parent))
from models.pytorch_model import Classifier

def convert_to_onnx():
    """
    Convert PyTorch model to ONNX format with preprocessing steps included.
    """
    # Initialize model
    model = Classifier()
    
    # Load weights
    weights_path = Path(__file__).parent.parent / "models" / "pytorch_model_weights.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Define output path
    output_path = Path(__file__).parent.parent / "models" / "model.onnx"
    
    # Export model
    torch.onnx.export(
        model,                  # model being run
        dummy_input,           # model input
        output_path,           # where to save the model
        export_params=True,    # store the trained parameter weights inside the model file
        opset_version=11,      # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model has been converted to ONNX and saved at: {output_path}")

if __name__ == "__main__":
    convert_to_onnx() 