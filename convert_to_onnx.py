import torch
import torch.nn as nn
from pytorch_model import Classifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_onnx(model_path: str, onnx_path: str, input_shape: tuple = (1, 3, 224, 224)):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to the PyTorch model weights
        onnx_path (str): Path to save the ONNX model
        input_shape (tuple): Input shape for the model (batch_size, channels, height, width)
    """
    try:

        model = Classifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Successfully converted model to ONNX format. Saved at: {onnx_path}")
        
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    parser.add_argument('--model_path', type=str, required=True, help='Path to PyTorch model weights')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to save ONNX model')
    
    args = parser.parse_args()
    
    convert_to_onnx(args.model_path, args.onnx_path) 