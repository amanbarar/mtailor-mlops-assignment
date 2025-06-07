# Image Classification Model Deployment

This project is based on MTailor Assignment which implements a serverless deployment of an image classification model using Cerebrium. The model is trained on the ImageNet dataset and can classify images into 1000 different classes.

## Project Structure

```
mtailor-mlops-assignment/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── app.py                   # FastAPI application
├── model.py                 # ONNX model implementation
├── convert_to_onnx.py       # PyTorch to ONNX conversion
├── test.py                  # Local testing
├── test_server.py           # Deployment testing
└── pytorch_model.py         # Original PyTorch model
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   conda create -n mtailor_assignment python=3.10
   conda activate mtailor_assignment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Convert PyTorch model to ONNX:
   ```bash
   python convert_to_onnx.py --model_path pytorch_model_weights.pth --onnx_path model.onnx
   ```

## Local Testing

Run the test suite:
```bash
pytest test.py
```


## Testing Deployment

1. Test single image:
   ```bash
   python test_server.py --image_path path/to/image.jpg
   ```

2. Run all test cases:
   ```bash
   python test_server.py --run_tests
   ```

## API Endpoints

- `POST /predict`: Upload an image for classification
  - Input: Image file
  - Output: JSON with class ID, confidence, and response time

- `GET /health`: Health check endpoint
  - Output: Status of the service

## Model Details

- Input: RGB image (224x224)
- Output: Class probabilities (1000 classes)
- Preprocessing:
  - Resize to 224x224
  - Normalize with ImageNet mean/std
  - Convert to NCHW format

## Performance

- Response time: < 3 seconds
- GPU acceleration enabled
- Batch size: 1


## License

This project is licensed under the MIT License - see the LICENSE file for details. 