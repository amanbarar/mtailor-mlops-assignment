# Image Classification Model Deployment

This project deploys an ImageNet-trained classification model on Cerebrium's serverless platform. The model is converted from PyTorch to ONNX format for optimized inference.

## Project Structure

```
mtailor-mlops-assignment/
├── src/                    # Source code
│   ├── model.py           # ONNX model loading and prediction
│   ├── preprocess.py      # Image preprocessing
│   └── app.py            # FastAPI application
├── models/                # Model files
│   ├── pytorch_model.py
│   ├── pytorch_model_weights.pth
│   └── model.onnx        # Converted ONNX model
├── tests/                 # Test files
│   ├── test_model.py     # Local model tests
│   └── test_server.py    # Deployment tests
├── assets/               # Test images
│   ├── n01440764_tench.jpeg
│   └── n01667114_mud_turtle.JPEG
├── Dockerfile           # Custom Docker image
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
echo "CEREBRIUM_API_KEY=your_api_key" > .env
echo "CEREBRIUM_API_URL=your_api_url" >> .env
```

## Model Conversion

Convert the PyTorch model to ONNX format:
```bash
python src/convert_to_onnx.py
```

## Testing

1. Test the local model:
```bash
pytest tests/test_model.py
```

2. Test the deployed model:
```bash
# Test single image
python tests/test_server.py --image_path assets/n01440764_tench.jpeg

# Run all test cases
python tests/test_server.py --run_tests
```

## Deployment

1. Build the Docker image:
```bash
docker build -t mtailor-mlops-assignment .
```

2. Deploy to Cerebrium:
```bash
cerebrium deploy
```

## API Usage

The deployed model exposes a `/predict` endpoint that accepts image files:

```python
import requests

url = "https://api.cortex.cerebrium.ai/v4/p-4419db4c/mtailor-mlops-assignment/predict"
headers = {
    "Authorization": f"Bearer {api_key}"
}

with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, headers=headers, files=files)

result = response.json()
print(result)
```

Response format:
```json
{
    "class_id": 0,
    "confidence": 0.95,
    "predictions": [...],
    "inference_time": 0.1
}
```

## Model Details

- Input: 224x224 RGB image
- Output: 1000 class probabilities (ImageNet classes)
- Preprocessing:
  - Convert to RGB
  - Resize to 224x224
  - Normalize using ImageNet mean/std
- Expected response time: < 3 seconds

## Testing Strategy

1. Local Model Tests:
   - Model initialization
   - Image preprocessing
   - Prediction format
   - Error handling

2. Deployment Tests:
   - API endpoint availability
   - Prediction accuracy
   - Response time monitoring
   - Error handling

## Performance Monitoring

The test suite monitors:
- Prediction accuracy
- Response times
- Inference times
- Error rates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 