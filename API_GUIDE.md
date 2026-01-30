# API Usage Guide

## Installation

All dependencies are already installed. Just make sure you're in the workspace directory.

## Quick Start

### 1. Train the Model (if not already done)

```bash
python src/main_rf.py
```

This will:

- Download the dataset (if needed)
- Train the Random Forest model
- Save `spam_model.pkl` and `vectorizer.pkl`

### 2. Start the API Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 3. Access the API Documentation

Open your browser and go to:

- **Interactive Docs (Swagger UI):** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

## API Endpoints

### 1. Health Check

```
GET /health
```

Check if the API is running and model is loaded.

**Response:**

```json
{
  "status": "running",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 2. Single Message Prediction

```
POST /predict
```

**Request:**

```json
{
  "message": "Congratulations! You've won a prize!"
}
```

**Response:**

```json
{
  "message": "Congratulations! You've won a prize!",
  "prediction": "spam",
  "confidence": "95.30%",
  "is_spam": true
}
```

### 3. Batch Predictions (Multiple Messages)

```
POST /batch
```

**Request:**

```json
{
  "messages": [
    "Hi, how are you?",
    "Click here to win $1000!",
    "Let's meet tomorrow"
  ]
}
```

**Response:**

```json
{
  "total": 3,
  "predictions": [
    {
      "message": "Hi, how are you?",
      "prediction": "ham",
      "confidence": "98.50%",
      "is_spam": false
    },
    {
      "message": "Click here to win $1000!",
      "prediction": "spam",
      "confidence": "92.15%",
      "is_spam": true
    },
    {
      "message": "Let's meet tomorrow",
      "prediction": "ham",
      "confidence": "99.10%",
      "is_spam": false
    }
  ]
}
```

### 4. Get Model Statistics

```
GET /stats
```

**Response:**

```json
{
  "model": "Random Forest Classifier",
  "features": "Bigrams (1-2 word combinations)",
  "training_data": "SMS Spam Collection Dataset",
  "framework": "scikit-learn",
  "estimators": 100,
  "class_weight": "balanced",
  "documentation": "See /docs for API documentation"
}
```

## Usage Examples

### Using cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "You won a prize!"}'

# Health check
curl "http://localhost:8000/health"

# Statistics
curl "http://localhost:8000/stats"
```

### Using Python

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"message": "Congratulations! You've won!"}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# Batch prediction
response = requests.post(
    f"{API_URL}/batch",
    json={
        "messages": [
            "Hi, how are you?",
            "Click here to claim prize!"
        ]
    }
)
results = response.json()
for pred in results['predictions']:
    print(f"{pred['message']}: {pred['prediction']}")
```

### Using JavaScript/Fetch

```javascript
const API_URL = 'http://localhost:8000';

// Single prediction
async function predictSpam(message) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  const result = await response.json();
  console.log(`Prediction: ${result.prediction}`);
  console.log(`Confidence: ${result.confidence}`);
  return result;
}

// Usage
predictSpam('You won a prize!');
```

### Using Node.js

```javascript
const fetch = require('node-fetch');

const API_URL = 'http://localhost:8000';

async function predictSpam(message) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  return await response.json();
}

// Usage
predictSpam('Click here to win!').then((result) => {
  console.log(result);
});
```

## Web Client

A simple HTML web client is included for testing:

### Generate the HTML Client

```bash
python src/web_client.py
```

This will create `client.html` in the current directory. Open it in your browser to test the API with a user-friendly interface.

## Configuration

### CORS (Cross-Origin Resource Sharing)

The API is currently configured to accept requests from all origins. For production, modify the `CORSMiddleware` in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Port and Host

To run on a different port or host:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 5000
```

## Deployment

### Using Gunicorn (Production)

```bash
pip install gunicorn
gunicorn src.api:app -w 4 -b 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t spam-detector .
docker run -p 8000:8000 spam-detector
```

## Troubleshooting

### "Model not loaded" Error

- Make sure you've run `python src/main_rf.py` first
- Check that `spam_model.pkl` and `vectorizer.pkl` exist in the workspace

### CORS Issues

- Enable CORS for your domain in `api.py`
- Use the web client provided or configure your frontend

### API Not Responding

- Check that the server is running
- Verify the port is not in use: `netstat -ano | findstr :8000` (Windows)
- Try a different port if needed
