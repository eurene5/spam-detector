"""
Spam Detection API
FastAPI-based REST API for the Random Forest spam detection model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from main_rf import load_model, predict_spam

# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="API for detecting spam messages using Random Forest",
    version="1.0.0"
)

# Add CORS middleware to allow requests from web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer at startup
model = None
vectorizer = None


@app.on_event("startup")
async def startup_event():
    """Load model and vectorizer when API starts."""
    global model, vectorizer
    try:
        model, vectorizer = load_model()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model files not found. Please run main_rf.py first to train the model.")
        raise


# Define request/response models
class PredictionRequest(BaseModel):
    """Request model for spam prediction."""
    message: str = Field(..., description="The message to classify", min_length=1)
    
    class Config:
        example = {
            "message": "Congratulations! You've won a prize!"
        }


class PredictionResponse(BaseModel):
    """Response model for spam prediction."""
    message: str = Field(..., description="The input message")
    prediction: str = Field(..., description="Classification: 'ham' or 'spam'")
    confidence: str = Field(..., description="Confidence percentage")
    is_spam: bool = Field(..., description="True if spam, False if ham")
    
    class Config:
        example = {
            "message": "Hi, how are you?",
            "prediction": "ham",
            "confidence": "98.50%",
            "is_spam": False
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    messages: list[str] = Field(..., description="List of messages to classify")
    
    class Config:
        example = {
            "messages": [
                "Hi, how are you?",
                "You won a prize! Click here!"
            ]
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    total: int
    predictions: list[PredictionResponse]


# API Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Spam Detection API",
        "version": "1.0.0",
        "description": "Detect spam messages using Machine Learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch": "/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="running" if model else "model not loaded",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict whether a message is spam or not.
    
    Parameters:
    - **message**: The message to classify
    
    Returns:
    - **prediction**: 'ham' or 'spam'
    - **confidence**: Confidence percentage
    - **is_spam**: Boolean flag
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        result = predict_spam(request.message, model, vectorizer)
        
        return PredictionResponse(
            message=result['message'],
            prediction=result['prediction'],
            confidence=result['confidence'],
            is_spam=result['prediction'].lower() == 'spam'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict spam for multiple messages at once.
    
    Parameters:
    - **messages**: List of messages to classify
    
    Returns:
    - **total**: Number of messages processed
    - **predictions**: List of predictions for each message
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    if len(request.messages) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 messages per request")
    
    try:
        predictions = []
        for message in request.messages:
            result = predict_spam(message, model, vectorizer)
            predictions.append(
                PredictionResponse(
                    message=result['message'],
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    is_spam=result['prediction'].lower() == 'spam'
                )
            )
        
        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get model statistics and information."""
    return {
        "model": "Random Forest Classifier",
        "features": "Bigrams (1-2 word combinations)",
        "training_data": "SMS Spam Collection Dataset",
        "framework": "scikit-learn",
        "estimators": 100,
        "class_weight": "balanced",
        "documentation": "See /docs for API documentation"
    }


# Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
