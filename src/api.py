"""
Détecteur de Spam API
API REST basée sur FastAPI pour le modèle Random Forest de détection de spam.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    title="API Détection de Spam",
    description="API pour détecter les messages spam en utilisant Random Forest",
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

# Monter le dossier public pour servir le frontend (index.html)
public_dir = os.path.join(os.path.dirname(__file__), '..', 'public')
if os.path.isdir(public_dir):
    app.mount('/public', StaticFiles(directory=public_dir), name='public')


@app.on_event("startup")
async def startup_event():
    """Charger le modèle et le vectorizer au démarrage de l'API."""
    global model, vectorizer
    try:
        model, vectorizer = load_model()
        print("✓ Modèle chargé avec succès")
    except FileNotFoundError:
        print("❌ Fichiers du modèle non trouvés. Veuillez exécuter main_rf.py d'abord pour entraîner le modèle.")
        raise


# Définir les modèles de requête/réponse
class PredictionRequest(BaseModel):
    """Modèle de requête pour la prédiction de spam."""
    message: str = Field(..., description="Le message à classer", min_length=1)
    
    class Config:
        example = {
            "message": "Félicitations! Vous avez gagné un prix!"
        }


class PredictionResponse(BaseModel):
    """Modèle de réponse pour la prédiction de spam."""
    message: str = Field(..., description="Le message d'entrée")
    prediction: str = Field(..., description="Classification: 'légitime' ou 'spam'")
    confidence: str = Field(..., description="Pourcentage de confiance")
    confidence_score: float = Field(..., description="Score de confiance (0.0 - 1.0)")
    is_spam: bool = Field(..., description="Vrai si spam, Faux si légitime")
    
    class Config:
        example = {
            "message": "Bonjour, comment allez-vous?",
            "prediction": "légitime",
            "confidence": "98.50%",
            "confidence_score": 0.985,
            "is_spam": False
        }


class HealthResponse(BaseModel):
    """Modèle de réponse pour la vérification de santé."""
    status: str
    model_loaded: bool
    version: str


class BatchPredictionRequest(BaseModel):
    """Modèle de requête pour les prédictions par lot."""
    messages: list[str] = Field(..., description="Liste des messages à classer")
    
    class Config:
        example = {
            "messages": [
                "Bonjour, comment allez-vous?",
                "Vous avez gagné un prix! Cliquez ici!"
            ]
        }


class BatchPredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions par lot."""
    total: int
    predictions: list[PredictionResponse]


# Points d'accès de l'API
@app.get("/", include_in_schema=False)
async def root():
    """Servir le frontend `public/index.html` si présent, sinon retourner les infos de l'API."""
    index_path = os.path.join(public_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    return {
        "nom": "API Détection de Spam",
        "version": "1.0.0",
        "description": "Détectez les messages spam en utilisant le Machine Learning",
        "endpoints": {
            "sante": "/health",
            "predire": "/predict",
            "lot": "/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Santé"])
async def health_check():
    """Vérifier la santé de l'API et le statut du modèle."""
    return HealthResponse(
        status="en fonctionnement" if model else "modèle non chargé",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(request: PredictionRequest):
    """
    Prédire si un message est spam ou non.
    
    Paramètres:
    - **message**: Le message à classer
    
    Retours:
    - **prediction**: 'légitime' ou 'spam'
    - **confidence**: Pourcentage de confiance
    - **is_spam**: Drapeau booléen
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Service indisponible.")
    
    try:
        result = predict_spam(request.message, model, vectorizer)
        
        # Traduction des prédictions
        prediction_fr = "spam" if result['prediction'].lower() == 'spam' else "légitime"
        
        return PredictionResponse(
            message=result['message'],
            prediction=prediction_fr,
            confidence=result['confidence'],
            confidence_score=result.get('confidence_score', 0.0),
            is_spam=result['prediction'].lower() == 'spam'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/batch", response_model=BatchPredictionResponse, tags=["Prédiction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Prédire le spam pour plusieurs messages à la fois.
    
    Paramètres:
    - **messages**: Liste des messages à classer
    
    Retours:
    - **total**: Nombre de messages traités
    - **predictions**: Liste des prédictions pour chaque message
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Service indisponible.")
    
    if len(request.messages) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 messages par requête")
    
    try:
        predictions = []
        for message in request.messages:
            result = predict_spam(message, model, vectorizer)
            prediction_fr = "spam" if result['prediction'].lower() == 'spam' else "légitime"
            
            predictions.append(
                PredictionResponse(
                    message=result['message'],
                    prediction=prediction_fr,
                    confidence=result['confidence'],
                    confidence_score=result.get('confidence_score', 0.0),
                    is_spam=result['prediction'].lower() == 'spam'
                )
            )
        
        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction par lot: {str(e)}")


@app.get("/stats", tags=["Informations"])
async def get_stats():
    """Obtenir les statistiques et informations du modèle."""
    return {
        "modèle": "Classificateur Random Forest",
        "features": "Bigrammes (combinaisons de 1-2 mots)",
        "données_entrainement": "Dataset SMS Spam Collection",
        "framework": "scikit-learn",
        "estimateurs": 100,
        "poids_classe": "équilibré",
        "documentation": "Voir /docs pour la documentation de l'API"
    }


# Run with: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    # Utiliser la variable d'environnement PORT si présente (plateformes cloud comme Render)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
