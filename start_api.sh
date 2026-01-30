#!/bin/bash
# Script pour démarrer l'API et ouvrir le client web

echo "================================"
echo "🚀 Démarrage du Détecteur de Spam"
echo "================================"
echo ""

# Vérifier si le modèle existe
if [ ! -f "spam_model.pkl" ] || [ ! -f "vectorizer.pkl" ]; then
    echo "⚠️  Le modèle n'existe pas. Entraînement du modèle..."
    python src/main_rf.py
    echo ""
fi

# Démarrer l'API
echo "🔧 Démarrage de l'API..."
echo "📡 L'API sera disponible sur: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo ""
echo "⏱️  Appuyez sur Ctrl+C pour arrêter"
echo ""

python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
