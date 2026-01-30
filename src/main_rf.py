import os
import string
import io
import zipfile

import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Download NLTK stopwords if not already available
try:
    stopwords.words('french')
except LookupError:
    nltk.download('stopwords')

def load_and_get_data():
    """Charger le dataset SMS Spam Collection en français."""
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'SMSSpamCollectionFR')
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset français non trouvé!")
        print(f"Créez le fichier {dataset_path} avec les messages de spam/ham")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print("Chargement du dataset français...")
    return pd.read_csv(dataset_path, sep='\t', names=['label', 'message'])


def text_process(message):
    """
    Nettoyer le texte:
    1. Supprimer la ponctuation
    2. Supprimer les stopwords français
    3. Retourner la liste des mots nettoyés
    """
    # Supprimer la ponctuation
    no_punc = [char for char in message if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    # Supprimer les stopwords français
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('french')]


def custom_tokenizer(text):
    """Appliquer la fonction text_process et retourner les mots nettoyés pour CountVectorizer."""
    return text_process(text)


def visualize_feature_importance(model, feature_names, top_n=20):
    """Visualiser l'importance des features du modèle entraîné."""
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(top_n)
    
    print(f"\n--- Top {top_n} Features les plus importants (Random Forest) ---")
    print(feature_importance_df)
    
    # plt.figure(figsize=(14, 7))
    # sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', legend=False)
    # plt.title(f'Top {top_n} Features les plus importants (Random Forest)')
    # plt.xlabel('Importance du Feature')
    # plt.ylabel('Feature (Mot/Bigramme)')
    # plt.tight_layout()
    # plt.show()


def evaluate_model(model, X_test, y_test):
    """Évaluer les performances du modèle sur l'ensemble de test."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=['ham', 'spam'], average='weighted')
    recall = recall_score(y_test, y_pred, labels=['ham', 'spam'], average='weighted')
    f1 = f1_score(y_test, y_pred, labels=['ham', 'spam'], average='weighted')
    
    print("\n" + "="*50)
    print("--- ÉVALUATION DU MODÈLE SUR L'ENSEMBLE DE TEST ---")
    print("="*50)
    print(f"Précision:  {accuracy:.4f}")
    print(f"Exactitude: {precision:.4f}")
    print(f"Rappel:     {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}\n")
    
    print("--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Matrice de Confusion ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_model(model, vectorizer, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
    """Sauvegarder le modèle entraîné et le vectorizer sur le disque."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Modèle sauvegardé dans {model_path}")
    print(f"✓ Vectorizer sauvegardé dans {vectorizer_path}")


def load_model(model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
    """Charger le modèle entraîné et le vectorizer à partir du disque."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✓ Modèle chargé depuis {model_path}")
    print(f"✓ Vectorizer chargé depuis {vectorizer_path}")
    return model, vectorizer


def predict_spam(message, model, vectorizer):
    """Prédire si un message est du spam ou non.
    
    Args:
        message (str): Le message à classer
        model: Modèle Random Forest entraîné
        vectorizer: CountVectorizer ajusté
    
    Returns:
        dict: {message, prediction, confidence}
    """
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])
    
    return {
        'message': message,
        'prediction': prediction,
        'confidence': f"{confidence:.2%}"
    }


def test_model_with_examples(model, vectorizer, test_messages):
    """Tester le modèle avec des messages personnalisés."""
    print("\n" + "="*50)
    print("--- Test Real-World avec Random Forest ---")
    print("="*50)
    for message in test_messages:
        result = predict_spam(message, model, vectorizer)
        print(f"Message: '{result['message']}'")
        print(f"Prédiction: {result['prediction'].upper()}")
        print(f"Confiance: {result['confidence']}\n")


# ============== Exécution Principale ==============
print("\n--- Entraînement avec Random Forest ---")

# Charger les données
df = load_and_get_data()
print(f"\nFormes du dataset: {df.shape}")
print(f"Premiers 5 lignes:\n{df.head()}\n")

# Créer le vectorizer avec des bigrammes
bow_transformer_bigram = CountVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2))

# Transformer les données
print("Vectorisation des messages avec bigrammes...")
messages_bow_bigram = bow_transformer_bigram.fit_transform(df['message'])

print(f'Forme de la Matrice Creuse (avec bigrammes): {messages_bow_bigram.shape}')
print(f'Nombre d\'occurrences non-nulles (avec bigrammes): {messages_bow_bigram.nnz}\n')

# Diviser les données: 80% entraînement, 20% test
X_train_bigram, X_test_bigram, y_train_bigram, y_test_bigram = train_test_split(
    messages_bow_bigram, df['label'], test_size=0.2, random_state=42
)

# Initialiser et entraîner le modèle Random Forest
spam_detect_model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Entraînement du modèle Random Forest avec les features bigrammes...")
spam_detect_model_rf.fit(X_train_bigram, y_train_bigram)
print("Entraînement terminé!\n")

# Évaluer le modèle sur l'ensemble de test
metrics = evaluate_model(spam_detect_model_rf, X_test_bigram, y_test_bigram)

# Obtenir les noms des features et visualiser leur importance
feature_names_rf = bow_transformer_bigram.get_feature_names_out()
visualize_feature_importance(spam_detect_model_rf, feature_names_rf, top_n=20)

# Sauvegarder le modèle et le vectorizer pour la production
save_model(spam_detect_model_rf, bow_transformer_bigram)

# Tester avec des messages personnalisés
test_messages = [
    "Félicitations! Vous avez gagné 1000€!",
    "Bonjour, comment allez-vous?",
    "Cliquez ici pour réclamer votre prix maintenant!",
    "On se voit demain?"
]
test_model_with_examples(spam_detect_model_rf, bow_transformer_bigram, test_messages)

print("\n" + "="*50)
print("--- PRÊT POUR LA PRODUCTION ---")
print("="*50)
print("Pour utiliser le modèle en production:")
print("\n1. Charger le modèle:")
print("   model, vectorizer = load_model()")
print("\n2. Faire des prédictions:")
print("   result = predict_spam('Votre message ici', model, vectorizer)")
print("   print(result)")
print("\nMétriques du modèle:")
for metric, value in metrics.items():
    print(f"  {metric.capitalize()}: {value:.4f}")
