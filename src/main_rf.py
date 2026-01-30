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

# Download NLTK stopwords if not already available (used for French)
try:
    stopwords.words('french')
except LookupError:
    nltk.download('stopwords')

def load_and_get_data():
    """Charger le dataset augmenté et inclure les exemples en malagasy si présents.

    Le code tente d'utiliser la colonne `text_mg` dans `data-augmented.csv`. Si elle
    n'existe pas, il recherche `data_mg.csv` à la racine du projet et l'ajoute.
    Retourne un DataFrame avec colonnes: `label` et `message`.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data-augmented.csv')
    mg_local_path = os.path.join(os.path.dirname(__file__), '..', 'data_mg.csv')

    if not os.path.exists(dataset_path):
        print("❌ Dataset CSV non trouvé!")
        print(f"Créez le fichier {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print("Chargement du dataset augmenté...")
    df_all = pd.read_csv(dataset_path)

    frames = []

    # Colonne française si présente
    if 'text_fr' in df_all.columns:
        df_fr = df_all[['labels', 'text_fr']].rename(columns={'labels': 'label', 'text_fr': 'message'})
        df_fr = df_fr.dropna()
        frames.append(df_fr)

    # Si le CSV principal contient déjà du malagasy
    if 'text_mg' in df_all.columns:
        df_mg = df_all[['labels', 'text_mg']].rename(columns={'labels': 'label', 'text_mg': 'message'})
        df_mg = df_mg.dropna()
        frames.append(df_mg)
    else:
        # Sinon, charger le dataset local `data_mg.csv` si disponible
        if os.path.exists(mg_local_path):
            try:
                df_mg_local = pd.read_csv(mg_local_path)
                if 'labels' in df_mg_local.columns and 'text_mg' in df_mg_local.columns:
                    df_mg_local = df_mg_local[['labels', 'text_mg']].rename(columns={'labels': 'label', 'text_mg': 'message'})
                    df_mg_local = df_mg_local.dropna()
                    frames.append(df_mg_local)
                else:
                    print(f"⚠ Le fichier {mg_local_path} doit contenir les colonnes 'labels' et 'text_mg'. Ignoré.")
            except Exception as e:
                print(f"⚠ Impossible de lire {mg_local_path}: {e}")

    if not frames:
        raise ValueError("Aucune donnée valide trouvée dans le dataset. Vérifiez les fichiers CSV.")

    df = pd.concat(frames, ignore_index=True)

    print(f"✓ Dataset combiné chargé: {len(df)} messages")
    print(f"  - Spam: {len(df[df['label'] == 'spam'])}")
    print(f"  - Légitime: {len(df[df['label'] == 'ham'])}")

    return df


def text_process(message):
    """
    Nettoyer le texte:
    - Supprimer la ponctuation
    - Retourner la liste des tokens (sans suppression de stopwords spécifiques)

    Note: pour supporter Malagasy (et d'autres langues), nous n'appliquons pas
    une suppression stricte de stopwords spécifique à une langue ici.
    """
    if not isinstance(message, str):
        message = str(message)

    # Supprimer la ponctuation
    no_punc = [char for char in message if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    # Tokenize basique et filtrage sur la longueur minimale
    tokens = [word for word in no_punc.split() if len(word) > 1]
    return tokens


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
    proba = model.predict_proba(vector)[0]
    # confidence_score: valeur numérique (0.0 - 1.0)
    confidence_score = float(max(proba))

    return {
        'message': message,
        'prediction': prediction,
        'confidence': f"{confidence_score:.2%}",
        'confidence_score': confidence_score
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
print("\n--- Entraînement avec Random Forest (Dataset Augmenté) ---")

# Charger les données
df = load_and_get_data()
print(f"\nFormes du dataset: {df.shape}")
print(f"Premiers 3 lignes:\n{df.head(3)}\n")

# Afficher un exemple de message en français
print("Exemple de message spam en français:")
spam_example = df[df['label'] == 'spam'].iloc[0]['message']
print(f"  {spam_example[:100]}...\n")

print("Exemple de message légitime en français:")
ham_example = df[df['label'] == 'ham'].iloc[0]['message']
print(f"  {ham_example[:100]}...\n")

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

# Tester avec des messages personnalisés en français
test_messages = [
    "Félicitations! Vous avez gagné 1000€! Cliquez ici maintenant!",
    "Bonjour, comment allez-vous ce matin?",
    "Urgent: Veuillez vérifier votre compte bancaire immédiatement",
    "On se voit demain à 19h à la gare"
]
test_model_with_examples(spam_detect_model_rf, bow_transformer_bigram, test_messages)

print("\n" + "="*50)
print("--- PRÊT POUR LA PRODUCTION ---")
print("="*50)
print("Pour utiliser le modèle en production:")
print("\n1. Charger le modèle:")
print("   model, vectorizer = load_model()")
print("\n2. Faire des prédictions:")
print("   result = predict_spam('Votre message en français', model, vectorizer)")
print("   print(result)")
print("\nMétriques du modèle:")
for metric, value in metrics.items():
    print(f"  {metric.capitalize()}: {value:.4f}")
