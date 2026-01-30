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
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
def load_and_get_data():
    """Load the SMS Spam Collection dataset, downloading it if necessary."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    dataset_path = 'SMSSpamCollection'

    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall()
        print("Download complete!")
    else:
        print("Dataset already exists. Skipping download...")
    
    return pd.read_csv(dataset_path, sep='\t', names=['label', 'message'])


def text_process(message):
    """
    Clean text by:
    1. Removing punctuation
    2. Removing stopwords
    3. Return list of clean text words
    """
    # Remove punctuation
    no_punc = [char for char in message if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    # Remove stopwords
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]


def custom_tokenizer(text):
    """Apply text_process function and return clean words for CountVectorizer."""
    return text_process(text)


def visualize_feature_importance(model, feature_names, top_n=20):
    """Visualize feature importance from the trained model."""
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(top_n)
    
    print(f"\n--- Top {top_n} Most Important Features (Random Forest) ---")
    print(feature_importance_df)
    
    plt.figure(figsize=(14, 7))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis', legend=False)
    plt.title(f'Top {top_n} Most Important Features (Random Forest)')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature (Word/Bigram)')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=['ham', 'spam'], average='weighted')
    recall = recall_score(y_test, y_pred, labels=['ham', 'spam'], average='weighted')
    f1 = f1_score(y_test, y_pred, labels=['ham', 'spam'], average='weighted')
    
    print("\n" + "="*50)
    print("--- MODEL EVALUATION ON TEST SET ---")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_model(model, vectorizer, model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
    """Save trained model and vectorizer to disk."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Vectorizer saved to {vectorizer_path}")


def load_model(model_path='spam_model.pkl', vectorizer_path='vectorizer.pkl'):
    """Load trained model and vectorizer from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Vectorizer loaded from {vectorizer_path}")
    return model, vectorizer


def predict_spam(message, model, vectorizer):
    """Predict whether a message is spam or not.
    
    Args:
        message (str): The message to classify
        model: Trained RandomForest model
        vectorizer: Fitted CountVectorizer
    
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
    """Test the model with custom messages."""
    print("\n" + "="*50)
    print("--- Real-World Test with Random Forest ---")
    print("="*50)
    for message in test_messages:
        result = predict_spam(message, model, vectorizer)
        print(f"Message: '{result['message']}'")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']}\n")


# ============== Main Execution ==============
print("\n--- Training with Random Forest ---")

# Load data
df = load_and_get_data()
print(f"\nDataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}\n")

# Create vectorizer with bigrams
bow_transformer_bigram = CountVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2))

# Fit and transform the data
print("Vectorizing messages with bigrams...")
messages_bow_bigram = bow_transformer_bigram.fit_transform(df['message'])

print(f'Shape of Sparse Matrix (with bigrams): {messages_bow_bigram.shape}')
print(f'Amount of Non-Zero occurences (with bigrams): {messages_bow_bigram.nnz}\n')

# Split data: 80% for training, 20% for testing
X_train_bigram, X_test_bigram, y_train_bigram, y_test_bigram = train_test_split(
    messages_bow_bigram, df['label'], test_size=0.2, random_state=42
)

# Initialize and train the Random Forest Model
spam_detect_model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Training Random Forest model with bigram features...")
spam_detect_model_rf.fit(X_train_bigram, y_train_bigram)
print("Training complete!\n")

# Evaluate model on test set
metrics = evaluate_model(spam_detect_model_rf, X_test_bigram, y_test_bigram)

# Get feature names and visualize importance
feature_names_rf = bow_transformer_bigram.get_feature_names_out()
visualize_feature_importance(spam_detect_model_rf, feature_names_rf, top_n=20)

# Save model and vectorizer for production use
save_model(spam_detect_model_rf, bow_transformer_bigram)

# Test with custom messages
test_messages = [
    "Congratulations! You've won a $1000 gift card. Call now to claim.",
    "Hey, are we still free for a winning lunch tomorrow? I will give your $800 gift. Congratulations",
    "Hi, how are you doing today?",
    "URGENT: Click here now to claim your prize!!!"
]
test_model_with_examples(spam_detect_model_rf, bow_transformer_bigram, test_messages)

print("\n" + "="*50)
print("--- PRODUCTION READY ---")
print("="*50)
print("To use the model in production:")
print("\n1. Load the model:")
print("   model, vectorizer = load_model()")
print("\n2. Make predictions:")
print("   result = predict_spam('Your message here', model, vectorizer)")
print("   print(result)")
print("\nModel metrics:")
for metric, value in metrics.items():
    print(f"  {metric.capitalize()}: {value:.4f}")
