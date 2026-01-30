import os
import string
import zipfile
import pandas as pd
import requests
from scipy import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


print("\n--- Retraining with Bigrams ---")

# To use both a custom analyzer (text_process) AND ngram_range,
# we need to ensure the CountVectorizer's 'analyzer' parameter is set to 'word'
# and pass a custom 'tokenizer' that includes our text_process logic.
def load_and_get_data():
    """
    Load the SMS Spam Collection dataset, downloading it if necessary.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

    # Check if dataset already exists
    if not os.path.exists('SMSSpamCollection'):
        print("Downloading dataset...")
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall() # Extracts to the current folder
        print("Download complete!")
    else:
        print("Dataset already exists. Skipping download...")
    return pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df = load_and_get_data()

# Let's check the first 5 rows
df.head()
def text_process(mess):
    """
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def custom_tokenizer(text):
    # Apply our text_process function first to clean the text
    clean_words = text_process(text)
    # Return the list of clean words for CountVectorizer to then form ngrams
    return clean_words

# Initialize the vectorizer with our custom tokenizer and ngram_range
bow_transformer_bigram = CountVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2))

# Fit and Transform the data
print("Vectorizing messages with bigrams...")
messages_bow_bigram = bow_transformer_bigram.fit_transform(df['message'])

print('Shape of Sparse Matrix (with bigrams): ', messages_bow_bigram.shape)
print('Amount of Non-Zero occurences (with bigrams): ', messages_bow_bigram.nnz)

# Split data: 80% for training, 20% for testing
X_train_bigram, X_test_bigram, y_train_bigram, y_test_bigram = train_test_split(
    messages_bow_bigram, df['label'], test_size=0.2, random_state=42
)

# Initialize the Model (Naive Bayes)
spam_detect_model_bigram = MultinomialNB()

# Train the Model
print("Training Naive Bayes model with bigram features...")
spam_detect_model_bigram.fit(X_train_bigram, y_train_bigram)

print("Model Training Complete!")

# Make predictions
predictions_bigram = spam_detect_model_bigram.predict(X_test_bigram)

print("\n--- Evaluation with Bigrams ---")
# Check the results
print(classification_report(y_test_bigram, predictions_bigram))

# Visualize Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test_bigram, predictions_bigram), annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Bigrams)')
plt.show()


# Test with custom message using the new bigram model
print("\n--- Real-World Test with Bigrams ---")
input_message_test = ["Congratulations! You've won a $1000 gift card. Call now to claim."]
input_message_test_vector = bow_transformer_bigram.transform(input_message_test)
prediction_test = spam_detect_model_bigram.predict(input_message_test_vector)
print(f"Test message: '{input_message_test[0]}'")
print(f"Prediction (with bigrams): {prediction_test[0]}")

input_message_test_2 = ["Hey, are we still free for a wining lunch tomorrow? I will give your $800 gift.Congratulations"]
input_message_test_vector_2 = bow_transformer_bigram.transform(input_message_test_2)
prediction_test_2 = spam_detect_model_bigram.predict(input_message_test_vector_2)
print(f"Test message: '{input_message_test_2[0]}'")
print(f"Prediction (with bigrams): {prediction_test_2[0]}")
#show which bigram had the most impact
print("\n--- Bigram Impact Analysis ---")

