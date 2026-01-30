import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import requests
import zipfile
import io
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_nltk_stopwords():
    """
    Load NLTK stopwords, downloading them if necessary.
    """
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

load_nltk_stopwords()

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

# Load the dataset using Pandas
# Note: The file is tab-separated (TSV), not comma-separated (CSV)


# Visualize it
def visualize_data_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title("Distribution of Spam vs Ham")
    plt.show()

def visualize_message_length_distribution(df):
    # Create a new column 'length'
    df['length'] = df['message'].apply(len)

    # Plot the distribution
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='length', hue='label', bins=50, kde=True)
    plt.title("Message Length: Spam vs Ham")
    plt.show()


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

def vectorize_messages(df):
    """
        # Initialize the vectorizer with our custom cleaning function
    """
    bow_transformer = CountVectorizer(analyzer=text_process)

    # Fit and Transform the data (this might take a few seconds)
    print("Vectorizing... this takes a moment.")
    messages_bow = bow_transformer.fit_transform(df['message'])

    print('Shape of Sparse Matrix: ', messages_bow.shape)
    print('Amount of Non-Zero occurences: ', messages_bow.nnz)
    return messages_bow, bow_transformer

def splitting_and_training(messages, df):
    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(messages, df['label'], test_size=0.2, random_state=42)

    # Initialize the Model (Naive Bayes)
    spam_detect_model = MultinomialNB()

    # Train the Model
    spam_detect_model.fit(X_train, y_train)

    print("Model Training Complete!")
    return spam_detect_model, X_test, y_test


def evaluation_metrics(spam_detect_model, X_test, y_test):
    """
        1 Make predictions
        2 Check the results
        3 Visualize Confusion Matrix
    """
    predictions = spam_detect_model.predict(X_test)

    print(classification_report(y_test, predictions))

    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

def real_world_test(bow_transformer, spam_detect_model):
    # Try your own message!
    input_message = [" Call Congratulations! You've won a $1000 gift card.  now to claim."]
    #input_message = ["Hey, are we still free for a wining lunch tomorrow? I will give your $800 gift.Congratulations cause you won"]

    # We must transform the input just like we did the training data
    input_vector = bow_transformer.transform(input_message)
    prediction = spam_detect_model.predict(input_vector)

    print(f"Prediction: {prediction[0]}")


df = load_and_get_data()

# Let's check the first 5 rows
df.head()

print(df['label'].value_counts())


visualize_data_distribution(df)
visualize_message_length_distribution(df)

# Let's test it on a random message
sample_message = "Hello! This is a sample message... notice the punctuation is gone?"
print(text_process(sample_message))

messages_bow, bow_transformers = vectorize_messages(df)
spam_detect_model, X_test, y_test = splitting_and_training(messages_bow, df)
evaluation_metrics(spam_detect_model, X_test, y_test)
real_world_test(bow_transformers, spam_detect_model)