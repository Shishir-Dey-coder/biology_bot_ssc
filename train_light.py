import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("🚀 Starting training...")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare training data
patterns = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        cleaned_pattern = ' '.join(words)
        
        patterns.append(cleaned_pattern)
        tags.append(intent['tag'])

print(f"✅ Prepared {len(patterns)} training samples")

# Create model
model = make_pipeline(
    TfidfVectorizer(max_features=100),
    MultinomialNB(alpha=0.1)
)

# Train
print("🧠 Training model...")
model.fit(patterns, tags)

# Save
joblib.dump(model, 'model.pkl')

with open('intents_data.pkl', 'wb') as f:
    pickle.dump(intents, f)

print("✅ Training completed!")
