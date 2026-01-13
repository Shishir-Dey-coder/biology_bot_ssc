import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import ssl
import os

print("=" * 60)
print("🧬 BANGLADESH CLASS 9-10 BIOLOGY CHATBOT TRAINING")
print("=" * 60)

# Fix SSL for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
print("\n📥 Setting up NLTK...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK ready")
except:
    print("⚠️ Using fallback tokenizer")

# Load Biology intents
print("📖 Loading Biology curriculum...")
try:
    with open('biology_intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    print(f"✅ Loaded {len(intents['intents'])} Biology topics")
except FileNotFoundError:
    print("❌ biology_intents.json not found! Using default...")
    with open('biology_intents.json', 'w', encoding='utf-8') as f:
        json.dump({"intents": []}, f)
    with open('biology_intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

# Process data
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',', '।', '?']

print("🔄 Processing Biology questions...")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize
        word_list = nltk.word_tokenize(pattern.lower())
        # Simple cleaning for Bengali/English mix
        word_list = [w for w in word_list if w not in ignore_chars]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize (English words only)
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"📊 Vocabulary: {len(words)} words")
print(f"📚 Topics: {len(classes)}")
print(f"📝 Training samples: {len(documents)}")

# Save words and classes
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("💾 Saved vocabulary and topics")

# Create training data
training = []
output_empty = [0] * len(classes)

print("🧮 Creating training dataset...")
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words if w.isalpha()]
    
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build model (optimized for 4GB RAM)
print("\n🏗️ Building neural network...")
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(f"🚀 Training on {len(train_x)} Biology questions...")
print("⏳ This may take 1-2 minutes...")

history = model.fit(
    np.array(train_x), 
    np.array(train_y), 
    epochs=200,
    batch_size=5,
    verbose=1
)

# Save model
model.save('chatbot_model.h5')
print("\n💾 Saved: chatbot_model.h5")

# Calculate accuracy
loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y), verbose=0)
print(f"📈 Training Accuracy: {accuracy:.2%}")

print("\n" + "=" * 60)
print("🎓 BIOLOGY CHATBOT TRAINING COMPLETE!")
print("=" * 60)
print("\n📚 Topics covered:")
for i, cls in enumerate(classes, 1):
    print(f"   {i:2d}. {cls.replace('_', ' ').title()}")

print(f"\n📊 Stats:")
print(f"   • Biology Topics: {len(classes)}")
print(f"   • Vocabulary: {len(words)} words")
print(f"   • Training Samples: {len(train_x)}")
print(f"   • Model Accuracy: {accuracy:.2%}")

print("\n🚀 To start the Biology Study Assistant:")
print("   python app.py")
print("\n🌐 Then open: http://localhost:5000")
print("=" * 60)