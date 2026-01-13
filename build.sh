#!/bin/bash

echo "🚀 Building Biology Chatbot on Render..."

# Create necessary directories
mkdir -p static/images

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# If model files don't exist, train a simple model
if [ ! -f "chatbot_model.h5" ]; then
    echo "🧠 Training a simple model..."
    python train_simple.py
fi

echo "✅ Build completed!"