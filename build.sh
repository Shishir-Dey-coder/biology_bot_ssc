#!/bin/bash

echo "🚀 Building Biology Chatbot on Render..."

# Install dependencies using Poetry
poetry install --no-interaction --no-ansi

# OR using pip (uncomment if Poetry fails)
# pip install --upgrade pip
# pip install -r requirements.txt

# Create necessary directories
mkdir -p static/images

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')" || echo "NLTK download failed, continuing..."

# Check if we need to train model
if [ ! -f "chatbot_model.pkl" ] && [ ! -f "chatbot_model.h5" ]; then
    echo "🧠 No model found, using simple responses..."
    # Create a simple model file
    python -c "
import pickle
data = {'simple': True}
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(data, f)
print('Simple model created')
    "
fi

echo "✅ Build completed!"