import os
import sys
from flask import Flask, render_template, request, jsonify
import json
import random
import pickle
import re

app = Flask(__name__)

# Try to import optional dependencies
try:
    import nltk
    NLTK_AVAILABLE = True
    # Try to download data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available, using simple mode")

try:
    import joblib
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Machine learning libraries not available")

# Load biology data
try:
    with open('biology_intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    print("✅ Loaded biology knowledge base")
except FileNotFoundError:
    print("❌ biology_intents.json not found")
    intents = {"intents": []}

# Simple keyword responses for fallback
KEYWORD_RESPONSES = {
    'cell': 'কোষ হল জীবনের মৌলিক গাঠনিক ও কার্যকরী একক। সকল জীব কোষ দ্বারা গঠিত।',
    'photosynthesis': 'সালোকসংশ্লেষণ: সবুজ উদ্ভিদ সূর্যালোক, পানি ও কার্বন ডাই-অক্সাইড ব্যবহার করে শর্করা তৈরি করে।',
    'mitosis': 'মাইটোসিস হল সমবিভাজন যা দুইটি অভিন্ন কোষ সৃষ্টি করে।',
    'respiration': 'শ্বসন: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + শক্তি',
    'genetics': 'জিনতত্ত্ব বংশগতি ও প্রকরণের অধ্যয়ন। মেন্ডেল জিনতত্ত্বের জনক।',
    'ecology': 'বাস্তুবিদ্যা জীব ও পরিবেশের পারস্পরিক সম্পর্ক নিয়ে আলোচনা করে।',
    'tissue': 'টিস্যু হল একই গঠন ও কার্য সম্পাদনকারী কোষের সমষ্টি।',
    'hello': 'আসসালামু আলাইকুম! আমি আপনার Biology Study Assistant। কীভাবে সাহায্য করতে পারি?',
    'hi': 'Hello! Ask me any Biology question from Class 9-10 syllabus.',
    'help': 'আমাকে Biology সম্পর্কিত প্রশ্ন করুন! যেমন: কোষ কী? সালোকসংশ্লেষণ কী? মানব হৃদপিণ্ডের গঠন ইত্যাদি।'
}

def simple_tokenize(text):
    """Simple tokenizer without NLTK"""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def get_keyword_response(user_input):
    """Get response based on keywords"""
    user_input = user_input.lower()
    
    for keyword, response in KEYWORD_RESPONSES.items():
        if keyword in user_input:
            return response
    
    # Check in intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in user_input:
                return random.choice(intent['responses'])
    
    return "আমি Biology সম্পর্কিত প্রশ্নের উত্তর দিতে পারি। 'কোষ কী?' বা 'সালোকসংশ্লেষণ কী?' বা 'মানব পরিপাক তন্ত্র বর্ণনা করুন' জিজ্ঞাসা করুন।"

def load_model():
    """Try to load trained model"""
    if not ML_AVAILABLE:
        return None
    
    try:
        if os.path.exists('chatbot_model.pkl'):
            model = joblib.load('chatbot_model.pkl')
            print("✅ Loaded trained model")
            return model
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
    
    return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please type a Biology question!'})
        
        # Get response
        response = get_keyword_response(user_message)
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'response': 'দুঃখিত, সমস্যা হয়েছে! দয়া করে আবার চেষ্টা করুন।',
            'success': False
        })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'biology-chatbot',
        'nltk': NLTK_AVAILABLE,
        'ml': ML_AVAILABLE,
        'intents': len(intents['intents']) if intents else 0
    })

@app.route('/topics')
def topics():
    """Return available topics"""
    topics_list = []
    if intents and 'intents' in intents:
        topics_list = [intent['tag'].replace('_', ' ').title() for intent in intents['intents']]
    
    return jsonify({
        'topics': topics_list,
        'count': len(topics_list)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Biology Chatbot on port {port}")
    print(f"📚 Available topics: {len(intents['intents']) if intents else 0}")
    app.run(host='0.0.0.0', port=port, debug=False)