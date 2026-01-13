from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import ssl

print("=" * 60)
print("🧬 BANGLADESH CLASS 9-10 BIOLOGY STUDY ASSISTANT")
print("=" * 60)

# SSL fix
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

app = Flask(__name__)

# Load Biology model and data
print("\n📦 Loading Biology knowledge base...")

try:
    # Load TensorFlow model
    model = tf.keras.models.load_model('chatbot_model.h5')
    print("✅ Loaded AI model (chatbot_model.h5)")
    
    # Load vocabulary
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    print(f"✅ Loaded vocabulary ({len(words)} words)")
    
    # Load topics
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    print(f"✅ Loaded {len(classes)} Biology topics")
    
    # Load Biology intents
    with open('biology_intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    print("✅ Loaded Biology curriculum data")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Please run train.py first!")
    exit(1)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and tokenize text (handles Bengali/English)"""
    try:
        # Try English tokenization
        words_list = nltk.word_tokenize(text.lower())
        words_list = [lemmatizer.lemmatize(w) for w in words_list if w.isalpha()]
        return words_list
    except:
        # Fallback for Bengali
        return text.lower().split()

def bag_of_words(text):
    """Convert text to bag of words"""
    text_words = clean_text(text)
    bag = [0] * len(words)
    
    for s in text_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                
    return np.array(bag)

def predict_topic(text):
    """Predict which Biology topic the question belongs to"""
    bow = bag_of_words(text)
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    # Set threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    topic_list = []
    for r in results:
        topic_list.append({
            'topic': classes[r[0]],
            'confidence': float(r[1])
        })
    
    return topic_list

def get_biology_answer(text):
    """Get answer from Biology curriculum"""
    topics = predict_topic(text)
    
    if not topics:
        return "I'm not sure about that Biology topic. Could you ask about cells, tissues, human body systems, genetics, or ecology?"
    
    main_topic = topics[0]['topic']
    
    # Find matching intent
    for intent in intents['intents']:
        if intent['tag'] == main_topic:
            # Return random response from that topic
            return random.choice(intent['responses'])
    
    return f"I know about {main_topic.replace('_', ' ')}, but need more specific question."

@app.route('/')
def home():
    """Render Biology chatbot interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle Biology questions"""
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'response': 'Please ask a Biology question!',
                'type': 'error'
            })
        
        # Get Biology answer
        response = get_biology_answer(user_message)
        
        # Log the question
        print(f"📝 Question: {user_message[:50]}...")
        
        return jsonify({
            'response': response,
            'type': 'biology'
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'response': 'Sorry, having trouble with that Biology question. Try again!',
            'type': 'error'
        })

@app.route('/topics')
def get_topics():
    """Return list of Biology topics covered"""
    topics = [cls.replace('_', ' ').title() for cls in classes]
    return jsonify({'topics': topics, 'count': len(topics)})

if __name__ == '__main__':
    print("\n🎯 Biology Topics Available:")
    print("-" * 40)
    for i, topic in enumerate(classes, 1):
        print(f"{i:2d}. {topic.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    print("🌐 Biology Study Assistant is READY!")
    print("=" * 60)
    print("\n💡 Ask questions like:")
    print("   • 'What is photosynthesis?'")
    print("   • 'মাইটোসিস কী?'")
    print("   • 'Explain human digestive system'")
    print("   • 'জৈববৈচিত্র্যের গুরুত্ব কী?'")
    print("\n🚀 Starting server...")
    print("📚 Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)