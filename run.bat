@echo off
echo 🤖 Starting AI Chatbot Setup...
echo Step 1: Training the model...
python train_light.py

echo.
echo Step 2: Starting the server...
echo.
echo 🌐 Open: http://localhost:5000
echo 📱 Press Ctrl+C to stop
echo.
python app.py
pause
