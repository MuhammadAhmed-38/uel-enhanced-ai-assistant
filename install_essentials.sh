#!/bin/bash

echo "🚀 Installing essential packages for Enhanced Interview System..."

# Install core packages that usually work
pip install numpy pandas requests sqlite3

# Try installing video/audio packages one by one
echo "📹 Installing OpenCV..."
pip install opencv-python || echo "❌ OpenCV failed"

echo "🎯 Installing MediaPipe..."
pip install mediapipe || pip install mediapipe-silicon || echo "❌ MediaPipe failed"

echo "🎵 Installing audio packages..."
pip install librosa || echo "❌ Librosa failed"
pip install SpeechRecognition || echo "❌ SpeechRecognition failed"
pip install pyttsx3 || echo "❌ pyttsx3 failed"

echo "🌐 Installing optional packages..."
pip install webrtcvad || echo "❌ WebRTC VAD failed (optional)"
pip install face-recognition || echo "❌ Face recognition failed (optional)"

echo "✅ Installation completed!"
echo "📝 Some packages may have failed - the system will work with fallbacks"
