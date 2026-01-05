"""
UEL AI System - Sentiment Analysis Module
"""

from datetime import datetime
from typing import Dict, List, Tuple

from utils import get_logger

# Try to import optional libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class SentimentAnalysisEngine:
    """Enhanced sentiment analysis"""
    
    def __init__(self):
        self.sentiment_history = []
        self.logger = get_logger(f"{__name__}.SentimentAnalysisEngine")
    
    def analyze_message_sentiment(self, message: str) -> Dict:
        """Analyze sentiment of message"""
        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(message)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            else:
                # Fallback simple sentiment analysis
                polarity, subjectivity = self._simple_sentiment_analysis(message)
            
            sentiment_label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
            
            emotions = self._detect_emotions(message)
            urgency = self._detect_urgency(message)
            
            sentiment_data = {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment_label,
                "emotions": emotions,
                "urgency": urgency,
                "timestamp": datetime.now().isoformat()
            }
            
            self.sentiment_history.append(sentiment_data)
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment": "neutral", "polarity": 0.0, "error": str(e)}
    
    def _simple_sentiment_analysis(self, message: str) -> Tuple[float, float]:
        """Simple fallback sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'happy', 'pleased', 'excited']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'frustrated', 'disappointed']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        total_words = len(message.split())
        if total_words == 0:
            return 0.0, 0.0
        
        polarity = (positive_count - negative_count) / total_words
        subjectivity = (positive_count + negative_count) / total_words
        
        return polarity, subjectivity
    
    def _detect_emotions(self, message: str) -> List[str]:
        """Detect emotions in message"""
        emotion_keywords = {
            'anxiety': ['worried', 'anxious', 'nervous', 'stressed', 'concerned', 'fear'],
            'excitement': ['excited', 'thrilled', 'eager', 'enthusiastic', 'happy'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'upset', 'angry'],
            'confusion': ['confused', 'unclear', "don't understand", 'puzzled', 'lost'],
            'satisfaction': ['satisfied', 'pleased', 'glad', 'thankful', 'grateful']
        }
        
        detected_emotions = []
        message_lower = message.lower()
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    def _detect_urgency(self, message: str) -> str:
        """Detect urgency level"""
        urgent_indicators = ['urgent', 'asap', 'immediately', 'emergency', 'deadline', 'hurry']
        high_indicators = ['soon', 'quickly', 'fast', 'rush', 'quick']
        
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in urgent_indicators):
            return "urgent"
        elif any(indicator in message_lower for indicator in high_indicators):
            return "high"
        else:
            return "normal"