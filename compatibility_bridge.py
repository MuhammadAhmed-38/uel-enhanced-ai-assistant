"""
Compatibility Bridge for Legacy Imports
This file maintains compatibility with existing main.py imports
while using the new modular architecture underneath.
"""

# Re-export everything from the new modular system
from main_system import UELAISystem
from profile_manager import UserProfile, ProfileManager
from course_recommendation import AdvancedCourseRecommendationSystem
from predictive_analytics import PredictiveAnalyticsEngine
from sentiment_analysis import SentimentAnalysisEngine
from voice_service import VoiceService
from document_verification import DocumentVerificationAI
from ollama_service import OllamaService
from data_manager import DataManager
from database_manager import DatabaseManager
from config import config, research_config, PROFILE_DATA_DIR
from utils import *

# Legacy class names that might be referenced
try:
    from interview_preparation import EnhancedInterviewSystem
except ImportError:
    EnhancedInterviewSystem = None

# For any legacy unified imports
__all__ = [
    'UELAISystem',
    'UserProfile', 
    'ProfileManager',
    'AdvancedCourseRecommendationSystem',
    'PredictiveAnalyticsEngine',
    'SentimentAnalysisEngine',
    'VoiceService',
    'DocumentVerificationAI',
    'OllamaService',
    'DataManager',
    'DatabaseManager',
    'EnhancedInterviewSystem',
    'config',
    'research_config',
    'PROFILE_DATA_DIR'
]

# Alias for legacy code compatibility
unified_uel_ai_system = UELAISystem