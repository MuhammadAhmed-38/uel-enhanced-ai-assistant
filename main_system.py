"""
UEL AI System - Main System Module
"""

from datetime import datetime
from typing import Dict, Optional

from config import config, PROFILE_DATA_DIR
from utils import get_logger
from database_manager import DatabaseManager
from data_manager import DataManager
from profile_manager import ProfileManager, UserProfile
from ollama_service import OllamaService
from sentiment_analysis import SentimentAnalysisEngine
from voice_service import VoiceService
from document_verification import DocumentVerificationAI
from course_recommendation import AdvancedCourseRecommendationSystem
from predictive_analytics import PredictiveAnalyticsEngine

# Import interview system if available
try:
    from interview_preparation import EnhancedInterviewSystem
    INTERVIEW_AVAILABLE = True
except ImportError:
    INTERVIEW_AVAILABLE = False


class UELAISystem:
    """Main UEL AI System orchestrating all components"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.UELAISystem")
        self.logger.info("🚀 Initializing UEL AI System...")

        # Initialize core components
        try:
            self.db_manager = DatabaseManager()
            self.data_manager = DataManager()
            self.profile_manager = ProfileManager(self.db_manager, profile_data_dir=PROFILE_DATA_DIR)
            self.logger.info("✅ Core components initialized")
        except Exception as e:
            self.logger.error(f"❌ Core component initialization failed: {e}")
            raise

        # Initialize AI services FIRST, as other components depend on them
        try:
            self.ollama_service = OllamaService()
            self.sentiment_engine = SentimentAnalysisEngine()
            self.document_verifier = DocumentVerificationAI()
            self.voice_service = VoiceService()
            self.logger.info("✅ AI services initialized")
        except Exception as e:
            self.logger.error(f"⚠️ Some AI services failed to initialize: {e}")
            self.ollama_service = None
            self.sentiment_engine = None
            self.document_verifier = None
            self.voice_service = None

        # Initialize interview preparation system if available
        try:
            if INTERVIEW_AVAILABLE:
                self.interview_system = EnhancedInterviewSystem()
                self.logger.info("✅ Interview preparation system initialized")
            else:
                self.interview_system = None
                self.logger.warning("⚠️ Interview preparation system not available")
        except Exception as e:
            self.logger.error(f"⚠️ Interview preparation system initialization failed: {e}")
            self.interview_system = None

        # Initialize ML components
        try:
            self.course_recommender = AdvancedCourseRecommendationSystem(self.data_manager)
            self.predictive_engine = PredictiveAnalyticsEngine(self.data_manager)
            self.logger.info("✅ ML components initialized")
        except Exception as e:
            self.logger.error(f"⚠️ ML component initialization failed: {e}")
            self.course_recommender = None
            self.predictive_engine = None

        # System status
        self.is_ready = True
        self.logger.info("🎉 UEL AI System fully initialized and ready!")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "system_ready": self.is_ready,
            "ollama_available": self.ollama_service.is_available() if self.ollama_service else False,
            "voice_available": self.voice_service.is_available() if self.voice_service else False,
            "ml_ready": self.predictive_engine.models_trained if self.predictive_engine else False,
            "data_loaded": not self.data_manager.courses_df.empty if self.data_manager else False,
            "data_stats": self.data_manager.get_data_stats() if self.data_manager else {},
            "interview_available": self.interview_system is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_user_message(self, message: str, user_profile: UserProfile = None, context: Dict = None) -> Dict:
        """Process user message with full AI pipeline"""
        try:
            # Analyze sentiment
            sentiment_data = {}
            if self.sentiment_engine:
                sentiment_data = self.sentiment_engine.analyze_message_sentiment(message)
            
            # Generate AI response
            ai_response = "I am sorry, my AI model is not available at the moment."
            if self.ollama_service:
                system_prompt = self._build_system_prompt(user_profile, context)
                ai_response = self.ollama_service.generate_response(message, system_prompt)
            
            # Search for relevant information
            search_results = []
            if self.data_manager:
                search_results = self.data_manager.intelligent_search(message)
            
            # Update profile interaction
            if user_profile and self.profile_manager:
                user_profile.add_interaction("chat")
                self.profile_manager.save_profile(user_profile)
            
            return {
                "ai_response": ai_response,
                "sentiment": sentiment_data,
                "search_results": search_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            return {
                "ai_response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_system_prompt(self, user_profile: UserProfile = None, context: Dict = None) -> str:
        """Build system prompt with context"""
        base_prompt = f"""You are an intelligent AI assistant for the University of East London (UEL). 
        You help students with applications, course information, and university services.
        
        Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        University information:
        - Name: University of East London (UEL)
        - Admissions Email: {config.admissions_email}
        - Phone: {config.admissions_phone}
        """
        
        if user_profile:
            base_prompt += f"""
            
        Student context:
        - Name: {user_profile.first_name} {user_profile.last_name}
        - Interest: {user_profile.field_of_interest}
        - Academic Level: {user_profile.academic_level}
        - Country: {user_profile.country}
        """
        
        base_prompt += """
        
        Guidelines:
        - Be helpful, friendly, and professional
        - Provide accurate UEL information
        - Offer specific guidance for applications
        - Ask clarifying questions when needed
        - Always end with how you can further assist
        """
        
        return base_prompt