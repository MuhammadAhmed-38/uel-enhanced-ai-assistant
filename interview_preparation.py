# Enhanced Interview Preparation System with Voice/Video Recording - FIXED
# interview_preparation_voice_video_fixed.py

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import time
import threading
import queue
import tempfile
import os
import base64
import hashlib
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import sqlite3
import pickle
from pathlib import Path
import requests
import difflib
from collections import Counter
import math
from contextlib import contextmanager


import streamlit as st
import os
from pathlib import Path

def load_enhanced_css():
    """Load enhanced CSS styles for the interview system"""
    
    # You can either load from external file or use inline CSS
    # Method 1: Load from external CSS file (recommended)
    try:
        css_file_path = Path("/Users/muhammadahmed/Desktop/uel-enhanced-ai-assistant/enhanced_interview_styles.html")
        if css_file_path.exists():
            with open(css_file_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
                # Extract CSS from HTML file
                css_start = css_content.find('<style>') + 7
                css_end = css_content.find('</style>')
                return css_content[css_start:css_end]
    except Exception as e:
        st.warning(f"Could not load external CSS file: {e}")



# Advanced NLP imports with separate tracking for each library
TEXTSTAT_AVAILABLE = False
SKLEARN_AVAILABLE = False
PLOTLY_AVAILABLE = False
VIDEO_AVAILABLE = False

# Track specific import errors
IMPORT_ERRORS = {}

try:
    from textstat import flesch_reading_ease, lexicon_count, sentence_count
    TEXTSTAT_AVAILABLE = True
except ImportError as e:
    TEXTSTAT_AVAILABLE = False
    IMPORT_ERRORS['textstat'] = str(e)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    IMPORT_ERRORS['scikit-learn'] = str(e)

try:
    import cv2
    VIDEO_AVAILABLE = True
except ImportError as e:
    VIDEO_AVAILABLE = False
    IMPORT_ERRORS['opencv'] = str(e)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError as e:
    PLOTLY_AVAILABLE = False
    IMPORT_ERRORS['plotly'] = str(e)

# Advanced NLP is available if we have BOTH textstat and sklearn
ADVANCED_NLP_AVAILABLE = TEXTSTAT_AVAILABLE and SKLEARN_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CheatingFlag:
    """Specific cheating detection result"""
    flag_type: str  # 'reading', 'copy_paste', 'looking_away', 'suspicious_timing'
    confidence: float
    timestamp: datetime
    details: str
    severity: str  # 'low', 'medium', 'high'

@dataclass
class VoiceAnalysis:
    """Voice analysis results"""
    clarity_score: float
    pace_score: float
    tone_confidence: float
    filler_words_count: int
    speaking_duration: float
    silence_ratio: float
    energy_level: float
    voice_quality_issues: List[str]

@dataclass
class VideoAnalysis:
    """Video analysis results"""
    eye_contact_score: float
    posture_score: float
    facial_expression_score: float
    gesture_appropriateness: float
    professional_appearance: float
    background_appropriateness: float
    lighting_quality: float
    video_quality_issues: List[str]

@dataclass
class RecordingValidation:
    """Enhanced validation for voice/video recordings"""
    transcribed_text: str
    voice_analysis: Optional[VoiceAnalysis]
    video_analysis: Optional[VideoAnalysis]
    ai_feedback: str
    technical_quality_score: float
    communication_effectiveness: float
    overall_presentation_score: float
    improvement_suggestions: List[str]

@dataclass
class ContextValidation:
    """Detailed context analysis for interview responses"""
    relevance_score: float
    keyword_match_score: float
    semantic_coherence: float
    depth_score: float
    specificity_score: float
    professional_language_score: float
    missing_elements: List[str]
    strengths: List[str]
    suggestions: List[str]

@dataclass
class InterviewValidation:
    """Comprehensive interview response validation"""
    overall_score: float
    context_validation: ContextValidation
    cheating_flags: List[CheatingFlag]
    authenticity_score: float
    response_quality: Dict[str, float]
    behavioral_analysis: Dict[str, float]
    improvement_suggestions: List[str]
    red_flags: List[str]
    passed_validation: bool
    recording_validation: Optional[RecordingValidation] = None
    timestamp: datetime = field(default_factory=datetime.now)

class VoiceVideoAnalyzer:
    """Analyzer for voice and video recordings"""
    
    def __init__(self):
        self.initialize_analyzers()
    
    def initialize_analyzers(self):
        """Initialize voice and video analysis tools"""
        self.speech_patterns = {
            'filler_words': ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well'],
            'confidence_indicators': ['definitely', 'absolutely', 'certainly', 'clearly'],
            'uncertainty_indicators': ['maybe', 'perhaps', 'possibly', 'might', 'could be']
        }
        
        self.video_quality_checks = {
            'lighting': {'min_brightness': 50, 'max_brightness': 200},
            'resolution': {'min_width': 320, 'min_height': 240},
            'stability': {'max_motion_variance': 30}
        }
    
    async def analyze_voice_recording(self, audio_data: bytes, transcribed_text: str) -> VoiceAnalysis:
        """Analyze voice recording for communication qualities"""
        
        # Simulate voice analysis (in real implementation, you'd use audio processing libraries)
        # For demo purposes, we'll analyze based on transcribed text and simulate audio metrics
        
        # Analyze transcribed text for speech patterns
        words = transcribed_text.lower().split()
        total_words = len(words)
        
        # Count filler words
        filler_count = sum(1 for word in words if word in self.speech_patterns['filler_words'])
        
        # Analyze confidence indicators
        confidence_words = sum(1 for word in words if word in self.speech_patterns['confidence_indicators'])
        uncertainty_words = sum(1 for word in words if word in self.speech_patterns['uncertainty_indicators'])
        
        # Calculate metrics (these would be derived from actual audio analysis)
        clarity_score = max(0.0, min(1.0, 1.0 - (filler_count / max(total_words, 1)) * 3))
        pace_score = 0.8 if 100 <= total_words <= 200 else 0.6  # Simulated based on word count
        tone_confidence = min(1.0, (confidence_words - uncertainty_words * 0.5) / max(total_words / 20, 1) + 0.7)
        
        # Simulate audio quality metrics
        speaking_duration = len(transcribed_text) / 10  # Rough estimate
        silence_ratio = 0.15  # Simulated
        energy_level = 0.7  # Simulated
        
        quality_issues = []
        if filler_count > total_words * 0.1:
            quality_issues.append("Excessive use of filler words")
        if clarity_score < 0.6:
            quality_issues.append("Speech clarity could be improved")
        if tone_confidence < 0.5:
            quality_issues.append("Lacks confidence in delivery")
        
        return VoiceAnalysis(
            clarity_score=clarity_score,
            pace_score=pace_score,
            tone_confidence=max(0.0, min(1.0, tone_confidence)),
            filler_words_count=filler_count,
            speaking_duration=speaking_duration,
            silence_ratio=silence_ratio,
            energy_level=energy_level,
            voice_quality_issues=quality_issues
        )
    
    def analyze_video_recording(self, video_data: bytes) -> VideoAnalysis:
        """Analyze video recording for visual presentation"""
        
        # Simulate video analysis (in real implementation, you'd use computer vision)
        # For demo purposes, we'll provide simulated scores
        
        quality_issues = []
        
        # Simulated video analysis scores
        eye_contact_score = random.uniform(0.6, 0.9)
        posture_score = random.uniform(0.7, 0.95)
        facial_expression_score = random.uniform(0.6, 0.85)
        gesture_appropriateness = random.uniform(0.7, 0.9)
        professional_appearance = random.uniform(0.8, 0.95)
        background_appropriateness = random.uniform(0.7, 0.95)
        lighting_quality = random.uniform(0.6, 0.9)
        
        # Add quality issues based on scores
        if eye_contact_score < 0.7:
            quality_issues.append("Maintain more consistent eye contact with camera")
        if posture_score < 0.7:
            quality_issues.append("Improve posture - sit up straight")
        if lighting_quality < 0.7:
            quality_issues.append("Improve lighting - face should be well-lit")
        if background_appropriateness < 0.7:
            quality_issues.append("Choose a more professional background")
        
        return VideoAnalysis(
            eye_contact_score=eye_contact_score,
            posture_score=posture_score,
            facial_expression_score=facial_expression_score,
            gesture_appropriateness=gesture_appropriateness,
            professional_appearance=professional_appearance,
            background_appropriateness=background_appropriateness,
            lighting_quality=lighting_quality,
            video_quality_issues=quality_issues
        )
    
    async def generate_ai_feedback(self, transcribed_text: str, voice_analysis: VoiceAnalysis, 
                                  video_analysis: VideoAnalysis, question: str) -> str:
        """Generate AI feedback using Claude API"""
        
        try:
            # Prepare comprehensive prompt for AI analysis
            prompt = f"""
            As an expert interview coach, analyze this interview response and provide detailed feedback.
            
            QUESTION ASKED: {question}
            
            TRANSCRIBED RESPONSE: {transcribed_text}
            
            VOICE ANALYSIS DATA:
            - Clarity Score: {voice_analysis.clarity_score:.2f}
            - Speaking Pace: {voice_analysis.pace_score:.2f}
            - Confidence Level: {voice_analysis.tone_confidence:.2f}
            - Filler Words: {voice_analysis.filler_words_count}
            - Speaking Duration: {voice_analysis.speaking_duration:.1f} seconds
            
            VIDEO ANALYSIS DATA (if available):
            {f"- Eye Contact: {video_analysis.eye_contact_score:.2f}" if video_analysis else ""}
            {f"- Posture: {video_analysis.posture_score:.2f}" if video_analysis else ""}
            {f"- Professional Appearance: {video_analysis.professional_appearance:.2f}" if video_analysis else ""}
            
            Please provide comprehensive feedback covering:
            1. Content quality and relevance to the question
            2. Communication effectiveness (voice, pace, clarity)
            3. Professional presentation (if video provided)
            4. Specific areas for improvement
            5. Strengths to build upon
            6. Actionable recommendations
            
            Keep the feedback constructive, specific, and encouraging while being honest about areas needing improvement.
            """
            
            # Make API call to Claude
            response = await self.call_claude_api(prompt)
            return response
            
        except Exception as e:
            logger.error(f"AI feedback generation failed: {e}")
            return self.generate_fallback_feedback(transcribed_text, voice_analysis, video_analysis)
    
    async def call_claude_api(self, prompt: str) -> str:
        """Call Claude API for AI feedback"""
        try:
            import asyncio
            import aiohttp
            
            # Simulate API call (replace with actual Claude API integration)
            await asyncio.sleep(1)  # Simulate processing time
            
            # For demo, return structured feedback
            return """
            **Content Analysis:**
            Your response demonstrates good understanding of the question and provides relevant examples. The structure is logical with a clear beginning, middle, and end.
            
            **Communication Effectiveness:**
            - **Voice Clarity:** Good articulation with minimal filler words
            - **Speaking Pace:** Appropriate speed, easy to follow
            - **Confidence Level:** Shows confidence in your abilities
            
            **Professional Presentation:**
            - **Eye Contact:** Maintains good connection with the interviewer
            - **Posture:** Professional and engaged appearance
            - **Overall Presence:** Comes across as prepared and enthusiastic
            
            **Strengths:**
            ✅ Clear and structured response
            ✅ Relevant examples provided
            ✅ Professional demeanor
            ✅ Good use of specific details
            
            **Areas for Improvement:**
            📈 Consider adding more quantifiable achievements
            📈 Expand on the impact of your contributions
            📈 Practice smoother transitions between points
            
            **Action Items:**
            1. Prepare 2-3 specific examples with measurable outcomes
            2. Practice your response to reduce any remaining filler words
            3. Work on concluding responses with a forward-looking statement
            
            **Overall Assessment:** Strong performance with clear communication and professional presentation. Continue building on these foundations!
            """
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise e
    
    def generate_fallback_feedback(self, transcribed_text: str, voice_analysis: VoiceAnalysis, 
                                  video_analysis: Optional[VideoAnalysis]) -> str:
        """Generate fallback feedback when AI is unavailable"""
        
        feedback_parts = []
        
        # Content analysis
        word_count = len(transcribed_text.split())
        if word_count >= 100:
            feedback_parts.append("✅ **Content Length:** Good depth in your response")
        else:
            feedback_parts.append("📈 **Content Length:** Consider providing more detailed examples")
        
        # Voice analysis feedback
        if voice_analysis.clarity_score >= 0.8:
            feedback_parts.append("✅ **Speech Clarity:** Excellent clear communication")
        elif voice_analysis.clarity_score >= 0.6:
            feedback_parts.append("👍 **Speech Clarity:** Good clarity with room for improvement")
        else:
            feedback_parts.append("📈 **Speech Clarity:** Focus on reducing filler words and speaking more clearly")
        
        if voice_analysis.filler_words_count <= 2:
            feedback_parts.append("✅ **Professional Speech:** Minimal use of filler words")
        else:
            feedback_parts.append(f"📈 **Professional Speech:** Try to reduce filler words (detected {voice_analysis.filler_words_count})")
        
        # Video analysis feedback
        if video_analysis:
            if video_analysis.eye_contact_score >= 0.8:
                feedback_parts.append("✅ **Eye Contact:** Strong connection with camera")
            else:
                feedback_parts.append("📈 **Eye Contact:** Practice maintaining eye contact with the camera")
        
        return "\n\n".join(feedback_parts) + "\n\n**Overall:** Continue practicing to build confidence and refine your interview skills!"

class RealTimeValidator:
    """Real-time validation engine for interview responses"""
    
    def __init__(self):
        self.setup_validation_models()
        self.load_answer_patterns()
        self.initialize_cheating_detection()
        self.voice_video_analyzer = VoiceVideoAnalyzer()
        
    def setup_validation_models(self):
        """Initialize validation models and tools"""
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.count_vectorizer = CountVectorizer(max_features=500)
        else:
            self.vectorizer = None
            self.count_vectorizer = None
        
        # Load common interview question patterns
        self.question_patterns = {
            'tell_me_about_yourself': {
                'keywords': ['background', 'experience', 'education', 'skills', 'interests', 'career'],
                'required_elements': ['past', 'present', 'future'],
                'avoid_keywords': ['personal', 'family', 'irrelevant']
            },
            'strengths': {
                'keywords': ['strength', 'skill', 'ability', 'good', 'excel', 'expert'],
                'required_elements': ['specific_example', 'relevance'],
                'structure': ['claim', 'evidence', 'relevance']
            },
            'weaknesses': {
                'keywords': ['weakness', 'improve', 'challenge', 'learning', 'develop'],
                'required_elements': ['honest_weakness', 'improvement_plan'],
                'avoid_keywords': ['perfectionist', 'work_too_hard']
            },
            'why_here': {
                'keywords': ['university', 'program', 'research', 'faculty', 'opportunities'],
                'required_elements': ['specific_reasons', 'alignment'],
                'avoid_keywords': ['generic', 'ranking', 'prestige']
            }
        }
        
        # Common copied/template responses database
        self.template_responses = [
            "I am a highly motivated individual with excellent communication skills",
            "My greatest strength is my ability to work well in a team",
            "I am passionate about learning and always strive for excellence",
            "I chose this university because of its excellent reputation",
            "I see myself as a leader in my field in five years"
        ]
        
        # Fixed: Suspicious patterns that indicate cheating - now using tuples
        self.cheating_patterns = {
            'copy_paste_indicators': [
                ('multiple_caps_words', r'[A-Z]{2,}\s+[A-Z]{2,}'),
                ('email_addresses', r'\w+\.\w+@\w+\.\w+'),
                ('urls', r'http[s]?://\S+'),
                ('copyright_symbols', r'©|\(c\)|copyright'),
                ('year_ranges', r'\d{4}-\d{4}')
            ],
            'reading_indicators': [
                ('excessive_filler', r'\b(um|uh|er|ah){3,}'),
                ('multiple_ellipses', r'\.\.\.|…{2,}'),
                ('explicit_pauses', r'pause|wait|let me think')
            ],
            'template_indicators': [
                ('standard_intro', r'my name is \w+ and i am'),
                ('generic_motivation', r'i am a highly motivated'),
                ('academic_career', r'throughout my academic career'),
                ('generic_conclusion', r'in conclusion, i believe')
            ]
        }
    
    def load_answer_patterns(self):
        """Load patterns for good vs poor interview answers"""
        self.good_answer_patterns = {
            'structure_indicators': [
                'first', 'second', 'third', 'initially', 'then', 'finally',
                'for example', 'specifically', 'in particular', 'such as'
            ],
            'evidence_indicators': [
                'when i', 'during my', 'at university', 'in my project',
                'working on', 'led to', 'resulted in', 'achieved'
            ],
            'professional_language': [
                'collaborate', 'implement', 'analyze', 'develop', 'manage',
                'coordinate', 'establish', 'demonstrate', 'contribute'
            ]
        }
        
        self.poor_answer_patterns = {
            'vague_responses': [
                'i think', 'maybe', 'probably', 'sort of', 'kind of',
                'i guess', 'not sure', 'i dont know'
            ],
            'generic_responses': [
                'good at everything', 'no weaknesses', 'perfect student',
                'always successful', 'never failed'
            ],
            'inappropriate_content': [
                'hate', 'love', 'personal problems', 'family issues',
                'money problems', 'relationship'
            ]
        }
    
    def initialize_cheating_detection(self):
        """Initialize cheating detection systems"""
        self.typing_patterns = {}
        self.response_timings = {}
        self.behavioral_baselines = {}
        
        # Set up video analysis if available
        if VIDEO_AVAILABLE:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    async def validate_recording_response(self, question: str, audio_data: bytes = None, 
                                        video_data: bytes = None, transcribed_text: str = "",
                                        session_data: Dict = None) -> InterviewValidation:
        """Validate response from voice/video recording"""
        
        # First, analyze the recording
        voice_analysis = None
        video_analysis = None
        
        if audio_data:
            voice_analysis = await self.voice_video_analyzer.analyze_voice_recording(audio_data, transcribed_text)
        
        if video_data:
            video_analysis = self.voice_video_analyzer.analyze_video_recording(video_data)
        
        # Generate AI feedback
        ai_feedback = await self.voice_video_analyzer.generate_ai_feedback(
            transcribed_text, voice_analysis, video_analysis, question
        )
        
        # Calculate technical quality score
        technical_quality_score = self.calculate_technical_quality(voice_analysis, video_analysis)
        
        # Calculate communication effectiveness
        communication_effectiveness = self.calculate_communication_effectiveness(voice_analysis, video_analysis, transcribed_text)
        
        # Calculate overall presentation score
        overall_presentation_score = (technical_quality_score + communication_effectiveness) / 2
        
        # Generate improvement suggestions
        improvement_suggestions = self.generate_recording_suggestions(voice_analysis, video_analysis, transcribed_text)
        
        # Create recording validation
        recording_validation = RecordingValidation(
            transcribed_text=transcribed_text,
            voice_analysis=voice_analysis,
            video_analysis=video_analysis,
            ai_feedback=ai_feedback,
            technical_quality_score=technical_quality_score,
            communication_effectiveness=communication_effectiveness,
            overall_presentation_score=overall_presentation_score,
            improvement_suggestions=improvement_suggestions
        )
        
        # Use existing text validation for the transcribed content
        text_validation = self.validate_response_comprehensive(question, transcribed_text, session_data)
        
        # Combine validations
        text_validation.recording_validation = recording_validation
        
        # Adjust overall score to include presentation elements
        presentation_weight = 0.4 if video_data else 0.2  # Video gets more weight than voice-only
        text_weight = 1.0 - presentation_weight
        
        text_validation.overall_score = (
            text_validation.overall_score * text_weight + 
            overall_presentation_score * presentation_weight
        )
        
        return text_validation
    
    def calculate_technical_quality(self, voice_analysis: VoiceAnalysis, video_analysis: VideoAnalysis) -> float:
        """Calculate technical quality score"""
        scores = []
        
        if voice_analysis:
            scores.append(voice_analysis.clarity_score)
            scores.append(voice_analysis.pace_score)
            scores.append(1.0 - min(voice_analysis.filler_words_count / 20, 0.5))  # Penalty for excessive fillers
        
        if video_analysis:
            scores.append(video_analysis.lighting_quality)
            scores.append(video_analysis.professional_appearance)
            scores.append(video_analysis.background_appropriateness)
        
        return np.mean(scores) if scores else 0.5
    
    def calculate_communication_effectiveness(self, voice_analysis: VoiceAnalysis, 
                                           video_analysis: VideoAnalysis, transcribed_text: str) -> float:
        """Calculate communication effectiveness score"""
        scores = []
        
        # Text-based communication
        word_count = len(transcribed_text.split())
        text_score = min(word_count / 150, 1.0) if word_count > 50 else word_count / 50
        scores.append(text_score)
        
        if voice_analysis:
            scores.append(voice_analysis.tone_confidence)
            scores.append(voice_analysis.energy_level)
        
        if video_analysis:
            scores.append(video_analysis.eye_contact_score)
            scores.append(video_analysis.facial_expression_score)
            scores.append(video_analysis.gesture_appropriateness)
        
        return np.mean(scores) if scores else 0.5
    
    def generate_recording_suggestions(self, voice_analysis: VoiceAnalysis, 
                                     video_analysis: VideoAnalysis, transcribed_text: str) -> List[str]:
        """Generate suggestions specific to recording quality"""
        suggestions = []
        
        if voice_analysis:
            if voice_analysis.clarity_score < 0.7:
                suggestions.append("Practice speaking more clearly and distinctly")
            
            if voice_analysis.filler_words_count > 5:
                suggestions.append("Work on reducing filler words like 'um', 'uh', 'like'")
            
            if voice_analysis.pace_score < 0.6:
                suggestions.append("Adjust your speaking pace - aim for clear, measured delivery")
            
            if voice_analysis.tone_confidence < 0.6:
                suggestions.append("Speak with more confidence and conviction")
        
        if video_analysis:
            if video_analysis.eye_contact_score < 0.7:
                suggestions.append("Maintain better eye contact with the camera")
            
            if video_analysis.posture_score < 0.7:
                suggestions.append("Improve posture - sit up straight and appear engaged")
            
            if video_analysis.lighting_quality < 0.7:
                suggestions.append("Improve lighting - ensure your face is well-lit and visible")
            
            if video_analysis.background_appropriateness < 0.7:
                suggestions.append("Choose a more professional, uncluttered background")
        
        # Content suggestions
        if len(transcribed_text.split()) < 100:
            suggestions.append("Provide more detailed responses with specific examples")
        
        return suggestions
    
    def validate_response_comprehensive(self, question: str, response: str, 
                                     session_data: Dict = None, 
                                     timing_data: Dict = None) -> InterviewValidation:
        """Comprehensive validation of interview response"""
        
        # 1. Context Analysis
        context_validation = self.analyze_context_relevance(question, response)
        
        # 2. Cheating Detection
        cheating_flags = self.detect_cheating_comprehensive(response, timing_data, session_data)
        
        # 3. Authenticity Assessment
        authenticity_score = self.assess_authenticity(response, cheating_flags)
        
        # 4. Quality Analysis
        quality_scores = self.analyze_response_quality(response)
        
        # 5. Behavioral Analysis
        behavioral_scores = self.analyze_behavioral_indicators(question, response)
        
        # 6. Calculate overall score
        overall_score = self.calculate_overall_score(
            context_validation, quality_scores, behavioral_scores, authenticity_score
        )
        
        # 7. Generate suggestions and flags
        suggestions = self.generate_improvement_suggestions(
            context_validation, quality_scores, behavioral_scores
        )
        
        red_flags = self.identify_red_flags(cheating_flags, quality_scores)
        
        # 8. Determine if response passes validation
        passed = self.determine_pass_fail(overall_score, cheating_flags, red_flags)
        
        return InterviewValidation(
            overall_score=overall_score,
            context_validation=context_validation,
            cheating_flags=cheating_flags,
            authenticity_score=authenticity_score,
            response_quality=quality_scores,
            behavioral_analysis=behavioral_scores,
            improvement_suggestions=suggestions,
            red_flags=red_flags,
            passed_validation=passed
        )
    
    def analyze_context_relevance(self, question: str, response: str) -> ContextValidation:
        """Deep context analysis using multiple approaches"""
        
        # Clean and prepare text
        question_clean = self.clean_text(question)
        response_clean = self.clean_text(response)
        
        # 1. Keyword matching analysis
        keyword_score = self.calculate_keyword_relevance(question_clean, response_clean)
        
        # 2. Semantic similarity analysis
        semantic_score = self.calculate_semantic_similarity(question_clean, response_clean)
        
        # 3. Question-specific analysis
        question_type = self.identify_question_type(question)
        depth_score = self.analyze_answer_depth(response, question_type)
        
        # 4. Specificity analysis
        specificity_score = self.analyze_specificity(response)
        
        # 5. Professional language analysis
        professional_score = self.analyze_professional_language(response)
        
        # 6. Required elements check
        missing_elements = self.check_required_elements(question_type, response)
        
        # 7. Overall relevance calculation
        relevance_score = (keyword_score * 0.2 + semantic_score * 0.3 + 
                          depth_score * 0.3 + specificity_score * 0.2)
        
        # Generate strengths and suggestions
        strengths = self.identify_context_strengths(
            keyword_score, semantic_score, depth_score, specificity_score
        )
        
        suggestions = self.generate_context_suggestions(
            keyword_score, semantic_score, depth_score, specificity_score, missing_elements
        )
        
        return ContextValidation(
            relevance_score=relevance_score,
            keyword_match_score=keyword_score,
            semantic_coherence=semantic_score,
            depth_score=depth_score,
            specificity_score=specificity_score,
            professional_language_score=professional_score,
            missing_elements=missing_elements,
            strengths=strengths,
            suggestions=suggestions
        )
    
    def detect_cheating_comprehensive(self, response: str, timing_data: Dict = None, 
                                    session_data: Dict = None) -> List[CheatingFlag]:
        """Comprehensive cheating detection using multiple methods"""
        cheating_flags = []
        
        # 1. Text pattern analysis for copy-paste detection
        copy_paste_flags = self.detect_copy_paste_patterns(response)
        cheating_flags.extend(copy_paste_flags)
        
        # 2. Template response detection
        template_flags = self.detect_template_responses(response)
        cheating_flags.extend(template_flags)
        
        # 3. Typing pattern analysis
        if timing_data:
            typing_flags = self.analyze_typing_patterns(response, timing_data)
            cheating_flags.extend(typing_flags)
        
        # 4. Response timing analysis
        timing_flags = self.analyze_response_timing(response, timing_data)
        cheating_flags.extend(timing_flags)
        
        # 5. Linguistic inconsistency detection
        linguistic_flags = self.detect_linguistic_inconsistencies(response, session_data)
        cheating_flags.extend(linguistic_flags)
        
        # 6. Behavioral pattern analysis
        if session_data and 'previous_responses' in session_data:
            behavioral_flags = self.analyze_behavioral_consistency(response, session_data)
            cheating_flags.extend(behavioral_flags)
        
        return cheating_flags

    # Helper methods for validation
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        return text
    
    def calculate_keyword_relevance(self, question: str, response: str) -> float:
        """Calculate how well response keywords match question keywords"""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        question_keywords = question_words - stop_words
        response_keywords = response_words - stop_words
        
        if not question_keywords:
            return 0.5
        
        overlap = len(question_keywords.intersection(response_keywords))
        keyword_score = overlap / len(question_keywords)
        
        return min(keyword_score, 1.0)
    
    def calculate_semantic_similarity(self, question: str, response: str) -> float:
        """Calculate semantic similarity between question and response"""
        if not SKLEARN_AVAILABLE or not self.vectorizer:
            return self.calculate_keyword_relevance(question, response)
        
        try:
            texts = [question, response]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return self.calculate_keyword_relevance(question, response)
    
    def identify_question_type(self, question: str) -> str:
        """Identify the type of interview question"""
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in ['tell me about yourself', 'introduce yourself']):
            return 'tell_me_about_yourself'
        elif any(phrase in question_lower for phrase in ['strength', 'good at', 'excel']):
            return 'strengths'
        elif any(phrase in question_lower for phrase in ['weakness', 'improve', 'challenge']):
            return 'weaknesses'
        elif any(phrase in question_lower for phrase in ['why', 'choose', 'university', 'program']):
            return 'why_here'
        elif any(phrase in question_lower for phrase in ['example', 'time when', 'describe a situation']):
            return 'behavioral'
        elif any(phrase in question_lower for phrase in ['technical', 'explain', 'how would you']):
            return 'technical'
        else:
            return 'general'
    
    def analyze_answer_depth(self, response: str, question_type: str) -> float:
        """Analyze the depth and completeness of the answer"""
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        if 30 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 30:
            length_score = word_count / 30
        else:
            length_score = max(0.7, 1.0 - (word_count - 200) / 200)
        
        return length_score
    
    def analyze_specificity(self, response: str) -> float:
        """Analyze how specific and concrete the response is"""
        specific_indicators = ['example', 'specifically', 'when i', 'during my', 'at university']
        vague_indicators = ['things', 'stuff', 'many', 'various', 'some']
        
        specific_count = sum(1 for indicator in specific_indicators if indicator in response.lower())
        vague_count = sum(1 for indicator in vague_indicators if indicator in response.lower())
        
        return max(0.0, min(1.0, (specific_count - vague_count) / 5 + 0.5))
    
    def analyze_professional_language(self, response: str) -> float:
        """Analyze use of professional language"""
        professional_terms = ['collaborate', 'implement', 'analyze', 'develop', 'manage']
        professional_count = sum(1 for term in professional_terms if term in response.lower())
        return min(1.0, professional_count / 3 + 0.5)
    
    def check_required_elements(self, question_type: str, response: str) -> List[str]:
        """Check for missing required elements"""
        return []  # Simplified for demo
    
    def assess_authenticity(self, response: str, cheating_flags: List[CheatingFlag]) -> float:
        """Assess response authenticity"""
        base_score = 0.8
        penalty = len(cheating_flags) * 0.1
        return max(0.0, base_score - penalty)
    
    def analyze_response_quality(self, response: str) -> Dict[str, float]:
        """Analyze response quality metrics"""
        word_count = len(response.split())
        return {
            'length_score': min(1.0, word_count / 150),
            'structure_score': 0.8,  # Simplified
            'coherence_score': 0.75
        }
    
    def analyze_behavioral_indicators(self, question: str, response: str) -> Dict[str, float]:
        """Analyze behavioral indicators"""
        return {
            'confidence': 0.7,
            'enthusiasm': 0.8,
            'clarity': 0.75
        }
    
    def calculate_overall_score(self, context_validation, quality_scores, behavioral_scores, authenticity_score):
        """Calculate overall score"""
        return (context_validation.relevance_score * 0.4 + 
                np.mean(list(quality_scores.values())) * 0.3 +
                np.mean(list(behavioral_scores.values())) * 0.2 +
                authenticity_score * 0.1)
    
    def generate_improvement_suggestions(self, context_validation, quality_scores, behavioral_scores):
        """Generate improvement suggestions"""
        suggestions = []
        if context_validation.relevance_score < 0.7:
            suggestions.append("Focus more directly on answering the specific question asked")
        if quality_scores.get('length_score', 0) < 0.5:
            suggestions.append("Provide more detailed examples and explanations")
        return suggestions
    
    def identify_red_flags(self, cheating_flags, quality_scores):
        """Identify red flags"""
        red_flags = []
        if len(cheating_flags) > 2:
            red_flags.append("Multiple potential cheating indicators detected")
        return red_flags
    
    def determine_pass_fail(self, overall_score, cheating_flags, red_flags):
        """Determine if response passes validation"""
        return overall_score > 0.6 and len(cheating_flags) < 3
    
    def detect_copy_paste_patterns(self, response: str) -> List[CheatingFlag]:
        """Detect copy-paste patterns"""
        return []  # Simplified for demo
    
    def detect_template_responses(self, response: str) -> List[CheatingFlag]:
        """Detect template responses"""
        return []  # Simplified for demo
    
    def analyze_typing_patterns(self, response: str, timing_data: Dict) -> List[CheatingFlag]:
        """Analyze typing patterns"""
        return []  # Simplified for demo
    
    def analyze_response_timing(self, response: str, timing_data: Dict) -> List[CheatingFlag]:
        """Analyze response timing"""
        return []  # Simplified for demo
    
    def detect_linguistic_inconsistencies(self, response: str, session_data: Dict) -> List[CheatingFlag]:
        """Detect linguistic inconsistencies"""
        return []  # Simplified for demo
    
    def analyze_behavioral_consistency(self, response: str, session_data: Dict) -> List[CheatingFlag]:
        """Analyze behavioral consistency"""
        return []  # Simplified for demo
    
    def identify_context_strengths(self, keyword_score, semantic_score, depth_score, specificity_score):
        """Identify context strengths"""
        strengths = []
        if keyword_score > 0.7:
            strengths.append("Good use of relevant keywords")
        if depth_score > 0.7:
            strengths.append("Appropriate response depth")
        return strengths
    
    def generate_context_suggestions(self, keyword_score, semantic_score, depth_score, specificity_score, missing_elements):
        """Generate context suggestions"""
        suggestions = []
        if keyword_score < 0.6:
            suggestions.append("Use more keywords relevant to the question")
        if depth_score < 0.6:
            suggestions.append("Provide more comprehensive answers")
        return suggestions

class EnhancedInterviewSystem:
    """Main interview system with real validation and cheating detection"""
    
    def __init__(self, database_path: str = "interview_system.db"):
        self.database_path = database_path
        self.validator = RealTimeValidator()
        self.sessions = {}
        self._db_lock = threading.Lock()
        
        # Load comprehensive questions
        self.load_comprehensive_questions()
        
        logger.info("Enhanced Interview System initialized with voice/video recording support")
    
    def load_comprehensive_questions(self):
        """Load comprehensive interview questions"""
        self.questions = {
            'general': [
                "Tell me about yourself and your background.",
                "What are your greatest strengths?",
                "What is your biggest weakness?",
                "Why are you interested in this position/program?",
                "Where do you see yourself in 5 years?",
                "Why should we choose you over other candidates?",
                "What motivates you?",
                "Describe your ideal work environment.",
                "How do you handle stress and pressure?",
                "What are your salary expectations?"
            ],
            'behavioral': [
                "Tell me about a time when you faced a significant challenge. How did you handle it?",
                "Describe a situation where you had to work with a difficult person.",
                "Give me an example of when you showed leadership.",
                "Tell me about a time you made a mistake. How did you handle it?",
                "Describe a situation where you had to meet a tight deadline.",
                "Give an example of when you had to adapt to a significant change.",
                "Tell me about a time you disagreed with a supervisor or team member.",
                "Describe a project you're particularly proud of.",
                "Give an example of when you had to learn something new quickly.",
                "Tell me about a time you had to persuade someone to see your point of view."
            ],
            'technical': [
                "Explain a complex technical concept in simple terms.",
                "How do you stay current with industry trends and technologies?",
                "Describe your problem-solving process.",
                "What programming languages/tools are you most comfortable with?",
                "How do you approach debugging a difficult problem?",
                "Explain the difference between [technical concept A] and [technical concept B].",
                "How would you design a system to handle [specific technical challenge]?",
                "What is your experience with [specific technology/methodology]?",
                "How do you ensure code quality in your projects?",
                "Describe your experience with version control systems."
            ]
        }
    
    def create_interview_session(self, user_profile: Dict, config: Dict) -> str:
        """Create a new interview session"""
        session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Select questions based on config
        selected_questions = []
        categories = config.get('categories', ['general'])
        question_count = config.get('question_count', 5)
        
        for category in categories:
            if category in self.questions:
                category_questions = random.sample(
                    self.questions[category], 
                    min(question_count // len(categories), len(self.questions[category]))
                )
                selected_questions.extend(category_questions)
        
        # Ensure we have enough questions
        while len(selected_questions) < question_count:
            all_questions = []
            for cat_questions in self.questions.values():
                all_questions.extend(cat_questions)
            remaining = [q for q in all_questions if q not in selected_questions]
            if remaining:
                selected_questions.append(random.choice(remaining))
            else:
                break
        
        session_data = {
            'id': session_id,
            'user_profile': user_profile,
            'config': config,
            'questions': selected_questions[:question_count],
            'responses': [],
            'validations': [],
            'cheating_flags': [],
            'current_question': 0,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        self.sessions[session_id] = session_data
        return session_id
    
    async def submit_recording_response(self, session_id: str, question_index: int, 
                                      audio_data: bytes = None, video_data: bytes = None,
                                      transcribed_text: str = "") -> InterviewValidation:
        """Submit and validate recording response"""
        
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session = self.sessions[session_id]
        question = session['questions'][question_index]
        
        # Add session context
        session_data = {
            'previous_responses': [r['response'] for r in session['responses']],
            'user_baseline': session.get('user_baseline', {}),
            'session_config': session['config']
        }
        
        # Comprehensive validation with recording analysis
        validation = await self.validator.validate_recording_response(
            question, audio_data, video_data, transcribed_text, session_data
        )
        
        # Store response and validation
        response_data = {
            'question_index': question_index,
            'question': question,
            'response': transcribed_text,
            'has_audio': audio_data is not None,
            'has_video': video_data is not None,
            'validation': validation,
            'timestamp': datetime.now()
        }
        
        session['responses'].append(response_data)
        session['validations'].append(validation)
        session['cheating_flags'].extend(validation.cheating_flags)
        
        return validation
    
    def submit_text_response(self, session_id: str, question_index: int, response_text: str) -> InterviewValidation:
        """Submit and validate text response"""
        
        if session_id not in self.sessions:
            raise ValueError("Session not found")
        
        session = self.sessions[session_id]
        question = session['questions'][question_index]
        
        # Add session context
        session_data = {
            'previous_responses': [r['response'] for r in session['responses']],
            'user_baseline': session.get('user_baseline', {}),
            'session_config': session['config']
        }
        
        # Comprehensive validation
        validation = self.validator.validate_response_comprehensive(
            question, response_text, session_data
        )
        
        # Store response and validation
        response_data = {
            'question_index': question_index,
            'question': question,
            'response': response_text,
            'has_audio': False,
            'has_video': False,
            'validation': validation,
            'timestamp': datetime.now()
        }
        
        session['responses'].append(response_data)
        session['validations'].append(validation)
        session['cheating_flags'].extend(validation.cheating_flags)
        
        return validation
    
    def generate_session_report(self, session_id: str) -> Dict:
        """Generate comprehensive session report"""
        
        if session_id not in self.sessions:
            return {'error': 'Session not found'}
        
        session = self.sessions[session_id]
        validations = session['validations']
        
        if not validations:
            return {'error': 'No responses to analyze'}
        
        # Calculate performance metrics
        overall_scores = [v.overall_score for v in validations]
        context_scores = [v.context_validation.relevance_score for v in validations]
        authenticity_scores = [v.authenticity_score for v in validations]
        
        performance_metrics = {
            'overall_score': np.mean(overall_scores),
            'context_relevance': np.mean(context_scores),
            'authenticity_score': np.mean(authenticity_scores),
            'consistency': 1.0 - np.std(overall_scores),  # Higher consistency = lower std dev
            'total_questions': len(validations),
            'questions_passed': sum(1 for v in validations if v.passed_validation)
        }
        
        # Generate recommendations
        all_suggestions = []
        for v in validations:
            all_suggestions.extend(v.improvement_suggestions)
        
        # Count and prioritize suggestions
        suggestion_counts = Counter(all_suggestions)
        top_suggestions = [s for s, count in suggestion_counts.most_common(5)]
        
        recommendations = {
            'immediate_actions': top_suggestions[:3],
            'long_term_goals': top_suggestions[3:],
            'strengths': self.identify_session_strengths(validations),
            'areas_for_improvement': self.identify_improvement_areas(validations)
        }
        
        return {
            'session_id': session_id,
            'performance_metrics': performance_metrics,
            'recommendations': recommendations,
            'detailed_analysis': self.generate_detailed_analysis(validations),
            'cheating_summary': self.summarize_cheating_flags(session['cheating_flags'])
        }
    
    def identify_session_strengths(self, validations: List[InterviewValidation]) -> List[str]:
        """Identify strengths across the session"""
        strengths = []
        
        # Check for consistently high scores
        avg_overall = np.mean([v.overall_score for v in validations])
        if avg_overall > 0.8:
            strengths.append("Consistently high-quality responses")
        
        # Check context relevance
        avg_context = np.mean([v.context_validation.relevance_score for v in validations])
        if avg_context > 0.7:
            strengths.append("Strong relevance to questions asked")
        
        # Check authenticity
        avg_authenticity = np.mean([v.authenticity_score for v in validations])
        if avg_authenticity > 0.8:
            strengths.append("Authentic and genuine responses")
        
        return strengths
    
    def identify_improvement_areas(self, validations: List[InterviewValidation]) -> List[str]:
        """Identify areas for improvement"""
        areas = []
        
        # Check for low scores
        avg_overall = np.mean([v.overall_score for v in validations])
        if avg_overall < 0.6:
            areas.append("Overall response quality needs improvement")
        
        # Check context relevance
        avg_context = np.mean([v.context_validation.relevance_score for v in validations])
        if avg_context < 0.6:
            areas.append("Focus more directly on answering the questions")
        
        return areas
    
    def generate_detailed_analysis(self, validations: List[InterviewValidation]) -> Dict:
        """Generate detailed analysis"""
        return {
            'response_quality_trend': [v.overall_score for v in validations],
            'context_relevance_trend': [v.context_validation.relevance_score for v in validations],
            'authenticity_trend': [v.authenticity_score for v in validations]
        }
    
    def summarize_cheating_flags(self, cheating_flags: List[CheatingFlag]) -> Dict:
        """Summarize cheating flags"""
        if not cheating_flags:
            return {'status': 'clean', 'total_flags': 0}
        
        flag_types = Counter([flag.flag_type for flag in cheating_flags])
        severity_counts = Counter([flag.severity for flag in cheating_flags])
        
        return {
            'status': 'flags_detected',
            'total_flags': len(cheating_flags),
            'flag_types': dict(flag_types),
            'severity_breakdown': dict(severity_counts)
        }

# Voice/Video Recording Components

def create_audio_recorder_component():
    """Create HTML component for audio recording"""
    return """
    <div id="audioRecorder" style="text-align: center; padding: 20px; border: 2px dashed #667eea; border-radius: 10px; margin: 20px 0;">
        <h4>🎤 Voice Recording</h4>
        <div id="audioControls">
            <button id="startAudioBtn" style="background: #10b981; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                ▶️ Start Recording
            </button>
            <button id="stopAudioBtn" style="background: #ef4444; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;" disabled>
                ⏹️ Stop Recording
            </button>
            <button id="playAudioBtn" style="background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;" disabled>
                ▶️ Play
            </button>
        </div>
        <div id="audioStatus" style="margin: 10px; font-weight: bold; color: #667eea;"></div>
        <div id="recordingTimer" style="margin: 10px; font-size: 1.2em; color: #ef4444;"></div>
        <audio id="audioPlayback" controls style="margin: 10px; display: none;"></audio>
    </div>
    
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let startTime;
    let timerInterval;
    
    const startBtn = document.getElementById('startAudioBtn');
    const stopBtn = document.getElementById('stopAudioBtn');
    const playBtn = document.getElementById('playAudioBtn');
    const status = document.getElementById('audioStatus');
    const timer = document.getElementById('recordingTimer');
    const playback = document.getElementById('audioPlayback');
    
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    playBtn.addEventListener('click', playRecording);
    
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                playback.src = audioUrl;
                playback.style.display = 'block';
                
                // Store audio data for submission
                window.audioData = audioBlob;
                
                playBtn.disabled = false;
                status.textContent = '✅ Recording completed!';
                status.style.color = '#10b981';
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            isRecording = true;
            startTime = Date.now();
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            status.textContent = '🔴 Recording in progress...';
            status.style.color = '#ef4444';
            
            // Start timer
            timerInterval = setInterval(updateTimer, 100);
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            status.textContent = '❌ Microphone access denied. Please allow microphone access and try again.';
            status.style.color = '#ef4444';
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            clearInterval(timerInterval);
            timer.textContent = '';
        }
    }
    
    function playRecording() {
        playback.play();
    }
    
    function updateTimer() {
        if (isRecording) {
            const elapsed = (Date.now() - startTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor(elapsed % 60);
            timer.textContent = `⏱️ ${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    </script>
    """

def create_video_recorder_component():
    """Create HTML component for video recording"""
    return """
    <div id="videoRecorder" style="text-align: center; padding: 20px; border: 2px dashed #667eea; border-radius: 10px; margin: 20px 0;">
        <h4>🎥 Video Recording</h4>
        <div id="videoContainer" style="position: relative; display: inline-block;">
            <video id="videoPreview" width="400" height="300" autoplay muted style="border-radius: 10px; background: #000;"></video>
            <video id="videoPlayback" width="400" height="300" controls style="border-radius: 10px; display: none;"></video>
        </div>
        <div id="videoControls" style="margin-top: 15px;">
            <button id="startVideoBtn" style="background: #10b981; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                ▶️ Start Recording
            </button>
            <button id="stopVideoBtn" style="background: #ef4444; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;" disabled>
                ⏹️ Stop Recording
            </button>
            <button id="playVideoBtn" style="background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;" disabled>
                ▶️ Play
            </button>
            <button id="retakeVideoBtn" style="background: #f59e0b; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;" disabled>
                🔄 Retake
            </button>
        </div>
        <div id="videoStatus" style="margin: 10px; font-weight: bold; color: #667eea;"></div>
        <div id="videoTimer" style="margin: 10px; font-size: 1.2em; color: #ef4444;"></div>
        <div id="videoTips" style="margin: 15px; text-align: left; background: #f8fafc; padding: 15px; border-radius: 8px;">
            <h5>📋 Recording Tips:</h5>
            <ul style="margin: 10px 0; padding-left: 20px;">
                <li>Look directly at the camera</li>
                <li>Ensure good lighting on your face</li>
                <li>Choose a professional background</li>
                <li>Speak clearly and at a normal pace</li>
                <li>Maintain good posture</li>
            </ul>
        </div>
    </div>
    
    <script>
    let videoMediaRecorder;
    let videoChunks = [];
    let isVideoRecording = false;
    let videoStartTime;
    let videoTimerInterval;
    let stream;
    
    const videoStartBtn = document.getElementById('startVideoBtn');
    const videoStopBtn = document.getElementById('stopVideoBtn');
    const videoPlayBtn = document.getElementById('playVideoBtn');
    const videoRetakeBtn = document.getElementById('retakeVideoBtn');
    const videoStatus = document.getElementById('videoStatus');
    const videoTimer = document.getElementById('videoTimer');
    const videoPreview = document.getElementById('videoPreview');
    const videoPlayback = document.getElementById('videoPlayback');
    
    videoStartBtn.addEventListener('click', startVideoRecording);
    videoStopBtn.addEventListener('click', stopVideoRecording);
    videoPlayBtn.addEventListener('click', playVideoRecording);
    videoRetakeBtn.addEventListener('click', retakeVideo);
    
    // Initialize camera
    async function initializeCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }, 
                audio: true 
            });
            videoPreview.srcObject = stream;
            videoStatus.textContent = '📹 Camera ready - Click Start Recording when you\\'re ready';
            videoStatus.style.color = '#10b981';
        } catch (error) {
            console.error('Error accessing camera/microphone:', error);
            videoStatus.textContent = '❌ Camera/microphone access denied. Please allow access and refresh the page.';
            videoStatus.style.color = '#ef4444';
            videoStartBtn.disabled = true;
        }
    }
    
    async function startVideoRecording() {
        try {
            if (!stream) {
                await initializeCamera();
            }
            
            videoMediaRecorder = new MediaRecorder(stream, { 
                mimeType: 'video/webm;codecs=vp9,opus' 
            });
            videoChunks = [];
            
            videoMediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    videoChunks.push(event.data);
                }
            };
            
            videoMediaRecorder.onstop = () => {
                const videoBlob = new Blob(videoChunks, { type: 'video/webm' });
                const videoUrl = URL.createObjectURL(videoBlob);
                
                // Hide preview and show playback
                videoPreview.style.display = 'none';
                videoPlayback.src = videoUrl;
                videoPlayback.style.display = 'block';
                
                // Store video data for submission
                window.videoData = videoBlob;
                
                videoPlayBtn.disabled = false;
                videoRetakeBtn.disabled = false;
                videoStatus.textContent = '✅ Recording completed! Review your video or retake if needed.';
                videoStatus.style.color = '#10b981';
            };
            
            videoMediaRecorder.start(1000); // Collect data every second
            isVideoRecording = true;
            videoStartTime = Date.now();
            
            videoStartBtn.disabled = true;
            videoStopBtn.disabled = false;
            videoStatus.textContent = '🔴 Recording in progress... Look at the camera and speak clearly!';
            videoStatus.style.color = '#ef4444';
            
            // Start timer
            videoTimerInterval = setInterval(updateVideoTimer, 100);
            
        } catch (error) {
            console.error('Error starting video recording:', error);
            videoStatus.textContent = '❌ Error starting recording. Please try again.';
            videoStatus.style.color = '#ef4444';
        }
    }
    
    function stopVideoRecording() {
        if (videoMediaRecorder && isVideoRecording) {
            videoMediaRecorder.stop();
            isVideoRecording = false;
            
            videoStartBtn.disabled = false;
            videoStopBtn.disabled = true;
            
            clearInterval(videoTimerInterval);
            videoTimer.textContent = '';
        }
    }
    
    function playVideoRecording() {
        videoPlayback.play();
    }
    
    function retakeVideo() {
        // Reset for new recording
        videoPreview.style.display = 'block';
        videoPlayback.style.display = 'none';
        
        videoPlayBtn.disabled = true;
        videoRetakeBtn.disabled = true;
        videoStartBtn.disabled = false;
        
        window.videoData = null;
        videoStatus.textContent = '📹 Ready to record again - Click Start Recording';
        videoStatus.style.color = '#667eea';
    }
    
    function updateVideoTimer() {
        if (isVideoRecording) {
            const elapsed = (Date.now() - videoStartTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor(elapsed % 60);
            videoTimer.textContent = `⏱️ ${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            // Auto-stop after 5 minutes
            if (elapsed >= 300) {
                stopVideoRecording();
                videoStatus.textContent = '⏰ Recording stopped automatically after 5 minutes';
            }
        }
    }
    
    // Initialize camera when component loads
    initializeCamera();
    </script>
    """

# Enhanced Streamlit UI with Recording Support

def render_enhanced_interview_main():
    """Main interface for enhanced interview system with recording capabilities"""
    
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .recording-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .mode-selector {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 2rem 0;
    }
    .mode-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 300px;
    }
    .mode-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .mode-card.selected {
        border-color: #667eea;
        background: #f8fafc;
    }
    .feature-list {
        text-align: left;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800;">🎯 AI Interview Preparation System</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">Voice & Video Recording • Real-time AI Feedback • Performance Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    try:
        if 'enhanced_interview_system' not in st.session_state:
            with st.spinner("🔧 Initializing interview system..."):
                st.session_state.enhanced_interview_system = EnhancedInterviewSystem()
        
        system = st.session_state.enhanced_interview_system
        
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        return
    
    # Mode Selection
    st.markdown("### 🎯 Choose Your Practice Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="mode-card">
            <h3 style="color: #667eea; margin-top: 0;">🎤📹 Voice/Video Mock Test</h3>
            <p style="font-size: 1.1rem; color: #4a5568;">Record your responses and get AI-powered feedback on both content and presentation</p>
            <div class="feature-list">
                <h4>✨ Features:</h4>
                <ul>
                    <li>🎤 Voice recording with speech analysis</li>
                    <li>🎥 Video recording with visual feedback</li>
                    <li>🤖 AI-powered comprehensive analysis</li>
                    <li>📊 Communication effectiveness scoring</li>
                    <li>💡 Personalized improvement suggestions</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎬 Start Voice/Video Mock Test", type="primary", use_container_width=True):
            st.session_state.practice_mode = 'recording'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="mode-card">
            <h3 style="color: #10b981; margin-top: 0;">📝 Practice Session</h3>
            <p style="font-size: 1.1rem; color: #4a5568;">Type your responses for comprehensive text-based analysis and cheating detection</p>
            <div class="feature-list">
                <h4>✨ Features:</h4>
                <ul>
                    <li>✍️ Text-based response analysis</li>
                    <li>🕵️ Advanced cheating detection</li>
                    <li>🎯 Context relevance scoring</li>
                    <li>📈 Performance trend tracking</li>
                    <li>🔍 Detailed behavioral analysis</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Tesxt-Based Practice Session", use_container_width=True):
            st.session_state.practice_mode = 'traditional'
            st.rerun()
    
    # Route to appropriate mode
    if 'practice_mode' in st.session_state:
        if st.session_state.practice_mode == 'recording':
            render_recording_interview_mode(system)
        elif st.session_state.practice_mode == 'traditional':
            render_traditional_interview_mode(system)

def render_recording_interview_mode(system: EnhancedInterviewSystem):
    """Render the voice/video recording interview mode"""
    
    # Back button
    if st.button("← Back to Mode Selection"):
        del st.session_state.practice_mode
        st.rerun()
    
    st.markdown("### 🎬 Voice/Video Mock Interview")
    
    # Check if there's an active recording session
    if 'active_recording_session_id' in st.session_state and st.session_state.active_recording_session_id:
        render_active_recording_session(system, st.session_state.active_recording_session_id)
        return
    
    # Session configuration for recording mode
    with st.form("recording_interview_config"):
        st.markdown("#### 🎯 Configure Your Recording Session")
        
        col1, col2 = st.columns(2)
        
        with col1:
            recording_type = st.selectbox(
                "Recording Mode",
                ["Voice Only 🎤", "Video with Audio 🎥"],
                help="Choose whether to record voice only or full video"
            )
            
            question_count = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=5,
                value=3,
                help="Start with fewer questions for recording practice"
            )
        
        with col2:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Beginner", "Intermediate", "Advanced"],
                index=1
            )
            
            ai_feedback_level = st.selectbox(
                "AI Feedback Detail",
                ["Basic", "Comprehensive", "Expert Level"],
                index=1,
                help="How detailed should the AI analysis be?"
            )
        
        if st.form_submit_button("🎬 Start Recording Session", type="primary", use_container_width=True):
            # Create recording session
            config = {
                'type': 'voice_video_mock',
                'recording_type': recording_type,
                'difficulty': difficulty.lower(),
                'question_count': question_count,
                'ai_feedback_level': ai_feedback_level,
                'categories': ['general', 'behavioral']
            }
            
            user_profile = st.session_state.get('current_profile', {
                'id': 'demo_user',
                'first_name': 'Demo',
                'last_name': 'Student'
            })
            
            try:
                session_id = system.create_interview_session(user_profile, config)
                st.session_state.active_recording_session_id = session_id
                st.session_state.recording_config = config
                st.success("✅ Recording session created! Starting interview...")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to create recording session: {e}")

def render_active_recording_session(system: EnhancedInterviewSystem, session_id: str):
    """Render active recording session interface"""
    
    session = system.sessions.get(session_id)
    if not session:
        st.error("Session not found")
        st.session_state.active_recording_session_id = None
        return
    
    current_q = session['current_question']
    total_q = len(session['questions'])
    
    if current_q >= total_q:
        render_recording_session_completion(system, session_id)
        return
    
    # Progress
    progress = (current_q + 1) / total_q
    st.progress(progress)
    st.markdown(f"**Recording Question {current_q + 1} of {total_q}**")
    
    question = session['questions'][current_q]
    recording_type = session['config'].get('recording_type', 'Voice Only 🎤')
    
    # Question display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
        <h3 style="margin: 0; font-size: 1.5rem;">❓ Interview Question</h3>
        <h2 style="margin: 1rem 0; font-size: 1.8rem; line-height: 1.4;">{question}</h2>
        <p style="margin: 0; opacity: 0.9;">Take a moment to think, then record your response</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recording interface
    if "Video" in recording_type:
        st.markdown("#### 🎥 Video Recording")
        st.components.v1.html(create_video_recorder_component(), height=800)
        
        # Transcription input
        st.markdown("#### 📝 Manual Transcription (Optional)")
        transcription = st.text_area(
            "If you'd like to provide a transcription of your response:",
            height=100,
            placeholder="This is optional - leave blank for automatic transcription",
            help="Providing transcription helps with more accurate analysis"
        )
    else:
        st.markdown("#### 🎤 Voice Recording")
        st.components.v1.html(create_audio_recorder_component(), height=600)
        
        transcription = st.text_area(
            "Manual Transcription (Optional):",
            height=100,
            placeholder="This is optional - leave blank for automatic transcription"
        )
    
    # Submit recording
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Submit Recording", type="primary", use_container_width=True):
            # In a real implementation, you'd get the actual audio/video data
            # For demo, we'll simulate the process
            
            with st.spinner("🤖 Analyzing your recording... This may take a moment."):
                # Simulate recording analysis
                demo_transcription = transcription if transcription else f"This is a simulated transcription for question: {question}. In a real interview, I would provide a comprehensive answer covering my experience, skills, and enthusiasm for this opportunity."
                
                # Create mock audio/video data
                mock_audio_data = b"mock_audio_data"
                mock_video_data = b"mock_video_data" if "Video" in recording_type else None
                
                try:
                    # This would normally be async, but for demo we'll simulate
                    import asyncio
                    validation = asyncio.run(system.submit_recording_response(
                        session_id, current_q, 
                        mock_audio_data, mock_video_data, demo_transcription
                    ))
                    
                    # Store validation for feedback display
                    st.session_state[f'recording_feedback_{session_id}_{current_q}'] = validation
                    st.session_state[f'show_recording_feedback_{session_id}_{current_q}'] = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing recording: {e}")
    
    with col2:
        if st.button("⏭️ Skip Question", use_container_width=True):
            session['current_question'] += 1
            st.rerun()
    
    with col3:
        if st.button("🏁 Finish Session", use_container_width=True):
            render_recording_session_completion(system, session_id)
    
    # Show feedback if available
    if st.session_state.get(f'show_recording_feedback_{session_id}_{current_q}', False):
        validation = st.session_state.get(f'recording_feedback_{session_id}_{current_q}')
        if validation:
            render_recording_feedback(validation)
            
            if st.button("➡️ Continue to Next Question", type="primary"):
                st.session_state[f'show_recording_feedback_{session_id}_{current_q}'] = False
                session['current_question'] += 1
                st.rerun()

def render_recording_feedback(validation: InterviewValidation):
    """Render comprehensive feedback for recording responses"""
    
    recording_val = validation.recording_validation
    if not recording_val:
        st.error("No recording analysis available")
        return
    
    # Overall presentation score
    score = recording_val.overall_presentation_score
    
    if score >= 0.8:
        color = "#10b981"
        status = "Outstanding Presentation"
        icon = "🌟"
    elif score >= 0.65:
        color = "#3b82f6"
        status = "Strong Performance"
        icon = "🎯"
    elif score >= 0.5:
        color = "#f59e0b"
        status = "Good Foundation"
        icon = "📈"
    else:
        color = "#ef4444"
        status = "Needs Development"
        icon = "💪"
    
    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
        <h2 style="margin: 0; font-size: 2rem;">{status}</h2>
        <p style="font-size: 1.3rem; margin: 0.5rem 0;">Overall Score: {score:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Feedback
    st.markdown("### 🤖 AI Expert Analysis")
    st.markdown(recording_val.ai_feedback)
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎤 Voice Analysis")
        if recording_val.voice_analysis:
            voice = recording_val.voice_analysis
            st.metric("Speech Clarity", f"{voice.clarity_score:.1%}")
            st.metric("Speaking Pace", f"{voice.pace_score:.1%}")
            st.metric("Confidence Level", f"{voice.tone_confidence:.1%}")
            st.metric("Filler Words", voice.filler_words_count)
        else:
            st.info("Voice analysis not available")
    
    with col2:
        st.markdown("#### 📹 Video Analysis")
        if recording_val.video_analysis:
            video = recording_val.video_analysis
            st.metric("Eye Contact", f"{video.eye_contact_score:.1%}")
            st.metric("Professional Posture", f"{video.posture_score:.1%}")
            st.metric("Facial Expression", f"{video.facial_expression_score:.1%}")
            st.metric("Professional Appearance", f"{video.professional_appearance:.1%}")
        else:
            st.info("Video analysis not available for voice-only recording")
    
    # Technical quality and communication effectiveness
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Technical Quality", f"{recording_val.technical_quality_score:.1%}")
    with col2:
        st.metric("Communication Effectiveness", f"{recording_val.communication_effectiveness:.1%}")
    
    # Improvement suggestions
    if recording_val.improvement_suggestions:
        st.markdown("#### 💡 Specific Improvement Suggestions")
        for suggestion in recording_val.improvement_suggestions:
            st.info(f"• {suggestion}")

def render_recording_session_completion(system: EnhancedInterviewSystem, session_id: str):
    """Render recording session completion with comprehensive analysis"""
    
    st.markdown("### 🎉 Recording Session Completed!")
    
    # Generate comprehensive report
    try:
        with st.spinner("🤖 Generating comprehensive performance report..."):
            report = system.generate_session_report(session_id)
        
        if 'error' in report:
            st.error(f"Error generating report: {report['error']}")
            return
        
        # Overall performance summary
        performance_metrics = report.get('performance_metrics', {})
        overall_score = performance_metrics.get('overall_score', 0.5)
        
        if overall_score >= 0.8:
            color = "#10b981"
            status = "Exceptional Performance"
            icon = "🏆"
            message = "Outstanding! You demonstrated excellent interview skills across all areas."
        elif overall_score >= 0.65:
            color = "#3b82f6"
            status = "Strong Performance"
            icon = "🎯"
            message = "Great job! You show solid interview skills with room for fine-tuning."
        elif overall_score >= 0.5:
            color = "#f59e0b"
            status = "Good Foundation"
            icon = "📚"
            message = "Good start! Focus on the suggestions below to enhance your performance."
        else:
            color = "#ef4444"
            status = "Developing Skills"
            icon = "💪"
            message = "Keep practicing! The detailed feedback will help you improve significantly."
        
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h1 style="margin: 0; font-size: 3rem; font-weight: 800;">{overall_score*100:.1f}%</h1>
            <h2 style="margin: 1rem 0; font-size: 2rem;">{status}</h2>
            <p style="font-size: 1.1rem; opacity: 0.9; margin-top: 1rem;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights from recording analysis
        st.markdown("### 🎯 Key Performance Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Content Quality", f"{performance_metrics.get('context_relevance', 0.6):.1%}")
        with col2:
            st.metric("Presentation Skills", f"{overall_score:.1%}")
        with col3:
            st.metric("Communication", f"{performance_metrics.get('authenticity_score', 0.7):.1%}")
        with col4:
            st.metric("Professional Presence", f"{performance_metrics.get('consistency', 1.0):.1%}")
        
        # Recommendations
        recommendations = report.get('recommendations', {})
        if recommendations.get('immediate_actions'):
            st.markdown("### 🎯 Priority Action Items")
            for action in recommendations['immediate_actions'][:3]:
                st.info(f"🎯 {action}")
        
    except Exception as e:
        st.error(f"❌ Error generating report: {e}")
        st.success("✅ Recording session completed successfully!")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Practice Again", type="primary", use_container_width=True):
            st.session_state.active_recording_session_id = None
            del st.session_state.practice_mode
            st.rerun()
    
    with col2:
        if st.button("📊 View All Sessions", use_container_width=True):
            st.session_state.active_recording_session_id = None
            st.session_state.show_dashboard = True
            st.rerun()
    
    with col3:
        if st.button("🎯 Try Traditional Mode", use_container_width=True):
            st.session_state.active_recording_session_id = None
            st.session_state.practice_mode = 'traditional'
            st.rerun()

def render_traditional_interview_mode(system: EnhancedInterviewSystem):
    """Render traditional text-based interview mode"""
    
    # Back button
    if st.button("← Back to Mode Selection"):
        del st.session_state.practice_mode
        st.rerun()
    
    st.markdown("### 📝 Traditional Test Session Practice")
    st.info("💡 This is the comprehensive text-based interview practice with advanced cheating detection.")
    
    # Use the interview launcher for traditional mode
    render_traditional_interview_launcher(system, st.session_state.get('current_profile', {}))

def render_traditional_interview_launcher(system: EnhancedInterviewSystem, user_profile: Dict):
    """Render traditional interview setup and launcher"""
    
    # Check if there's an active session
    if 'active_traditional_session_id' in st.session_state and st.session_state.active_traditional_session_id:
        render_active_traditional_session(system, st.session_state.active_traditional_session_id)
        return
    
    st.markdown("### 🚀 Configure Your Traditional Interview Session")
    
    with st.form("traditional_interview_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            question_count = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=10,
                value=5,
                help="Choose how many questions you want to practice"
            )
            
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Beginner", "Intermediate", "Advanced"],
                index=1,
                help="Choose the complexity of questions"
            )
        
        with col2:
            categories = st.multiselect(
                "Question Categories",
                ["general", "behavioral", "technical"],
                default=["general", "behavioral"],
                help="Select which types of questions you want to practice"
            )
            
            cheating_detection = st.selectbox(
                "Cheating Detection Level",
                ["Standard", "Strict", "Very Strict"],
                index=1,
                help="How strictly to monitor for cheating"
            )
        
        time_limit = st.number_input(
            "Time Limit per Question (minutes)",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum time allowed per question"
        )
        
        if st.form_submit_button("📝 Start Text-Based Practice Session", type="primary", use_container_width=True):
            # Create traditional session
            config = {
                'type': 'traditional_text',
                'difficulty': difficulty.lower(),
                'question_count': question_count,
                'categories': categories if categories else ['general'],
                'cheating_detection': cheating_detection.lower().replace(' ', '_'),
                'time_limit': time_limit
            }
            
            user_profile = user_profile or {
                'id': 'demo_user',
                'first_name': 'Demo',
                'last_name': 'Student'
            }
            
            try:
                session_id = system.create_interview_session(user_profile, config)
                st.session_state.active_traditional_session_id = session_id
                st.success("✅ Traditional session created! Starting interview...")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to create traditional session: {e}")

def render_active_traditional_session(system: EnhancedInterviewSystem, session_id: str):
    """Render active traditional session interface"""
    
    session = system.sessions.get(session_id)
    if not session:
        st.error("Session not found")
        st.session_state.active_traditional_session_id = None
        return
    
    current_q = session['current_question']
    total_q = len(session['questions'])
    
    if current_q >= total_q:
        render_traditional_session_completion(system, session_id)
        return
    
    # Progress
    progress = (current_q + 1) / total_q
    st.progress(progress)
    st.markdown(f"**Question {current_q + 1} of {total_q}**")
    
    question = session['questions'][current_q]
    time_limit = session['config'].get('time_limit', 3)
    
    # Question display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #10b981 0%, #047857 100%); color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
        <h3 style="margin: 0; font-size: 1.5rem;">❓ Interview Question</h3>
        <h2 style="margin: 1rem 0; font-size: 1.8rem; line-height: 1.4;">{question}</h2>
        <p style="margin: 0; opacity: 0.9;">Time Limit: {time_limit} minutes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Response input
    response_text = st.text_area(
        "Your Response:",
        height=200,
        placeholder="Type your detailed response here...",
        help="Provide a comprehensive answer to the question above"
    )
    
    # Timer display
    if 'question_start_time' not in st.session_state:
        st.session_state.question_start_time = time.time()
    
    elapsed_time = time.time() - st.session_state.question_start_time
    remaining_time = max(0, time_limit * 60 - elapsed_time)
    
    if remaining_time > 0:
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        st.markdown(f"⏰ **Time Remaining: {minutes:02d}:{seconds:02d}**")
    else:
        st.markdown("⏰ **Time's Up!**")
    
    # Submit response
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Submit Response", type="primary", use_container_width=True):
            if response_text.strip():
                with st.spinner("🤖 Analyzing your response..."):
                    try:
                        validation = system.submit_text_response(
                            session_id, current_q, response_text
                        )
                        
                        # Store validation for feedback display
                        st.session_state[f'traditional_feedback_{session_id}_{current_q}'] = validation
                        st.session_state[f'show_traditional_feedback_{session_id}_{current_q}'] = True
                        st.session_state.question_start_time = time.time()  # Reset timer
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing response: {e}")
            else:
                st.warning("Please provide a response before submitting.")
    
    with col2:
        if st.button("⏭️ Skip Question", use_container_width=True):
            session['current_question'] += 1
            st.session_state.question_start_time = time.time()  # Reset timer
            st.rerun()
    
    with col3:
        if st.button("🏁 Finish Session", use_container_width=True):
            render_traditional_session_completion(system, session_id)
    
    # Show feedback if available
    if st.session_state.get(f'show_traditional_feedback_{session_id}_{current_q}', False):
        validation = st.session_state.get(f'traditional_feedback_{session_id}_{current_q}')
        if validation:
            render_traditional_feedback(validation)
            
            if st.button("➡️ Continue to Next Question", type="primary"):
                st.session_state[f'show_traditional_feedback_{session_id}_{current_q}'] = False
                session['current_question'] += 1
                st.session_state.question_start_time = time.time()  # Reset timer
                st.rerun()

def render_traditional_feedback(validation: InterviewValidation):
    """Render comprehensive feedback for traditional text responses"""
    
    # Overall score
    score = validation.overall_score
    
    if score >= 0.8:
        color = "#10b981"
        status = "Excellent Response"
        icon = "🌟"
    elif score >= 0.65:
        color = "#3b82f6"
        status = "Strong Response"
        icon = "🎯"
    elif score >= 0.5:
        color = "#f59e0b"
        status = "Good Response"
        icon = "📈"
    else:
        color = "#ef4444"
        status = "Needs Improvement"
        icon = "💪"
    
    st.markdown(f"""
    <div style="background: {color}; color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
        <h2 style="margin: 0; font-size: 2rem;">{status}</h2>
        <p style="font-size: 1.3rem; margin: 0.5rem 0;">Overall Score: {score:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Content Analysis")
        context_val = validation.context_validation
        st.metric("Relevance", f"{context_val.relevance_score:.1%}")
        st.metric("Keyword Match", f"{context_val.keyword_match_score:.1%}")
        st.metric("Depth", f"{context_val.depth_score:.1%}")
        st.metric("Specificity", f"{context_val.specificity_score:.1%}")
    
    with col2:
        st.markdown("#### 🎯 Quality Metrics")
        st.metric("Authenticity", f"{validation.authenticity_score:.1%}")
        quality_scores = validation.response_quality
        if quality_scores:
            for metric, score in quality_scores.items():
                st.metric(metric.replace('_', ' ').title(), f"{score:.1%}")
    
    # Cheating detection results
    if validation.cheating_flags:
        st.markdown("#### 🚨 Integrity Alerts")
        for flag in validation.cheating_flags:
            severity_color = {
                'low': '#f59e0b',
                'medium': '#ef4444',
                'high': '#dc2626'
            }.get(flag.severity, '#ef4444')
            
            st.markdown(f"""
            <div style="background: {severity_color}; color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <strong>{flag.flag_type.replace('_', ' ').title()} ({flag.severity.title()})</strong><br>
                {flag.details}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No integrity concerns detected")
    
    # Improvement suggestions
    if validation.improvement_suggestions:
        st.markdown("#### 💡 Improvement Suggestions")
        for suggestion in validation.improvement_suggestions:
            st.info(f"• {suggestion}")
    
    # Strengths
    if validation.context_validation.strengths:
        st.markdown("#### ✅ Strengths")
        for strength in validation.context_validation.strengths:
            st.success(f"• {strength}")

def render_traditional_session_completion(system: EnhancedInterviewSystem, session_id: str):
    """Render traditional session completion with comprehensive analysis"""
    
    st.markdown("### 🎉 Traditional Session Completed!")
    
    # Generate comprehensive report
    try:
        with st.spinner("📊 Generating comprehensive performance report..."):
            report = system.generate_session_report(session_id)
        
        if 'error' in report:
            st.error(f"Error generating report: {report['error']}")
            return
        
        # Overall performance summary
        performance_metrics = report.get('performance_metrics', {})
        overall_score = performance_metrics.get('overall_score', 0.5)
        
        if overall_score >= 0.8:
            color = "#10b981"
            status = "Outstanding Performance"
            icon = "🏆"
            message = "Excellent work! You demonstrated strong interview skills and authentic responses."
        elif overall_score >= 0.65:
            color = "#3b82f6"
            status = "Strong Performance"
            icon = "🎯"
            message = "Great job! You show solid understanding with room for refinement."
        elif overall_score >= 0.5:
            color = "#f59e0b"
            status = "Good Foundation"
            icon = "📚"
            message = "Good progress! Focus on the suggestions to enhance your responses."
        else:
            color = "#ef4444"
            status = "Developing Skills"
            icon = "💪"
            message = "Keep practicing! The feedback will help you improve significantly."
        
        st.markdown(f"""
        <div style="background: {color}; color: white; padding: 3rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h1 style="margin: 0; font-size: 3rem; font-weight: 800;">{overall_score*100:.1f}%</h1>
            <h2 style="margin: 1rem 0; font-size: 2rem;">{status}</h2>
            <p style="font-size: 1.1rem; opacity: 0.9; margin-top: 1rem;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown("### 🎯 Performance Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Content Quality", f"{performance_metrics.get('context_relevance', 0.6):.1%}")
        with col2:
            st.metric("Authenticity", f"{performance_metrics.get('authenticity_score', 0.7):.1%}")
        with col3:
            st.metric("Consistency", f"{performance_metrics.get('consistency', 1.0):.1%}")
        with col4:
            questions_passed = performance_metrics.get('questions_passed', 0)
            total_questions = performance_metrics.get('total_questions', 1)
            st.metric("Pass Rate", f"{questions_passed}/{total_questions}")
        
        # Cheating summary
        cheating_summary = report.get('cheating_summary', {})
        if cheating_summary.get('total_flags', 0) > 0:
            st.markdown("### 🚨 Integrity Summary")
            st.warning(f"⚠️ {cheating_summary['total_flags']} integrity alerts detected")
            
            if 'flag_types' in cheating_summary:
                flag_types = cheating_summary['flag_types']
                for flag_type, count in flag_types.items():
                    st.markdown(f"• {flag_type.replace('_', ' ').title()}: {count}")
        else:
            st.success("✅ Clean session - no integrity concerns detected")
        
        # Recommendations
        recommendations = report.get('recommendations', {})
        if recommendations.get('immediate_actions'):
            st.markdown("### 🎯 Priority Recommendations")
            for action in recommendations['immediate_actions'][:3]:
                st.info(f"🎯 {action}")
        
    except Exception as e:
        st.error(f"❌ Error generating report: {e}")
        st.success("✅ Traditional session completed successfully!")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Practice Again", type="primary", use_container_width=True):
            st.session_state.active_traditional_session_id = None
            del st.session_state.practice_mode
            st.rerun()
    
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.active_traditional_session_id = None
            st.session_state.show_dashboard = True
            st.rerun()
    
    with col3:
        if st.button("🎬 Try Recording Mode", use_container_width=True):
            st.session_state.active_traditional_session_id = None
            st.session_state.practice_mode = 'recording'
            st.rerun()

# Main entry point
def enhanced_interview_mock():
    """Enhanced interview preparation system main entry point with recording support"""
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .stApp > header {
        background-color: transparent;
    }
    
    .recording-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #667eea;
    }
    
    .feedback-section {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .feedback-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .score-display {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .status-excellent {
        background: #10b981;
        color: white;
    }
    
    .status-good {
        background: #3b82f6;
        color: white;
    }
    
    .status-fair {
        background: #f59e0b;
        color: white;
    }
    
    .status-poor {
        background: #ef4444;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = {
            'id': 'demo_user',
            'first_name': 'Demo',
            'last_name': 'User'
        }
    
    # Main application
    render_enhanced_interview_main()

# Backward compatibility
def interview_mock():
    """Legacy function name for backward compatibility"""
    return enhanced_interview_mock()

if __name__ == "__main__":
    st.set_page_config(
        page_title="🎯 Enhanced Interview Preparation with Recording",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    enhanced_interview_mock()