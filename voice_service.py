"""
UEL AI System - Voice Service Module
"""

import re
import threading
from utils import get_logger

# Try to import optional libraries
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


class VoiceService:
    """Enhanced voice recognition and text-to-speech service"""
    
    def __init__(self):
        self.is_listening = False
        self.is_speaking = False
        self.recognizer = None
        self.engine = None
        self.microphone = None
        self.logger = get_logger(f"{__name__}.VoiceService")
        
        if VOICE_AVAILABLE:
            try:
                # Initialize speech recognition
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                # Test microphone access
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Initialize text-to-speech
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.8)
                
                # Set voice (try to use a pleasant voice)
                voices = self.engine.getProperty('voices')
                if voices:
                    # Prefer female voice if available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                
                self.logger.info("Voice service initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Voice service initialization error: {e}")
                self.recognizer = None
                self.engine = None
                self.microphone = None
    
    def is_available(self) -> bool:
        """Check if voice service is available"""
        return (VOICE_AVAILABLE and 
                self.recognizer is not None and 
                self.engine is not None and 
                self.microphone is not None)
    
    def speech_to_text(self) -> str:
        """Convert speech to text with better error handling"""
        if not self.is_available():
            return "Voice service not available. Please install: pip install SpeechRecognition pyttsx3 pyaudio"
        
        try:
            with self.microphone as source:
                self.logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                self.logger.info("Listening for speech...")
                # Increase timeout and phrase time limit
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
            
            self.logger.info("Processing speech...")
            
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio)
                self.logger.info(f"Speech recognized: {text}")
                return text
            except sr.RequestError:
                # Fallback to offline recognition if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    self.logger.info(f"Speech recognized (offline): {text}")
                    return text
                except:
                    return "Speech recognition service unavailable. Please check internet connection."
        
        except sr.WaitTimeoutError:
            return "No speech detected within 10 seconds. Please try again."
        except sr.UnknownValueError:
            return "Could not understand speech. Please speak more clearly and try again."
        except sr.RequestError as e:
            return f"Speech recognition request failed: {e}"
        except OSError as e:
            if "No Default Input Device Available" in str(e):
                return "No microphone detected. Please connect a microphone and try again."
            else:
                return f"Microphone error: {e}"
        except Exception as e:
            self.logger.error(f"Speech recognition error: {e}")
            return f"Voice input failed: {str(e)}"
    
    def text_to_speech(self, text: str) -> bool:
        """Convert text to speech with better error handling"""
        if not self.is_available():
            self.logger.warning("TTS not available")
            return False
        
        if self.is_speaking:
            self.logger.warning("Already speaking")
            return False
        
        try:
            # Clean text for better speech
            clean_text = self._clean_text_for_speech(text)
            
            self.is_speaking = True
            
            def speak_thread():
                try:
                    self.engine.say(clean_text)
                    self.engine.runAndWait()
                except Exception as e:
                    self.logger.error(f"TTS thread error: {e}")
                finally:
                    self.is_speaking = False
            
            # Start speaking in background thread
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            self.is_speaking = False
            self.logger.error(f"Text-to-speech error: {e}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        
        # Remove emojis and special characters
        clean_text = re.sub(r'[🎓📄💰📞📧🌐📋📄⌛🎉⚠️⚙️💡✅⚠️📊📅🤔👋💳🚀🎯🎤📊]', '', text)
        
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # Bold
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # Italic
        clean_text = re.sub(r'#{1,6}\s*', '', clean_text)           # Headers
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)        # Code
        
        # Replace common abbreviations for better pronunciation
        replacements = {
            'UEL': 'University of East London',
            'AI': 'A I',
            'ML': 'Machine Learning',
            'UK': 'United Kingdom',
            'USA': 'United States of America',
            'IELTS': 'I E L T S',
            'GPA': 'G P A',
            'MSc': 'Master of Science',
            'BSc': 'Bachelor of Science',
            'MBA': 'Master of Business Administration'
        }
        
        for abbr, full_form in replacements.items():
            clean_text = clean_text.replace(abbr, full_form)
        
        # Remove extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text