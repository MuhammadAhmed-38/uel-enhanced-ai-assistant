"""
UEL AI System - Ollama Service Module
"""

import time
import requests
from typing import Optional

from config import config
from utils import get_logger


class OllamaService:
    """Enhanced Ollama service with robust error handling and fallbacks"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.default_model
        self.base_url = base_url or config.ollama_host
        self.api_url = f"{self.base_url}/api/generate"
        self.conversation_history = []
        self.max_history_length = 10
        self.is_available_cached = None
        self.last_check_time = 0
        self.logger = get_logger(f"{__name__}.OllamaService")
        
        self.logger.info(f"Initializing Ollama service: {self.base_url} with model {self.model_name}")
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is available with caching"""
        current_time = time.time()
        
        # Cache availability check for 30 seconds
        if self.is_available_cached is not None and (current_time - self.last_check_time) < 30:
            return self.is_available_cached
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json().get('models', [])]
                
                if self.model_name not in available_models:
                    self.logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                    # Try to use the first available model as fallback
                    if available_models:
                        self.model_name = available_models[0]
                        self.logger.info(f"Using fallback model: {self.model_name}")
                    else:
                        self.logger.error("No models available in Ollama")
                        self.is_available_cached = False
                        self.last_check_time = current_time
                        return False
                else:
                    self.logger.info(f"Successfully connected to Ollama. Using: {self.model_name}")
                
                self.is_available_cached = True
                self.last_check_time = current_time
                return True
            else:
                self.logger.warning(f"Ollama returned status code {response.status_code}")
                self.is_available_cached = False
                self.last_check_time = current_time
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            self.is_available_cached = False
            self.last_check_time = current_time
            return False
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        return self._check_availability()
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, 
                          temperature: float = None, max_tokens: int = None) -> str:
        """Generate response using Ollama with robust error handling"""
        try:
            if not self.is_available():
                self.logger.warning("Ollama not available, using fallback response")
                return self._fallback_response(prompt)
            
            # Prepare request data
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or config.llm_temperature,
                    "num_predict": max_tokens or config.max_tokens
                }
            }
            
            if system_prompt:
                data["system"] = system_prompt
            
            self.logger.info(f"Sending request to Ollama: {prompt[:100]}...")
            response = requests.post(self.api_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', 'No response generated')
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                if len(self.conversation_history) > self.max_history_length:
                    self.conversation_history = self.conversation_history[-self.max_history_length:]
                
                self.logger.info(f"Successfully received response from Ollama: {len(ai_response)} characters")
                return ai_response
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._fallback_response(prompt)
                
        except requests.exceptions.Timeout:
            self.logger.error("Ollama request timed out")
            return self._fallback_response(prompt, error_type="timeout")
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error to Ollama")
            return self._fallback_response(prompt, error_type="connection")
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}")
            return self._fallback_response(prompt, error_type="general")
    
    def _fallback_response(self, prompt: str, error_type: str = "general") -> str:
        """Provide intelligent fallback response when LLM is unavailable"""
        prompt_lower = prompt.lower()
        
        # Add context about the issue
        if error_type == "timeout":
            prefix = "I'm experiencing high load right now, but I can still help! "
        elif error_type == "connection":
            prefix = "I'm having connectivity issues, but here's what I can tell you: "
        else:
            prefix = "I'm using my fallback knowledge to help you: "
        
        # Course-related queries
        if any(word in prompt_lower for word in ['course', 'program', 'study', 'degree']):
            return f"""{prefix}

UEL Course Information

We offer excellent programs including:
• Computer Science - Programming, AI, Software Development
• Business Management - Leadership, Strategy, Entrepreneurship  
• Data Science - Analytics, Machine Learning, Statistics
• Engineering - Civil, Mechanical, Electronic Engineering
• Psychology - Clinical, Counseling, Research Psychology

Key Features:
✅ Industry-focused curriculum
✅ Experienced faculty
✅ Modern facilities
✅ Strong career support

For detailed course information, visit our website or contact admissions at {config.admissions_email}."""

        # Application-related queries
        elif any(word in prompt_lower for word in ['apply', 'application', 'admission', 'entry', 'requirement']):
            return f"""{prefix}

UEL Application Process

Entry Requirements:
• Academic qualifications (varies by course)
• English proficiency (IELTS 6.0-6.5)
• Personal statement
• References

Application Steps:
1. Choose your course
2. Check entry requirements
3. Submit online application
4. Upload supporting documents
5. Attend interview (if required)

Deadlines:
• September intake: August 1st
• January intake: November 1st

Contact: {config.admissions_email} | {config.admissions_phone}"""

        # Default response
        else:
            return f"""{prefix}

Thank you for contacting University of East London. I'm here to help with information about:

• Courses & Programs
• Applications & Admissions
• Fees & Scholarships
• Campus & Facilities

Quick Contact:
Email: {config.admissions_email}
Phone: {config.admissions_phone}
Website: uel.ac.uk

Could you please be more specific about what you'd like to know? I'm ready to provide detailed information about any aspect of UEL!"""