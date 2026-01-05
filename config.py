"""
UEL AI System - Configuration and Enums Module
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class AIModelType(Enum):
    """Available AI model types"""
    DEEPSEEK_CODER = "deepseek-coder"
    DEEPSEEK_LATEST = "deepseek:latest"
    LLAMA2 = "llama2"
    CODELLAMA = "codellama"
    MISTRAL = "mistral"
    PHI = "phi"


class DocumentType(Enum):
    """Supported document types for verification"""
    TRANSCRIPT = "transcript"
    IELTS_CERTIFICATE = "ielts_certificate"
    PASSPORT = "passport"
    PERSONAL_STATEMENT = "personal_statement"
    REFERENCE_LETTER = "reference"


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Database
    database_path: str = "uel_ai_system.db"
    data_directory: str = "/Users/muhammadahmed/Downloads/uel-enhanced-ai-assistant/data"
    
    # AI Models
    ollama_host: str = "http://localhost:11434"
    default_model: str = AIModelType.DEEPSEEK_LATEST.value
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # ML Settings
    enable_ml_predictions: bool = True
    enable_sentiment_analysis: bool = True
    enable_document_verification: bool = True
    
    # Performance
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    
    # Security
    session_timeout_minutes: int = 60
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = field(default_factory=lambda: ['pdf', 'jpg', 'jpeg', 'png', 'doc', 'docx'])
    
    # University Info
    university_name: str = "University of East London"
    university_short_name: str = "UEL"
    admissions_email: str = "admissions@uel.ac.uk"
    admissions_phone: str = "+44 20 8223 3000"


@dataclass
class ResearchConfig:
    """Research configuration for academic evaluation"""
    # Evaluation settings
    enable_ab_testing: bool = True
    enable_statistical_testing: bool = True
    enable_explainable_ai: bool = True
    
    # Research parameters
    min_sample_size: int = 50
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    
    # Baseline models for comparison
    baseline_models: List[str] = field(default_factory=lambda: [
        'random', 'popularity_based', 'content_based', 'collaborative_filtering'
    ])
    
    # Metrics to track
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'precision_at_k', 'recall_at_k', 'ndcg', 'mrr', 'auc_roc', 'mse'
    ])


# Global configuration instances
config = SystemConfig()
research_config = ResearchConfig()

# Define the local folder path for profile data
PROFILE_DATA_DIR = "/Users/muhammadahmed/Downloads/uel-enhanced-ai-assistant/Profile Data"