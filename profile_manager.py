"""
UEL AI System - Profile Management Module
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from config import PROFILE_DATA_DIR
from utils import get_logger


@dataclass
class UserProfile:
    """Enhanced user profile with comprehensive data"""
    # Basic Information
    id: str
    first_name: str
    last_name: str
    email: str = ""
    password_hash: str = ""
    phone: str = ""
    date_of_birth: str = ""
    
    # Location & Demographics
    country: str = ""
    nationality: str = ""
    city: str = ""
    postal_code: str = ""
    
    # Academic Background
    academic_level: str = ""
    field_of_interest: str = ""
    current_institution: str = ""
    current_major: str = ""
    gpa: float = 0.0
    graduation_year: int = 0
    
    # English Proficiency
    ielts_score: float = 0.0
    toefl_score: int = 0
    other_english_cert: str = ""
    
    # Professional Background
    work_experience_years: int = 0
    current_job_title: str = ""
    target_industry: str = ""
    professional_skills: List[str] = field(default_factory=list)
    
    # Interests & Preferences
    interests: List[str] = field(default_factory=list)
    career_goals: str = ""
    preferred_study_mode: str = ""
    preferred_start_date: str = ""
    budget_range: str = ""
    
    # Application History
    previous_applications: List[str] = field(default_factory=list)
    rejected_courses: List[str] = field(default_factory=list)
    preferred_courses: List[str] = field(default_factory=list)
    preferred_modules: List[str] = field(default_factory=list)
    
    # System Data
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    profile_completion: float = 0.0
    
    # AI Interaction History
    interaction_count: int = 0
    favorite_features: List[str] = field(default_factory=list)
    ai_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary"""
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'password_hash': self.password_hash,
            'phone': self.phone,
            'date_of_birth': self.date_of_birth,
            'country': self.country,
            'nationality': self.nationality,
            'city': self.city,
            'postal_code': self.postal_code,
            'academic_level': self.academic_level,
            'field_of_interest': self.field_of_interest,
            'current_institution': self.current_institution,
            'current_major': self.current_major,
            'gpa': self.gpa,
            'graduation_year': self.graduation_year,
            'ielts_score': self.ielts_score,
            'toefl_score': self.toefl_score,
            'other_english_cert': self.other_english_cert,
            'work_experience_years': self.work_experience_years,
            'current_job_title': self.current_job_title,
            'target_industry': self.target_industry,
            'professional_skills': self.professional_skills,
            'interests': self.interests,
            'career_goals': self.career_goals,
            'preferred_study_mode': self.preferred_study_mode,
            'preferred_start_date': self.preferred_start_date,
            'budget_range': self.budget_range,
            'previous_applications': self.previous_applications,
            'rejected_courses': self.rejected_courses,
            'preferred_courses': self.preferred_courses,
            'preferred_modules': self.preferred_modules,
            'created_date': self.created_date,
            'updated_date': self.updated_date,
            'last_active': self.last_active,
            'profile_completion': self.profile_completion,
            'interaction_count': self.interaction_count,
            'favorite_features': self.favorite_features,
            'ai_preferences': self.ai_preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create profile from dictionary"""
        return cls(**data)
    
    def calculate_completion(self) -> float:
        """Calculate profile completion percentage"""
        fields_to_check = [
            'first_name', 'last_name', 'email', 'country', 'nationality',
            'academic_level', 'field_of_interest', 'gpa', 'ielts_score',
            'career_goals', 'interests'
        ]
        
        completed = 0
        for field in fields_to_check:
            value = getattr(self, field, None)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    completed += 1
                elif isinstance(value, (str, int, float)) and value:
                    completed += 1
        
        self.profile_completion = (completed / len(fields_to_check)) * 100
        return self.profile_completion
    
    def update_activity(self):
        """Update last active timestamp"""
        self.last_active = datetime.now().isoformat()
        self.updated_date = datetime.now().isoformat()
    
    def add_interaction(self, feature_name: str):
        """Track feature usage"""
        self.interaction_count += 1
        self.update_activity()
        
        if feature_name not in self.favorite_features:
            self.favorite_features.append(feature_name)
        
        # Keep only top 5 favorite features
        if len(self.favorite_features) > 5:
            self.favorite_features = self.favorite_features[-5:]


class ProfileManager:
    """Enhanced profile management with persistence and validation"""
    
    def __init__(self, db_manager=None, profile_data_dir: str = PROFILE_DATA_DIR):
        self.db_manager = db_manager
        self.current_profile: Optional[UserProfile] = None
        self.profile_cache: Dict[str, UserProfile] = {}
        self.logger = get_logger(f"{__name__}.ProfileManager")
        self.profile_data_dir = Path(profile_data_dir)
        self._ensure_profile_data_dir_exists()

    def _ensure_profile_data_dir_exists(self):
        """Ensure the local directory for profiles exists."""
        try:
            self.profile_data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured profile data directory exists: {self.profile_data_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create profile data directory {self.profile_data_dir}: {e}")
            # Fallback to current directory if specified path is problematic
            self.profile_data_dir = Path("./Profile Data")
            self.profile_data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"Fallback to current directory for profile data: {self.profile_data_dir}")

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA256 for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against a stored hash."""
        return self._hash_password(password) == stored_hash

    def _get_profile_filepath(self, email: str) -> Path:
        """Generate a consistent file path for a given email."""
        hashed_email = hashlib.md5(email.lower().encode()).hexdigest()
        return self.profile_data_dir / f"{hashed_email}.json"

    def create_profile(self, profile_data: Dict, password: str) -> UserProfile:
        """Create new user profile, save it locally, and ensure unique email."""
        try:
            # Validate required fields
            required_fields = ['first_name', 'last_name', 'email', 'field_of_interest']
            for field in required_fields:
                if not profile_data.get(field):
                    raise ValueError(f"Required field missing: {field}")
            
            email = profile_data.get('email').lower()
            if not email:
                raise ValueError("Email is required for profile creation.")

            # Check if a profile with this email already exists
            if self.get_profile_by_email(email):
                raise ValueError(f"A profile with email '{email}' already exists. Please log in.")

            # Generate unique ID if not provided
            if 'id' not in profile_data:
                profile_data['id'] = self._generate_profile_id()
            
            # Hash the password and store it
            profile_data['password_hash'] = self._hash_password(password)

            # Create profile
            profile = UserProfile(**profile_data)
            profile.calculate_completion()
            
            # Save to local file
            self.save_profile(profile)
            
            # Set as current profile
            self.current_profile = profile
            self.profile_cache[profile.id] = profile
            
            self.logger.info(f"Created profile for {profile.first_name} {profile.last_name} with email {profile.email}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error creating profile: {e}")
            raise
    
    def update_profile(self, profile_id: str, updates: Dict) -> UserProfile:
        """Update existing profile and save changes locally."""
        try:
            profile = self.get_profile(profile_id)
            if not profile:
                raise ValueError(f"Profile not found: {profile_id}")
            
            # Update fields
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.calculate_completion()
            profile.update_activity()
            
            # Save changes
            self.save_profile(profile)
            
            self.logger.info(f"Updated profile {profile_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error updating profile: {e}")
            raise
    
    def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Get profile by ID (from cache or local file)."""
        try:
            # Check cache first
            if profile_id in self.profile_cache:
                return self.profile_cache[profile_id]
            
            # Iterate through profiles in the directory to find by ID
            for profile_file in self.profile_data_dir.glob("*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        data = json.load(f)
                    if data.get('id') == profile_id:
                        profile = UserProfile.from_dict(data)
                        self.profile_cache[profile_id] = profile
                        return profile
                except json.JSONDecodeError:
                    self.logger.warning(f"Skipping malformed JSON file: {profile_file}")
                except Exception as e:
                    self.logger.error(f"Error reading profile file {profile_file}: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting profile: {e}")
            return None

    def get_profile_by_email(self, email: str) -> Optional[UserProfile]:
        """Get profile by email (from cache or local file)."""
        try:
            # Check cache first
            for profile_id, profile in self.profile_cache.items():
                if profile.email.lower() == email.lower():
                    return profile
            
            # Check local file system
            profile_file = self._get_profile_filepath(email)
            if profile_file.exists():
                try:
                    with open(profile_file, 'r') as f:
                        data = json.load(f)
                    if data.get('email', '').lower() == email.lower():
                        profile = UserProfile.from_dict(data)
                        self.profile_cache[profile.id] = profile
                        return profile
                except json.JSONDecodeError:
                    self.logger.warning(f"Skipping malformed JSON file for email {email}: {profile_file}")
                except Exception as e:
                    self.logger.error(f"Error reading profile file for email {email}: {e}")
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting profile by email {email}: {e}")
            return None

    def login_profile(self, email: str, password: str) -> Optional[UserProfile]:
        """Authenticate user and load their profile."""
        try:
            profile = self.get_profile_by_email(email)
            if profile and self._verify_password(password, profile.password_hash):
                self.set_current_profile(profile)
                self.logger.info(f"User {email} logged in successfully.")
                return profile
            else:
                self.logger.warning(f"Login failed for email: {email}. Invalid credentials.")
                return None
        except Exception as e:
            self.logger.error(f"Login error for email {email}: {e}")
            return None

    def set_current_profile(self, profile: UserProfile):
        """Set the current active profile"""
        self.current_profile = profile
        profile.update_activity()
        self.save_profile(profile)
    
    def get_current_profile(self) -> Optional[UserProfile]:
        """Get current active profile"""
        return self.current_profile
    
    def save_profile(self, profile: UserProfile):
        """Save profile to local file."""
        try:
            profile_file = self._get_profile_filepath(profile.email)
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=4)
            self.logger.info(f"Profile saved locally: {profile_file}")
            
            # Update cache
            self.profile_cache[profile.id] = profile
            
        except Exception as e:
            self.logger.error(f"Error saving profile to local file {profile.email}: {e}")
    
    def _generate_profile_id(self) -> str:
        """Generate unique profile ID"""
        timestamp = int(time.time() * 1000)
        random_part = __import__('random').randint(1000, 9999)
        return f"UEL_{timestamp}_{random_part}"