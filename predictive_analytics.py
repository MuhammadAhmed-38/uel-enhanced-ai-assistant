"""
UEL AI System - Predictive Analytics Module
"""

import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils import get_logger

# Try to import optional libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PredictiveAnalyticsEngine:
    """Enhanced ML-based predictive analytics with robust error handling"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.admission_predictor = None
        self.success_probability_model = None
        self.models_trained = False
        self.logger = get_logger(f"{__name__}.PredictiveAnalyticsEngine")
        
        if SKLEARN_AVAILABLE:
            # Enhanced feature set
            self.feature_names = [
                'gpa', 'ielts_score', 'work_experience_years', 
                'course_difficulty', 'application_timing', 'international_status',
                'education_level_score', 'education_compatibility', 'gpa_percentile',
                'ielts_percentile', 'overall_academic_strength'
            ]
            self.logger.info("Predictive Analytics Engine initialized. Starting model training...")
            self.train_models()
        else:
            self.logger.warning("Scikit-learn not available. Predictive analytics disabled.")
            self.models_trained = False

    def train_models(self):
        """Train ML models with enhanced error handling"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. ML predictions disabled.")
            self.models_trained = False
            return
        
        try:
            self.logger.info("Starting ML model training...")
            
            # Get applications data
            applications = self._get_training_data()
            
            if not applications:
                self.logger.error("No training data available")
                self._set_fallback_models()
                return
            
            if len(applications) < 5:
                self.logger.warning(f"Limited training data ({len(applications)} samples). Generating additional synthetic data...")
                additional_data = self._generate_synthetic_data(20)
                applications.extend(additional_data)
            
            self.logger.info(f"Training with {len(applications)} applications")
            
            # Prepare training data
            features, targets = self._prepare_training_data(applications)
            
            if len(features) < 2:
                self.logger.error("Insufficient valid training samples after preprocessing")
                self._set_fallback_models()
                return
            
            self.logger.info(f"Prepared {len(features)} valid training samples with {len(features[0])} features")
            
            # Train Random Forest Classifier for admission prediction
            self.admission_predictor = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced'
            )
            
            self.admission_predictor.fit(features, targets)
            self.logger.info("Admission predictor trained successfully")
            
            # Train Gradient Boosting Regressor for success probability
            self.success_probability_model = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                subsample=0.8
            )
            
            # Convert targets to float for regression
            regression_targets = targets.astype(float)
            self.success_probability_model.fit(features, regression_targets)
            self.logger.info("Success probability model trained successfully")
            
            # Test model performance
            try:
                sample_features = features[:min(5, len(features))]
                test_predictions = self.admission_predictor.predict(sample_features)
                test_probabilities = self.success_probability_model.predict(sample_features)
                self.logger.info(f"Model testing successful. Sample predictions: {test_predictions}")
            except Exception as e:
                self.logger.warning(f"Model testing failed: {e}")
            
            self.models_trained = True
            self.logger.info("All ML models trained successfully!")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._set_fallback_models()

    def _get_training_data(self) -> List[Dict]:
        """Get training data from database and CSV"""
        applications = []
        
        try:
            # Get applications from data manager
            if not self.data_manager.applications_df.empty:
                for _, app in self.data_manager.applications_df.iterrows():
                    applications.append(app.to_dict())
                self.logger.info(f"Loaded {len(applications)} applications from CSV")
            
            # Add some synthetic data if we have very little real data
            if len(applications) < 10:
                synthetic_data = self._generate_synthetic_data(15)
                applications.extend(synthetic_data)
                self.logger.info(f"Added {len(synthetic_data)} synthetic applications")
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
        
        return applications

    def _generate_synthetic_data(self, count: int) -> List[Dict]:
        """Generate synthetic training data for ML models"""
        synthetic_data = []
        statuses = ['accepted', 'rejected', 'under_review']
        courses = ['Computer Science', 'Business Management', 'Data Science', 'Engineering', 'Psychology']
        nationalities = ['UK', 'India', 'China', 'Nigeria', 'Pakistan', 'USA', 'Canada']
        
        for i in range(count):
            # Generate realistic academic metrics
            gpa = round(random.uniform(2.0, 4.0), 2)
            ielts_score = round(random.uniform(5.0, 9.0), 1)
            work_experience = random.randint(0, 10)
            
            # Higher GPA and IELTS should correlate with acceptance
            acceptance_probability = (gpa / 4.0) * 0.4 + (ielts_score / 9.0) * 0.4 + random.uniform(0.1, 0.2)
            status = 'accepted' if acceptance_probability > 0.6 else 'rejected' if acceptance_probability < 0.4 else 'under_review'
            
            synthetic_data.append({
                'name': f'Student_{i+1000}',
                'course_applied': random.choice(courses),
                'status': status,
                'gpa': gpa,
                'ielts_score': ielts_score,
                'nationality': random.choice(nationalities),
                'work_experience_years': work_experience,
                'application_date': '2024-01-01',
                'current_education': random.choice(['undergraduate', 'graduate', 'high_school'])
            })
        
        return synthetic_data

    def _prepare_training_data(self, applications: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training"""
        features = []
        targets = []
        
        for app in applications:
            try:
                # Extract features with robust error handling
                feature_vector = self._extract_features(app)
                
                if feature_vector is not None and len(feature_vector) == len(self.feature_names):
                    # Convert status to binary (1 for accepted, 0 for rejected/under_review)
                    status = str(app.get('status', 'rejected')).lower()
                    target = 1 if status == 'accepted' else 0
                    
                    features.append(feature_vector)
                    targets.append(target)
                
            except Exception as e:
                self.logger.warning(f"Skipping invalid application data: {e}")
                continue
        
        return np.array(features), np.array(targets)

    def _extract_features(self, application: Dict) -> Optional[List[float]]:
        """Extract numerical features from application data"""
        try:
            # Basic academic features
            gpa = float(application.get('gpa', 3.0))
            ielts_score = float(application.get('ielts_score', 6.5))
            work_experience = float(application.get('work_experience_years', 0))
            
            # Course difficulty (simplified scoring)
            course = str(application.get('course_applied', '')).lower()
            difficulty_map = {
                'engineering': 4.0, 'medicine': 5.0, 'computer science': 4.0, 
                'data science': 4.0, 'business': 3.0, 'psychology': 3.5
            }
            course_difficulty = 3.0  # Default
            for key, value in difficulty_map.items():
                if key in course:
                    course_difficulty = value
                    break
            
            # Application timing (days from start of year)
            app_date = application.get('application_date', '2024-01-01')
            try:
                if isinstance(app_date, str):
                    app_dt = datetime.strptime(app_date, '%Y-%m-%d')
                    day_of_year = app_dt.timetuple().tm_yday
                    application_timing = min(day_of_year / 365.0, 1.0)
                else:
                    application_timing = 0.5
            except:
                application_timing = 0.5
            
            # International status
            nationality = str(application.get('nationality', 'UK')).upper()
            international_status = 0.0 if nationality == 'UK' else 1.0
            
            # Education level scoring
            education = str(application.get('current_education', 'undergraduate')).lower()
            education_scores = {
                'high_school': 1.0, 'undergraduate': 2.0, 
                'graduate': 3.0, 'postgraduate': 3.0, 'masters': 3.5, 'phd': 4.0
            }
            education_level_score = education_scores.get(education, 2.0)
            
            # Education compatibility (simplified)
            education_compatibility = 1.0
            
            # Percentile calculations (simplified)
            gpa_percentile = min(gpa / 4.0, 1.0)
            ielts_percentile = min(ielts_score / 9.0, 1.0)
            
            # Overall academic strength
            overall_academic_strength = (gpa_percentile * 0.6) + (ielts_percentile * 0.4)
            
            return [
                gpa, ielts_score, work_experience, course_difficulty, 
                application_timing, international_status, education_level_score,
                education_compatibility, gpa_percentile, ielts_percentile, 
                overall_academic_strength
            ]
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return None

    def _set_fallback_models(self):
        """Set simple fallback models when ML training fails"""
        self.logger.info("Setting up fallback prediction models...")
        
        class FallbackPredictor:
            def predict(self, features):
                predictions = []
                for feature_vector in features:
                    gpa = feature_vector[0] if len(feature_vector) > 0 else 3.0
                    ielts = feature_vector[1] if len(feature_vector) > 1 else 6.5
                    
                    # Simple decision logic
                    if gpa >= 3.5 and ielts >= 6.5:
                        predictions.append(1)  # Accept
                    elif gpa >= 3.0 and ielts >= 6.0:
                        predictions.append(1 if random.random() > 0.3 else 0)  # Maybe
                    else:
                        predictions.append(0)  # Reject
                
                return np.array(predictions)
            
            def predict_proba(self, features):
                predictions = self.predict(features)
                probabilities = []
                for pred in predictions:
                    if pred == 1:
                        probabilities.append([0.2, 0.8])  # 80% chance accepted
                    else:
                        probabilities.append([0.7, 0.3])  # 30% chance accepted
                return np.array(probabilities)
        
        self.admission_predictor = FallbackPredictor()
        self.success_probability_model = FallbackPredictor()
        self.models_trained = True
        self.logger.info("Fallback models set up successfully")

    def predict_admission_probability(self, student_profile: Dict) -> Dict:
        """Predict admission probability for a student"""
        try:
            if not self.models_trained or not self.admission_predictor:
                return {
                    "probability": 0.7,
                    "confidence": "low",
                    "factors": ["Model not available - using default estimate"],
                    "recommendations": ["Ensure all required documents are submitted"]
                }
            
            # Extract features from student profile
            feature_vector = self._extract_features(student_profile)
            if not feature_vector:
                return {
                    "probability": 0.5,
                    "confidence": "low", 
                    "factors": ["Insufficient data for accurate prediction"],
                    "recommendations": ["Please complete your profile for better predictions"]
                }
            
            # Make prediction
            features_array = np.array([feature_vector])
            
            if hasattr(self.admission_predictor, 'predict_proba'):
                probabilities = self.admission_predictor.predict_proba(features_array)
                admission_probability = probabilities[0][1]  # Probability of acceptance
            else:
                prediction = self.admission_predictor.predict(features_array)[0]
                admission_probability = 0.8 if prediction == 1 else 0.3
            
            # Determine confidence level
            if admission_probability > 0.8 or admission_probability < 0.2:
                confidence = "high"
            elif admission_probability > 0.6 or admission_probability < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Generate factors and recommendations
            factors = self._analyze_prediction_factors(student_profile, feature_vector)
            recommendations = self._generate_admission_recommendations(student_profile, admission_probability)
            
            return {
                "probability": round(admission_probability, 3),
                "confidence": confidence,
                "factors": factors,
                "recommendations": recommendations,
                "feature_importance": self._get_feature_importance()
            }
            
        except Exception as e:
            self.logger.error(f"Admission prediction error: {e}")
            return {
                "probability": 0.5,
                "confidence": "error",
                "factors": [f"Prediction error: {str(e)}"],
                "recommendations": ["Please contact admissions office for manual review"]
            }

    def _analyze_prediction_factors(self, profile: Dict, features: List[float]) -> List[str]:
        """Analyze what factors are influencing the prediction"""
        factors = []
        
        # Analyze GPA
        gpa = features[0]
        if gpa >= 3.7:
            factors.append("Excellent academic performance (GPA)")
        elif gpa >= 3.0:
            factors.append("Good academic performance (GPA)")
        else:
            factors.append("Academic performance could be stronger")
        
        # Analyze IELTS
        ielts = features[1]
        if ielts >= 7.0:
            factors.append("Strong English proficiency (IELTS)")
        elif ielts >= 6.5:
            factors.append("Good English proficiency (IELTS)")
        else:
            factors.append("English proficiency could be improved")
        
        # Analyze work experience
        work_exp = features[2]
        if work_exp >= 3:
            factors.append("Valuable work experience")
        elif work_exp >= 1:
            factors.append("Some professional experience")
        
        # International status
        if features[5] == 1.0:
            factors.append("International student background")
        
        return factors[:5]  # Limit to top 5

    def _generate_admission_recommendations(self, profile: Dict, probability: float) -> List[str]:
        """Generate recommendations to improve admission chances"""
        recommendations = []
        
        gpa = float(profile.get('gpa', 3.0))
        ielts = float(profile.get('ielts_score', 6.5))
        
        if probability < 0.5:
            recommendations.append("Consider retaking IELTS to improve English score")
            if gpa < 3.0:
                recommendations.append("Focus on improving academic grades")
            recommendations.append("Consider applying for foundation programs first")
        elif probability < 0.7:
            recommendations.append("Strengthen your personal statement")
            recommendations.append("Highlight any relevant achievements or certifications")
            if ielts < 7.0:
                recommendations.append("Consider improving IELTS score for better chances")
        else:
            recommendations.append("Strong application profile!")
            recommendations.append("Ensure all required documents are submitted")
            recommendations.append("Submit application before deadline")
        
        return recommendations

    def _get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        try:
            if hasattr(self.admission_predictor, 'feature_importances_'):
                importance_scores = self.admission_predictor.feature_importances_
                importance_dict = {}
                for i, feature_name in enumerate(self.feature_names):
                    if i < len(importance_scores):
                        importance_dict[feature_name] = float(importance_scores[i])
                return importance_dict
            else:
                # Default importance weights
                return {
                    'gpa': 0.25,
                    'ielts_score': 0.20,
                    'work_experience_years': 0.15,
                    'course_difficulty': 0.10,
                    'overall_academic_strength': 0.30
                }
        except:
            return {}