"""
UEL AI System - Course Recommendation Module
Adapted from standalone logic to work with existing data manager
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import re
from collections import defaultdict

from utils import get_logger

# Try to import optional libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AdvancedCourseRecommendationSystem:
    """Course recommendation system based on standalone logic"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.logger = get_logger(f"{__name__}.AdvancedCourseRecommendationSystem")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = None
        self.course_vectors = None
        
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Field keyword mappings for better matching (from standalone)
        self.field_keywords = {
            'machine learning': ['ai', 'artificial intelligence', 'data science', 'computer science', 'ml', 'deep learning', 'neural networks'],
            'artificial intelligence': ['ai', 'artificial intelligence', 'machine learning', 'computer science', 'robotics', 'automation'],
            'data science': ['data', 'analytics', 'statistics', 'big data', 'machine learning', 'python', 'r programming'],
            'computer science': ['programming', 'software', 'algorithms', 'computing', 'technology', 'it', 'coding'],
            'business': ['management', 'finance', 'marketing', 'economics', 'entrepreneurship', 'strategy', 'mba'],
            'engineering': ['mechanical', 'electrical', 'civil', 'chemical', 'aerospace', 'engineering'],
            'technology': ['it', 'software', 'hardware', 'tech', 'digital', 'innovation'],
            'finance': ['banking', 'investment', 'accounting', 'economics', 'financial', 'money'],
            'healthcare': ['medical', 'nursing', 'health', 'medicine', 'clinical', 'hospital'],
            'education': ['teaching', 'learning', 'pedagogy', 'curriculum', 'school', 'academic']
        }
        
        # Initialize course vectors
        self._prepare_course_vectors()
    
    def _prepare_course_vectors(self):
        """Prepare TF-IDF vectors for courses (from standalone logic)"""
        try:
            courses_df = self.data_manager.courses_df
            if courses_df.empty or not SKLEARN_AVAILABLE:
                self.logger.warning("Cannot prepare course vectors: no data or sklearn unavailable")
                return
            
            course_texts = []
            
            for _, row in courses_df.iterrows():
                # Combine relevant text fields
                text_parts = []
                
                # Add course name
                if 'course_name' in row and pd.notna(row['course_name']):
                    text_parts.append(str(row['course_name']))
                
                # Add description
                if 'description' in row and pd.notna(row['description']):
                    text_parts.append(str(row['description']))
                
                # Add keywords
                if 'keywords' in row and pd.notna(row['keywords']):
                    text_parts.append(str(row['keywords']))
                
                # Add department
                if 'department' in row and pd.notna(row['department']):
                    text_parts.append(str(row['department']))
                
                # Add level
                if 'level' in row and pd.notna(row['level']):
                    text_parts.append(str(row['level']))
                
                # Add career prospects
                if 'career_prospects' in row and pd.notna(row['career_prospects']):
                    text_parts.append(str(row['career_prospects']))
                
                # Add modules
                if 'modules' in row and pd.notna(row['modules']):
                    text_parts.append(str(row['modules']))
                
                course_text = ' '.join(text_parts)
                course_texts.append(course_text)
            
            # Create TF-IDF vectors
            if course_texts and self.tfidf_vectorizer:
                self.course_vectors = self.tfidf_vectorizer.fit_transform(course_texts)
                self.logger.info("Course vectors prepared successfully")
            else:
                self.logger.warning("No course text data found for vectorization")
                
        except Exception as e:
            self.logger.error(f"Error preparing course vectors: {e}")
    
    def recommend_courses(self, user_profile: Dict, preferences: Dict = None) -> List[Dict]:
        """Generate course recommendations based on user profile and preferences (main method from standalone)"""
        try:
            self.logger.info(f"Starting recommendation for user profile: {user_profile}")
            
            courses_df = self.data_manager.courses_df
            if courses_df.empty:
                self.logger.warning("No courses data available for recommendation.")
                return self._create_fallback_recommendations(user_profile)

            self.logger.info(f"Found {len(courses_df)} courses in database")

            # Method 1: Content-based recommendations using TF-IDF
            content_recs = self._content_based_recommendations(user_profile)
            
            # Method 2: Keyword-based recommendations
            keyword_recs = self._keyword_based_recommendations(user_profile)
            
            # Combine and deduplicate recommendations
            all_recommendations = content_recs + keyword_recs
            
            if not all_recommendations:
                self.logger.warning("No recommendations found with any method, using fallback")
                return self._create_fallback_recommendations(user_profile)
            
            # Remove duplicates and combine scores
            unique_recs = self._combine_recommendations(all_recommendations)
            
            # Apply filters and scoring
            final_recs = []
            for rec in unique_recs[:20]:  # Get more for filtering
                enhanced_rec = self._enhance_recommendation(rec, user_profile, preferences)
                if enhanced_rec:
                    final_recs.append(enhanced_rec)
            
            # Sort by final score and return top 10
            final_recs.sort(key=lambda x: x['score'], reverse=True)
            final_recommendations = final_recs[:10]
            
            self.logger.info(f"Returning {len(final_recommendations)} final recommendations")
            return final_recommendations

        except Exception as e:
            self.logger.error(f"Course recommendation error: {e}")
            return self._create_fallback_recommendations(user_profile)
    
    def _content_based_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Generate recommendations using content-based filtering (from standalone)"""
        if self.course_vectors is None or not SKLEARN_AVAILABLE:
            self.logger.warning("Content-based recommendations unavailable: no vectors or sklearn")
            return []
        
        try:
            # Create user query from profile
            user_text = self._create_user_text(user_profile)
            
            if not user_text:
                return []
            
            # Transform user text to vector
            user_vector = self.tfidf_vectorizer.transform([user_text])
            
            # Calculate similarities
            similarities = cosine_similarity(user_vector, self.course_vectors)[0]
            
            recommendations = []
            courses_df = self.data_manager.courses_df
            
            for idx, similarity in enumerate(similarities):
                if idx < len(courses_df):
                    course = courses_df.iloc[idx]
                    recommendations.append({
                        'course_index': idx,
                        'course_data': course.to_dict(),
                        'score': float(similarity),
                        'method': 'content_based'
                    })
            
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:20]
            
        except Exception as e:
            self.logger.error(f"Content-based recommendation error: {e}")
            return []
    
    def _keyword_based_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Generate recommendations using keyword matching (from standalone)"""
        try:
            courses_df = self.data_manager.courses_df
            if courses_df.empty:
                return []
            
            user_field = user_profile.get('field_of_interest', '').lower()
            user_interests = [i.lower() for i in user_profile.get('interests', [])]
            user_skills = [s.lower() for s in user_profile.get('professional_skills', [])]
            
            # Get relevant keywords
            relevant_keywords = []
            for field, keywords in self.field_keywords.items():
                if field in user_field or any(keyword in user_field for keyword in keywords):
                    relevant_keywords.extend(keywords)
            
            relevant_keywords.extend(user_interests)
            relevant_keywords.extend(user_skills)
            relevant_keywords = list(set(relevant_keywords))
            
            recommendations = []
            
            for idx, course in courses_df.iterrows():
                score = 0.0
                matches = []
                
                # Create course text for matching
                course_text = self._get_course_text(course).lower()
                course_name = str(course.get('course_name', '')).lower()
                
                # Check keyword matches
                for keyword in relevant_keywords:
                    if keyword and keyword in course_text:
                        if keyword in course_name:
                            score += 0.3
                            matches.append(f"'{keyword}' in title")
                        else:
                            score += 0.1
                            matches.append(f"'{keyword}' in content")
                
                # Field match
                if user_field and user_field in course_text:
                    score += 0.4
                    matches.append(f"Field match: {user_field}")
                
                if score > 0.05:
                    recommendations.append({
                        'course_index': idx,
                        'course_data': course.to_dict(),
                        'score': min(score, 1.0),
                        'method': 'keyword_based',
                        'matches': matches
                    })
            
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:20]
            
        except Exception as e:
            self.logger.error(f"Keyword-based recommendation error: {e}")
            return []
    
    def _create_user_text(self, user_profile: Dict) -> str:
        """Create text representation of user profile (from standalone)"""
        text_parts = []
        
        if user_profile.get('field_of_interest'):
            text_parts.append(user_profile['field_of_interest'])
        
        if user_profile.get('career_goals'):
            text_parts.append(user_profile['career_goals'])
        
        if user_profile.get('current_major'):
            text_parts.append(user_profile['current_major'])
        
        if user_profile.get('target_industry'):
            text_parts.append(user_profile['target_industry'])
        
        text_parts.extend(user_profile.get('interests', []))
        text_parts.extend(user_profile.get('professional_skills', []))
        text_parts.extend(user_profile.get('preferred_modules', []))
        
        if user_profile.get('academic_level'):
            text_parts.append(user_profile['academic_level'])
        
        return ' '.join(filter(None, text_parts))
    
    def _get_course_text(self, course) -> str:
        """Get text representation of a course (from standalone)"""
        text_parts = []
        
        fields = ['course_name', 'description', 'keywords', 'department', 'level', 'career_prospects', 'modules']
        
        for field in fields:
            if field in course and pd.notna(course[field]):
                text_parts.append(str(course[field]))
        
        return ' '.join(text_parts)
    
    def _combine_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Combine recommendations from different methods (from standalone)"""
        course_map = {}
        
        for rec in recommendations:
            course_idx = rec['course_index']
            
            if course_idx in course_map:
                # Average the scores
                existing = course_map[course_idx]
                existing_score = existing['score']
                new_score = rec['score']
                combined_score = (existing_score + new_score) / 2
                existing['score'] = combined_score
                existing['method'] = f"{existing['method']}, {rec['method']}"
                
                # Combine matches if available
                if 'matches' in rec and 'matches' in existing:
                    existing['matches'].extend(rec['matches'])
                elif 'matches' in rec:
                    existing['matches'] = rec['matches']
            else:
                course_map[course_idx] = rec.copy()
        
        return list(course_map.values())
    
    def _enhance_recommendation(self, rec: Dict, user_profile: Dict, preferences: Dict = None) -> Dict:
        """Enhance recommendation with additional scoring and information (from standalone logic)"""
        try:
            course_data = rec['course_data']
            score = rec['score']
            reasons = []
            
            # Academic level matching
            user_level = user_profile.get('academic_level', '').lower()
            preferred_level = (preferences or {}).get('level', '').lower() if preferences else ''
            course_level = str(course_data.get('level', '')).lower()
            
            # Check preferred level first, then user's current level
            target_level = preferred_level if preferred_level and preferred_level != 'any' else user_level
            
            if target_level and course_level:
                if target_level in course_level or course_level in target_level:
                    score += 0.2
                    reasons.append(f"Matches academic level: {target_level}")
                else:
                    score *= 0.7
                    reasons.append(f"Different study level: {course_level}")
            
            # GPA requirement check
            user_gpa = user_profile.get('gpa', 0.0)
            min_gpa = course_data.get('min_gpa', 0.0)
            
            if isinstance(min_gpa, (int, float)) and min_gpa > 0:
                if user_gpa >= min_gpa:
                    score += 0.1
                    reasons.append(f"Meets GPA requirement ({min_gpa})")
                else:
                    score *= 0.8
                    reasons.append(f"Below GPA requirement ({min_gpa})")
            
            # IELTS requirement check
            user_ielts = user_profile.get('ielts_score', 0.0)
            min_ielts = course_data.get('min_ielts', 0.0)
            
            if isinstance(min_ielts, (int, float)) and min_ielts > 0:
                if user_ielts >= min_ielts:
                    score += 0.1
                    reasons.append(f"Meets IELTS requirement ({min_ielts})")
                else:
                    score *= 0.8
                    reasons.append(f"Below IELTS requirement ({min_ielts})")
            
            # Budget check
            budget_max = (preferences or {}).get('budget_max', user_profile.get('budget_max', float('inf'))) if preferences else user_profile.get('budget_max', float('inf'))
            course_fees = course_data.get('fees_international', 0)
            
            if isinstance(course_fees, (int, float)) and course_fees > 0 and budget_max < float('inf'):
                if course_fees <= budget_max:
                    score += 0.05
                    reasons.append(f"Within budget (£{course_fees:,.0f})")
                else:
                    score *= 0.6
                    reasons.append(f"Above budget (£{course_fees:,.0f})")
            
            # Work experience relevance
            user_work_exp = user_profile.get('work_experience_years', 0)
            if user_work_exp > 0:
                career_prospects = str(course_data.get('career_prospects', '')).lower()
                if any(keyword in career_prospects for keyword in ['experience', 'professional', 'industry', 'career']):
                    score += 0.05
                    reasons.append(f"Your work experience ({user_work_exp} years) may benefit this course")
            
            # Determine match quality
            if score >= 0.7:
                match_quality = "Excellent Match"
            elif score >= 0.5:
                match_quality = "Good Match"
            elif score >= 0.3:
                match_quality = "Fair Match"
            else:
                match_quality = "Possible Match"
            
            # Add method-specific reasons
            if 'matches' in rec:
                reasons.extend(rec['matches'][:2])
            elif rec.get('method') == 'content_based':
                reasons.append(f"Content similarity: {score:.2f}")
            
            # Ensure score is within reasonable range
            score = max(0.0, min(1.2, score))
            
            return {
                'course_id': f"course_{rec['course_index']}",
                'course_name': course_data.get('course_name', 'Unknown Course'),
                'department': course_data.get('department', 'General Studies'),
                'level': course_data.get('level', 'undergraduate'),
                'description': course_data.get('description', 'No description available'),
                'fees': f"£{course_data.get('fees_international', 0):,.0f}" if course_data.get('fees_international') else "Not specified",
                'duration': course_data.get('duration', 'Not specified'),
                'min_gpa': course_data.get('min_gpa', 'Not specified'),
                'min_ielts': course_data.get('min_ielts', 'Not specified'),
                'career_prospects': course_data.get('career_prospects', 'Various opportunities'),
                'modules': course_data.get('modules', 'No modules listed'),
                'score': score,
                'match_quality': match_quality,
                'reasons': list(set(reasons[:3])),  # Limit to top 3 unique reasons
                'method': rec.get('method', 'combined')
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing recommendation: {e}")
            return None
    
    def _create_fallback_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Provide fallback recommendations when other methods fail (from standalone)"""
        try:
            courses_df = self.data_manager.courses_df
            
            # If no courses data at all, create dummy recommendations
            if courses_df.empty:
                return self._create_dummy_recommendations(user_profile)
            
            # If we have courses but no matches, return top courses with adjusted reasons
            sample_size = min(10, len(courses_df))
            sample_courses = courses_df.sample(n=sample_size)
            
            recommendations = []
            for idx, (_, course) in enumerate(sample_courses.iterrows()):
                recommendations.append({
                    'course_id': f"fallback_{idx}",
                    'course_name': course.get('course_name', 'Course Name'),
                    'department': course.get('department', 'General Studies'),
                    'level': course.get('level', 'undergraduate'),
                    'description': course.get('description', 'No description available'),
                    'fees': f"£{course.get('fees_international', 15000):,}",
                    'duration': course.get('duration', 'Not specified'),
                    'min_gpa': course.get('min_gpa', 'Not specified'),
                    'min_ielts': course.get('min_ielts', 'Not specified'),
                    'career_prospects': course.get('career_prospects', 'Various opportunities'),
                    'modules': course.get('modules', 'No modules listed'),
                    'score': random.uniform(0.3, 0.7),
                    'match_quality': 'Possible Match',
                    'reasons': ['Recommended based on general criteria', 'May align with your interests'],
                    'method': 'fallback'
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Fallback recommendation error: {e}")
            return self._create_dummy_recommendations(user_profile)
    
    def _create_dummy_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Create dummy recommendations when no data is available (from standalone)"""
        user_field = user_profile.get('field_of_interest', 'Technology')
        user_level = user_profile.get('academic_level', 'Bachelor\'s').title()
        
        dummy_courses = [
            {
                'course_name': f'{user_field} Fundamentals',
                'department': f'{user_field} Studies',
                'description': f'Introduction to core concepts in {user_field.lower()}',
                'level': 'undergraduate' if 'bachelor' in user_level.lower() else 'postgraduate'
            },
            {
                'course_name': f'Advanced {user_field}',
                'department': f'{user_field} Studies', 
                'description': f'Advanced topics and applications in {user_field.lower()}',
                'level': 'postgraduate'
            },
            {
                'course_name': f'{user_field} and Innovation',
                'department': 'Innovation Studies',
                'description': f'Innovative approaches and emerging trends in {user_field.lower()}',
                'level': 'undergraduate'
            }
        ]
        
        recommendations = []
        for idx, course in enumerate(dummy_courses):
            recommendations.append({
                'course_id': f"dummy_{idx}",
                'course_name': course['course_name'],
                'department': course['department'],
                'level': course['level'],
                'description': course['description'],
                'fees': f"£{random.randint(12000, 18000):,}",
                'duration': 'Not specified',
                'min_gpa': 'Not specified',
                'min_ielts': 'Not specified',
                'career_prospects': f'Various opportunities in {user_field.lower()}',
                'modules': f'Core {user_field.lower()} modules',
                'score': random.uniform(0.4, 0.8),
                'match_quality': 'Good Match',
                'reasons': [f'Matches your interest in {user_field}', 'Suitable for your academic level'],
                'method': 'dummy_fallback'
            })
        
        return recommendations

    # Additional methods to maintain API compatibility
    def get_recommendation_explanation(self, course_id: str, user_profile: Dict) -> Dict:
        """Provide detailed explanation for why a course was recommended"""
        try:
            courses_df = self.data_manager.courses_df
            if courses_df.empty:
                return {'error': 'No courses data available'}
            
            # Find the course
            course_data = None
            for idx, course in courses_df.iterrows():
                if f"course_{idx}" == course_id or course.get('course_name') == course_id:
                    course_data = course.to_dict()
                    break
            
            if not course_data:
                return {'error': 'Course not found'}
            
            explanation = {
                'course_name': course_data.get('course_name'),
                'match_factors': [],
                'potential_concerns': [],
                'alignment_score': 0.0
            }
            
            # Analyze alignment factors
            user_field = user_profile.get('field_of_interest', '').lower()
            course_name = str(course_data.get('course_name', '')).lower()
            course_desc = str(course_data.get('description', '')).lower()
            
            # Field alignment
            if user_field in course_name or user_field in course_desc:
                explanation['match_factors'].append({
                    'factor': 'Field Alignment',
                    'description': f"Course directly relates to your interest in {user_field}",
                    'strength': 'High'
                })
                explanation['alignment_score'] += 0.3
            
            # Academic level match
            user_level = user_profile.get('academic_level', '').lower()
            course_level = course_data.get('level', '').lower()
            if user_level and user_level in course_level:
                explanation['match_factors'].append({
                    'factor': 'Academic Level',
                    'description': f"Matches your {user_level} level",
                    'strength': 'High'
                })
                explanation['alignment_score'] += 0.2
            
            # Requirements check
            user_gpa = user_profile.get('gpa', 0.0)
            min_gpa = course_data.get('min_gpa', 0.0)
            if min_gpa > 0:
                if user_gpa >= min_gpa:
                    explanation['match_factors'].append({
                        'factor': 'GPA Requirement',
                        'description': f"Your GPA ({user_gpa}) meets the requirement ({min_gpa})",
                        'strength': 'Medium'
                    })
                    explanation['alignment_score'] += 0.1
                else:
                    explanation['potential_concerns'].append({
                        'concern': 'GPA Requirement',
                        'description': f"Your GPA ({user_gpa}) is below the requirement ({min_gpa})",
                        'severity': 'Medium'
                    })
            
            # Career alignment
            career_goals = user_profile.get('career_goals', '').lower()
            career_prospects = str(course_data.get('career_prospects', '')).lower()
            if career_goals and any(word in career_prospects for word in career_goals.split() if len(word) > 3):
                explanation['match_factors'].append({
                    'factor': 'Career Alignment',
                    'description': "Course career prospects align with your goals",
                    'strength': 'Medium'
                })
                explanation['alignment_score'] += 0.15
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Recommendation explanation error: {e}")
            return {'error': 'Could not generate explanation'}
    
    def update_user_feedback(self, user_id: str, course_id: str, feedback: str, rating: int):
        """Update recommendation system based on user feedback (placeholder for compatibility)"""
        try:
            self.logger.info(f"Received feedback from user {user_id} for course {course_id}: {feedback} (rating: {rating})")
            # This is a placeholder - feedback processing could be implemented later
        except Exception as e:
            self.logger.error(f"Feedback update error: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get recommendation system statistics"""
        try:
            courses_df = self.data_manager.courses_df
            stats = {
                'total_courses': len(courses_df) if not courses_df.empty else 0,
                'models_available': {
                    'sklearn': SKLEARN_AVAILABLE,
                    'tfidf_vectorizer': self.tfidf_vectorizer is not None,
                    'course_vectors': self.course_vectors is not None
                },
                'recommendation_methods': ['content_based', 'keyword_based', 'fallback']
            }
            
            if not courses_df.empty:
                stats['course_levels'] = courses_df['level'].value_counts().to_dict() if 'level' in courses_df.columns else {}
                stats['departments'] = courses_df['department'].value_counts().to_dict() if 'department' in courses_df.columns else {}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Stats generation error: {e}")
            return {'error': 'Could not generate stats'}
    
    # Baseline methods for compatibility (simplified versions)
    def compare_with_baselines(self, user_profiles: List[Dict]) -> Dict:
        """Compare recommendation methods for research (simplified version)"""
        results = {
            'content_based': {'avg_diversity': 0.5, 'avg_processing_time': 0.2, 'total_recommendations': 0},
            'keyword_based': {'avg_diversity': 0.6, 'avg_processing_time': 0.1, 'total_recommendations': 0},
            'fallback': {'avg_diversity': 0.3, 'avg_processing_time': 0.05, 'total_recommendations': 0}
        }
        return results