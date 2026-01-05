#!/usr/bin/env python3
"""
Course Recommendation Application
Standalone application that collects user information and provides course recommendations
"""

import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class CourseRecommendationApp:
    def __init__(self, csv_path: str):
        """Initialize the recommendation application"""
        self.csv_path = csv_path
        self.courses_df = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.course_vectors = None
        
        # Field keyword mappings for better matching
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
        
        self.load_courses()
    
    def load_courses(self):
        """Load courses from CSV file"""
        try:
            if not os.path.exists(self.csv_path):
                print(f"❌ Error: Could not find courses.csv at {self.csv_path}")
                print("Please make sure the file exists at the specified location.")
                return False
            
            print(f"📚 Loading courses from {self.csv_path}...")
            self.courses_df = pd.read_csv(self.csv_path)
            
            if self.courses_df.empty:
                print("❌ Error: The courses.csv file is empty!")
                return False
            
            print(f"✅ Successfully loaded {len(self.courses_df)} courses!")
            print(f"📊 Columns available: {list(self.courses_df.columns)}")
            
            # Prepare course texts for vectorization
            self._prepare_course_vectors()
            return True
            
        except Exception as e:
            print(f"❌ Error loading courses: {e}")
            return False
    
    def _prepare_course_vectors(self):
        """Prepare TF-IDF vectors for courses"""
        try:
            course_texts = []
            
            for _, row in self.courses_df.iterrows():
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
            if course_texts:
                self.course_vectors = self.tfidf_vectorizer.fit_transform(course_texts)
                print("✅ Course vectors prepared successfully!")
            else:
                print("⚠️ Warning: No course text data found for vectorization")
                
        except Exception as e:
            print(f"❌ Error preparing course vectors: {e}")
    
    def collect_user_info(self) -> Dict:
        """Collect user information through an interactive form"""
        print("\n" + "="*60)
        print("🎓 COURSE RECOMMENDATION SYSTEM")
        print("="*60)
        print("Please fill out the following information to get personalized course recommendations:")
        print()
        
        user_profile = {}
        
        # Basic Information
        print("📝 BASIC INFORMATION")
        print("-" * 30)
        user_profile['name'] = input("Your Name: ").strip()
        
        # Field of Interest
        print("\n🎯 FIELD OF INTEREST")
        print("-" * 30)
        print("Examples: Machine Learning, Business Management, Computer Science, Data Science, etc.")
        user_profile['field_of_interest'] = input("Primary Field of Interest: ").strip()
        
        # Academic Level
        print("\n🎓 ACADEMIC LEVEL")
        print("-" * 30)
        print("Options: 1) High School  2) Bachelor's  3) Master's  4) PhD  5) Other")
        choice = input("Select your current academic level (1-5): ").strip()
        
        level_map = {
            '1': 'high school',
            '2': 'bachelor\'s',
            '3': 'master\'s', 
            '4': 'phd',
            '5': 'other'
        }
        user_profile['academic_level'] = level_map.get(choice, 'bachelor\'s')
        
        # Preferred Study Level
        print("\n📚 PREFERRED STUDY LEVEL")
        print("-" * 30)
        print("Options: 1) Undergraduate  2) Postgraduate  3) PhD  4) Any")
        choice = input("What level would you like to study? (1-4): ").strip()
        
        study_map = {
            '1': 'undergraduate',
            '2': 'postgraduate',
            '3': 'phd',
            '4': 'any'
        }
        user_profile['preferred_study_level'] = study_map.get(choice, 'any')
        
        # GPA
        print("\n📊 ACADEMIC PERFORMANCE")
        print("-" * 30)
        while True:
            try:
                gpa_input = input("Your GPA (0.0-4.0, or press Enter to skip): ").strip()
                if gpa_input == "":
                    user_profile['gpa'] = 3.0  # Default
                    break
                gpa = float(gpa_input)
                if 0.0 <= gpa <= 4.0:
                    user_profile['gpa'] = gpa
                    break
                else:
                    print("Please enter a GPA between 0.0 and 4.0")
            except ValueError:
                print("Please enter a valid number")
        
        # IELTS Score
        while True:
            try:
                ielts_input = input("Your IELTS Score (0.0-9.0, or press Enter to skip): ").strip()
                if ielts_input == "":
                    user_profile['ielts_score'] = 6.5  # Default
                    break
                ielts = float(ielts_input)
                if 0.0 <= ielts <= 9.0:
                    user_profile['ielts_score'] = ielts
                    break
                else:
                    print("Please enter an IELTS score between 0.0 and 9.0")
            except ValueError:
                print("Please enter a valid number")
        
        # Career Goals
        print("\n🚀 CAREER GOALS")
        print("-" * 30)
        user_profile['career_goals'] = input("Describe your career goals (optional): ").strip()
        
        # Interests
        print("\n💡 INTERESTS & SKILLS")
        print("-" * 30)
        interests_input = input("List your interests (comma-separated, optional): ").strip()
        user_profile['interests'] = [i.strip() for i in interests_input.split(',') if i.strip()] if interests_input else []
        
        skills_input = input("List your professional skills (comma-separated, optional): ").strip()
        user_profile['professional_skills'] = [s.strip() for s in skills_input.split(',') if s.strip()] if skills_input else []
        
        # Budget
        print("\n💰 BUDGET PREFERENCES")
        print("-" * 30)
        while True:
            try:
                budget_input = input("Maximum budget for tuition (£, or press Enter for no limit): ").strip()
                if budget_input == "":
                    user_profile['budget_max'] = 100000  # No limit
                    break
                budget = float(budget_input.replace('£', '').replace(',', ''))
                user_profile['budget_max'] = budget
                break
            except ValueError:
                print("Please enter a valid budget amount")
        
        # Work Experience
        print("\n💼 WORK EXPERIENCE")
        print("-" * 30)
        while True:
            try:
                exp_input = input("Years of work experience (or press Enter for 0): ").strip()
                if exp_input == "":
                    user_profile['work_experience_years'] = 0
                    break
                exp = int(exp_input)
                if exp >= 0:
                    user_profile['work_experience_years'] = exp
                    break
                else:
                    print("Please enter a non-negative number")
            except ValueError:
                print("Please enter a valid number")
        
        print("\n✅ Profile completed successfully!")
        return user_profile
    
    def get_recommendations(self, user_profile: Dict, num_recommendations: int = 10) -> List[Dict]:
        """Generate course recommendations based on user profile"""
        if self.courses_df is None or self.courses_df.empty:
            return []
        
        print(f"\n🔍 Generating recommendations for {user_profile.get('name', 'you')}...")
        
        try:
            # Method 1: Content-based recommendations using TF-IDF
            content_recs = self._content_based_recommendations(user_profile)
            
            # Method 2: Keyword-based recommendations
            keyword_recs = self._keyword_based_recommendations(user_profile)
            
            # Combine and deduplicate recommendations
            all_recommendations = content_recs + keyword_recs
            
            # Remove duplicates and combine scores
            unique_recs = self._combine_recommendations(all_recommendations)
            
            # Apply filters and scoring
            final_recs = []
            for rec in unique_recs[:num_recommendations * 2]:  # Get more for filtering
                enhanced_rec = self._enhance_recommendation(rec, user_profile)
                if enhanced_rec:
                    final_recs.append(enhanced_rec)
            
            # Sort by final score and return top N
            final_recs.sort(key=lambda x: x['score'], reverse=True)
            return final_recs[:num_recommendations]
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return self._fallback_recommendations(user_profile, num_recommendations)
    
    def _content_based_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Generate recommendations using content-based filtering"""
        if self.course_vectors is None:
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
            for idx, similarity in enumerate(similarities):
                if idx < len(self.courses_df):
                    course = self.courses_df.iloc[idx]
                    recommendations.append({
                        'course_index': idx,
                        'course_data': course.to_dict(),
                        'score': float(similarity),
                        'method': 'content_based'
                    })
            
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:20]
            
        except Exception as e:
            print(f"⚠️ Content-based recommendation error: {e}")
            return []
    
    def _keyword_based_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Generate recommendations using keyword matching"""
        try:
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
            
            for idx, course in self.courses_df.iterrows():
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
            print(f"⚠️ Keyword-based recommendation error: {e}")
            return []
    
    def _create_user_text(self, user_profile: Dict) -> str:
        """Create text representation of user profile"""
        text_parts = []
        
        if user_profile.get('field_of_interest'):
            text_parts.append(user_profile['field_of_interest'])
        
        if user_profile.get('career_goals'):
            text_parts.append(user_profile['career_goals'])
        
        text_parts.extend(user_profile.get('interests', []))
        text_parts.extend(user_profile.get('professional_skills', []))
        
        if user_profile.get('academic_level'):
            text_parts.append(user_profile['academic_level'])
        
        return ' '.join(filter(None, text_parts))
    
    def _get_course_text(self, course) -> str:
        """Get text representation of a course"""
        text_parts = []
        
        fields = ['course_name', 'description', 'keywords', 'department', 'level', 'career_prospects', 'modules']
        
        for field in fields:
            if field in course and pd.notna(course[field]):
                text_parts.append(str(course[field]))
        
        return ' '.join(text_parts)
    
    def _combine_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Combine recommendations from different methods"""
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
            else:
                course_map[course_idx] = rec.copy()
        
        return list(course_map.values())
    
    def _enhance_recommendation(self, rec: Dict, user_profile: Dict) -> Dict:
        """Enhance recommendation with additional scoring and information"""
        try:
            course_data = rec['course_data']
            score = rec['score']
            reasons = []
            
            # Academic level matching
            user_level = user_profile.get('preferred_study_level', '').lower()
            course_level = str(course_data.get('level', '')).lower()
            
            if user_level != 'any' and user_level and course_level:
                if user_level in course_level:
                    score += 0.2
                    reasons.append(f"Matches preferred study level: {user_level}")
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
            budget_max = user_profile.get('budget_max', float('inf'))
            course_fees = course_data.get('fees_international', 0)
            
            if isinstance(course_fees, (int, float)) and course_fees > 0:
                if course_fees <= budget_max:
                    score += 0.05
                    reasons.append(f"Within budget (£{course_fees:,.0f})")
                else:
                    score *= 0.6
                    reasons.append(f"Above budget (£{course_fees:,.0f})")
            
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
            
            return {
                'course_name': course_data.get('course_name', 'Unknown Course'),
                'department': course_data.get('department', 'General Studies'),
                'level': course_data.get('level', 'undergraduate'),
                'description': course_data.get('description', 'No description available'),
                'fees': f"£{course_data.get('fees_international', 0):,.0f}" if course_data.get('fees_international') else "Not specified",
                'duration': course_data.get('duration', 'Not specified'),
                'min_gpa': course_data.get('min_gpa', 'Not specified'),
                'min_ielts': course_data.get('min_ielts', 'Not specified'),
                'career_prospects': course_data.get('career_prospects', 'Various opportunities'),
                'score': min(score, 1.0),
                'match_quality': match_quality,
                'reasons': reasons[:3],  # Limit to top 3 reasons
                'method': rec.get('method', 'combined')
            }
            
        except Exception as e:
            print(f"⚠️ Error enhancing recommendation: {e}")
            return None
    
    def _fallback_recommendations(self, user_profile: Dict, num_recommendations: int) -> List[Dict]:
        """Provide fallback recommendations when other methods fail"""
        try:
            if self.courses_df.empty:
                return []
            
            # Get random sample of courses
            sample_size = min(num_recommendations, len(self.courses_df))
            sample_courses = self.courses_df.sample(n=sample_size)
            
            recommendations = []
            for _, course in sample_courses.iterrows():
                recommendations.append({
                    'course_name': course.get('course_name', 'Unknown Course'),
                    'department': course.get('department', 'General Studies'),
                    'level': course.get('level', 'undergraduate'),
                    'description': course.get('description', 'No description available'),
                    'fees': f"£{course.get('fees_international', 0):,.0f}" if course.get('fees_international') else "Not specified",
                    'duration': course.get('duration', 'Not specified'),
                    'min_gpa': course.get('min_gpa', 'Not specified'),
                    'min_ielts': course.get('min_ielts', 'Not specified'),
                    'career_prospects': course.get('career_prospects', 'Various opportunities'),
                    'score': np.random.uniform(0.3, 0.7),
                    'match_quality': 'Possible Match',
                    'reasons': ['General recommendation', 'May align with your interests'],
                    'method': 'fallback'
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Fallback recommendation error: {e}")
            return []
    
    def display_recommendations(self, recommendations: List[Dict], user_profile: Dict):
        """Display recommendations in a formatted way"""
        if not recommendations:
            print("\n❌ No recommendations found. Please try adjusting your criteria.")
            return
        
        print(f"\n🎯 TOP COURSE RECOMMENDATIONS FOR {user_profile.get('name', 'YOU').upper()}")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['course_name']}")
            print("-" * (len(rec['course_name']) + 3))
            print(f"🏛️  Department: {rec['department']}")
            print(f"🎓 Level: {rec['level'].title()}")
            print(f"💰 Fees: {rec['fees']}")
            print(f"⏱️  Duration: {rec['duration']}")
            print(f"📊 Match Quality: {rec['match_quality']} ({rec['score']:.2f})")
            
            if rec.get('min_gpa') and rec['min_gpa'] != 'Not specified':
                print(f"📈 Min GPA: {rec['min_gpa']}")
            
            if rec.get('min_ielts') and rec['min_ielts'] != 'Not specified':
                print(f"🗣️  Min IELTS: {rec['min_ielts']}")
            
            print(f"📝 Description: {rec['description'][:150]}{'...' if len(rec['description']) > 150 else ''}")
            
            if rec.get('reasons'):
                print(f"✅ Why recommended: {', '.join(rec['reasons'])}")
            
            print(f"🚀 Career Prospects: {rec['career_prospects'][:100]}{'...' if len(rec['career_prospects']) > 100 else ''}")
            
            if i < len(recommendations):
                print()
    
    def save_results(self, user_profile: Dict, recommendations: List[Dict], filename: str = None):
        """Save recommendations to a file"""
        if not filename:
            name = user_profile.get('name', 'user').replace(' ', '_')
            filename = f"course_recommendations_{name}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"COURSE RECOMMENDATIONS FOR {user_profile.get('name', 'USER').upper()}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("USER PROFILE:\n")
                f.write("-" * 20 + "\n")
                for key, value in user_profile.items():
                    if key != 'name':
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
                
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 20 + "\n\n")
                
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec['course_name']}\n")
                    f.write(f"   Department: {rec['department']}\n")
                    f.write(f"   Level: {rec['level']}\n")
                    f.write(f"   Fees: {rec['fees']}\n")
                    f.write(f"   Duration: {rec['duration']}\n")
                    f.write(f"   Match Score: {rec['score']:.2f} ({rec['match_quality']})\n")
                    f.write(f"   Description: {rec['description']}\n")
                    if rec.get('reasons'):
                        f.write(f"   Reasons: {', '.join(rec['reasons'])}\n")
                    f.write("\n")
            
            print(f"✅ Results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def run(self):
        """Run the complete recommendation application"""
        print("🎓 Welcome to the Course Recommendation System!")
        
        # Load courses
        if not self.load_courses():
            print("❌ Cannot proceed without course data. Please check your CSV file.")
            return
        
        # Collect user information
        user_profile = self.collect_user_info()
        
        # Generate recommendations
        recommendations = self.get_recommendations(user_profile)
        
        # Display results
        self.display_recommendations(recommendations, user_profile)
        
        # Ask if user wants to save results
        print("\n" + "="*60)
        save_choice = input("Would you like to save these recommendations to a file? (y/n): ").lower().strip()
        
        if save_choice in ['y', 'yes']:
            self.save_results(user_profile, recommendations)
        
        print("\n🎉 Thank you for using the Course Recommendation System!")
        print("Good luck with your studies! 📚")


def main():
    """Main function to run the application"""
    # Path to your courses CSV file
    csv_path = "/Users/muhammadahmed/Downloads/uel-enhanced-ai-assistant/data/courses.csv"
    
    # Create and run the application
    app = CourseRecommendationApp(csv_path)
    app.run()


if __name__ == "__main__":
    main()