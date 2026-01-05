"""
UEL AI System - Data Management Module
"""

import os
import pandas as pd
from typing import List, Dict
from datetime import datetime

from config import config
from database_manager import DatabaseManager
from utils import get_logger

# Try to import optional libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DataManager:
    """Enhanced data management with robust CSV integration"""
    
    def __init__(self, data_dir: str = None):
        """Initialize data manager with real CSV files"""
        self.data_dir = data_dir or config.data_directory
        self.db_manager = DatabaseManager()
        self.logger = get_logger(f"{__name__}.DataManager")
        
        # Initialize empty DataFrames
        self.applications_df = pd.DataFrame()
        self.courses_df = pd.DataFrame()
        self.faqs_df = pd.DataFrame()
        self.counseling_df = pd.DataFrame()
        
        # Initialize ML components only if available
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.all_text_vectors = None
        else:
            self.vectorizer = None
            self.all_text_vectors = None
        
        self.combined_data = []
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"Data directory {self.data_dir} not found. Creating it...")
            try:
                os.makedirs(self.data_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Could not create data directory: {e}")
                self.data_dir = "."  # Fallback to current directory
        
        # Load data from CSV files
        self.load_all_data()
        
        # Create search index if possible
        if SKLEARN_AVAILABLE:
            try:
                self._create_search_index()
            except Exception as e:
                self.logger.warning(f"Could not create search index: {e}")
    
    def load_all_data(self):
        """Load all data from CSV files with robust error handling"""
        try:
            # Load CSV data with improved error handling
            self._load_csv_data_robust()
            self.logger.info("Data loading completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            # Create minimal sample data if loading fails
            self._create_minimal_sample_data()
    
    def _load_csv_data_robust(self):
        """Load CSV data files with robust error handling and flexible schemas"""
        csv_files = {
            'applications.csv': 'applications_df',
            'courses.csv': 'courses_df', 
            'faqs.csv': 'faqs_df',
            'counseling_slots.csv': 'counseling_df'
        }
    
        for filename, df_name in csv_files.items():
            csv_path = os.path.join(self.data_dir, filename)
        
            try:
                if os.path.exists(csv_path):
                    # Try to read CSV with different encodings
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    df = None
                    
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding)
                            self.logger.info(f"Loaded {filename} with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            self.logger.warning(f"Error reading {filename} with {encoding}: {e}")
                            continue
                    
                    if df is None:
                        self.logger.error(f"Could not read {filename} with any encoding")
                        continue
                    
                    # Clean column names
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
                    
                    # Store the dataframe
                    setattr(self, df_name, df)
                    self.logger.info(f"Loaded {len(df)} records from {filename}")
                
                    # Special handling for courses.csv
                    if filename == 'courses.csv':
                        self._process_courses_data(df)
                    
                    # Special handling for applications.csv
                    elif filename == 'applications.csv':
                        self._process_applications_data(df)
                    
                    # Special handling for faqs.csv
                    elif filename == 'faqs.csv':
                        self._process_faqs_data(df)
                        
                else:
                    self.logger.warning(f"{filename} not found at {csv_path}")
                    setattr(self, df_name, pd.DataFrame())
                
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")
                setattr(self, df_name, pd.DataFrame())
    
    def _process_courses_data(self, df):
        """Process and standardize courses data"""
        try:
            # Define required columns and their default values/mapping
            column_mappings = {
                'course_name': ['course_name', 'name'],
                'level': ['level', 'academic_level'],
                'description': ['description', 'course_description'],
                'department': ['department', 'school'],
                'duration': ['duration', 'course_duration'],
                'fees_domestic': ['fees_domestic', 'domestic_fees', 'uk_fees'],
                'fees_international': ['fees_international', 'international_fees', 'overseas_fees'],
                'min_gpa': ['min_gpa', 'gpa_requirement', 'gpa'],
                'min_ielts': ['min_ielts', 'ielts_requirement', 'ielts'],
                'trending_score': ['trending_score', 'popularity_score', 'trend'],
                'keywords': ['keywords', 'tags', 'search_terms'],
                'career_prospects': ['career_prospects', 'career_paths', 'job_opportunities'],
                'modules': ['modules', 'course_modules', 'curriculum']
            }
            
            # Process each required column
            for standard_col, possible_cols in column_mappings.items():
                found_col = None
                for p_col in possible_cols:
                    if p_col in df.columns:
                        found_col = p_col
                        break
                
                if found_col:
                    df[standard_col] = df[found_col].fillna('')
                    self.logger.info(f"Mapped '{found_col}' to '{standard_col}' in courses data")
                else:
                    # Set default values if column not found
                    if standard_col in ['fees_domestic', 'fees_international', 'min_gpa', 'min_ielts', 'trending_score']:
                        df[standard_col] = 0.0
                    elif standard_col == 'duration':
                        df[standard_col] = '1 year'
                    elif standard_col == 'level':
                        df[standard_col] = 'undergraduate'
                    else:
                        df[standard_col] = ''
                    self.logger.warning(f"Column '{standard_col}' not found in courses.csv, added with default values.")
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['fees_domestic', 'fees_international', 'min_gpa', 'min_ielts', 'trending_score']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            self.courses_df = df
            self.logger.info(f"Processed courses data - {len(df)} courses ready")
            
        except Exception as e:
            self.logger.error(f"Error processing courses data: {e}")
    
    def _process_applications_data(self, df):
        """Process and standardize applications data"""
        try:
            # Ensure required columns exist with fallbacks
            required_columns = {
                'name': 'Unknown Student',
                'applicant_name': 'Unknown Student', 
                'first_name': 'John',
                'last_name': 'Doe',
                'course_applied': 'General Studies',
                'status': 'under_review',
                'gpa': 3.0,
                'ielts_score': 6.5,
                'nationality': 'UK',
                'work_experience_years': 0,
                'application_date': datetime.now().strftime('%Y-%m-%d'),
                'current_education': 'undergraduate'
            }
            
            # Use existing columns or create with defaults
            for col, default_val in required_columns.items():
                if col not in df.columns:
                    # Try to map from similar column names
                    similar_cols = [c for c in df.columns if col.split('_')[0] in c]
                    if similar_cols:
                        df[col] = df[similar_cols[0]]
                        self.logger.info(f"Mapped '{similar_cols[0]}' to '{col}' in applications data")
                    else:
                        df[col] = default_val
                        self.logger.info(f"Added default column '{col}' to applications data")
                else:
                    # Fill missing values
                    df[col] = df[col].fillna(default_val)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['gpa', 'ielts_score', 'work_experience_years']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(required_columns[col])
            
            self.applications_df = df
            self.logger.info(f"Processed applications data - {len(df)} applications ready")
            
        except Exception as e:
            self.logger.error(f"Error processing applications data: {e}")
    
    def _process_faqs_data(self, df):
        """Process and standardize FAQs data"""
        try:
            # Ensure required columns exist
            if 'question' not in df.columns:
                # Try to find question-like columns
                question_cols = [c for c in df.columns if 'question' in c.lower() or 'q' in c.lower()]
                if question_cols:
                    df['question'] = df[question_cols[0]]
                else:
                    self.logger.warning("No question column found in FAQs data")
                    return
            
            if 'answer' not in df.columns:
                # Try to find answer-like columns
                answer_cols = [c for c in df.columns if 'answer' in c.lower() or 'a' in c.lower()]
                if answer_cols:
                    df['answer'] = df[answer_cols[0]]
                else:
                    self.logger.warning("No answer column found in FAQs data")
                    return
            
            # Remove rows with missing questions or answers
            df = df.dropna(subset=['question', 'answer'])
            
            self.faqs_df = df
            self.logger.info(f"Processed FAQs data - {len(df)} FAQs ready")
            
        except Exception as e:
            self.logger.error(f"Error processing FAQs data: {e}")
    
    def _create_minimal_sample_data(self):
        """Create minimal sample data if CSV loading fails"""
        self.logger.info("Creating minimal sample data as fallback...")
        
        # Sample courses
        self.courses_df = pd.DataFrame([
            {
                'course_name': 'Computer Science BSc',
                'level': 'undergraduate',
                'description': 'Comprehensive computer science program with focus on programming and software development.',
                'department': 'School of Computing',
                'duration': '3 years',
                'fees_domestic': 9250,
                'fees_international': 15000,
                'min_gpa': 3.0,
                'min_ielts': 6.0,
                'trending_score': 8.5,
                'keywords': 'programming, software, algorithms, data structures',
                'career_prospects': 'Software developer, systems analyst, data scientist',
                'modules': 'Introduction to Programming, Data Structures, Algorithms, Web Development, Database Systems'
            }
        ])
        
        # Sample applications
        self.applications_df = pd.DataFrame([
            {
                'name': 'John Smith',
                'course_applied': 'Computer Science BSc',
                'status': 'accepted',
                'gpa': 3.8,
                'ielts_score': 7.0,
                'nationality': 'UK',
                'work_experience_years': 2,
                'application_date': '2024-01-15',
                'current_education': 'undergraduate'
            }
        ])
        
        # Sample FAQs
        self.faqs_df = pd.DataFrame([
            {
                'question': 'What are the entry requirements?',
                'answer': 'Entry requirements vary by course. Generally we require IELTS 6.0-6.5 and relevant academic qualifications.'
            }
        ])
        
        self.logger.info("Minimal sample data created")
    
    def _create_search_index(self):
        """Create search index from loaded data"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available. Search functionality disabled.")
            return
        
        self.combined_data = []
        
        # Index courses from loaded data
        if not self.courses_df.empty:
            self.logger.info(f"Indexing {len(self.courses_df)} courses for search...")
            
            for _, course in self.courses_df.iterrows():
                # Build searchable text from available columns
                text_parts = []
                
                for col in ['course_name', 'description', 'department', 'keywords', 'level', 'modules']:
                    if col in course and pd.notna(course[col]):
                        text_parts.append(str(course[col]))
                
                search_text = ' '.join(text_parts).strip()
                
                if search_text:
                    self.combined_data.append({
                        'text': search_text,
                        'type': 'course',
                        'data': course.to_dict()
                    })
        
        # Index FAQs from loaded data
        if not self.faqs_df.empty:
            self.logger.info(f"Indexing {len(self.faqs_df)} FAQs for search...")
            
            for _, faq in self.faqs_df.iterrows():
                text_parts = []
                
                for col in ['question', 'answer']:
                    if col in faq and pd.notna(faq[col]):
                        text_parts.append(str(faq[col]))
                
                search_text = ' '.join(text_parts).strip()
                
                if search_text:
                    self.combined_data.append({
                        'text': search_text,
                        'type': 'faq',
                        'data': faq.to_dict()
                    })
        
        # Create TF-IDF vectors if we have data
        if self.combined_data:
            try:
                texts = [item['text'] for item in self.combined_data]
                self.all_text_vectors = self.vectorizer.fit_transform(texts)
                self.logger.info(f"Created search index with {len(self.combined_data)} items")
            except Exception as e:
                self.logger.error(f"Error creating TF-IDF vectors: {e}")
                self.all_text_vectors = None
        else:
            self.logger.warning("No data available for search indexing")
    
    def intelligent_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Intelligent search across all data"""
        if not SKLEARN_AVAILABLE or not self.combined_data or self.all_text_vectors is None:
            self.logger.warning("Search not available")
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.all_text_vectors).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    result = self.combined_data[idx].copy()
                    result['similarity'] = similarities[idx]
                    results.append(result)
            
            return results
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def get_data_stats(self) -> Dict:
        """Get comprehensive data statistics"""
        return {
            'courses': {
                'total': len(self.courses_df) if not self.courses_df.empty else 0,
                'columns': list(self.courses_df.columns) if not self.courses_df.empty else []
            },
            'applications': {
                'total': len(self.applications_df) if not self.applications_df.empty else 0,
                'columns': list(self.applications_df.columns) if not self.applications_df.empty else []
            },
            'faqs': {
                'total': len(self.faqs_df) if not self.faqs_df.empty else 0,
                'columns': list(self.faqs_df.columns) if not self.faqs_df.empty else []
            },
            'search_index': {
                'indexed_items': len(self.combined_data),
                'search_ready': self.all_text_vectors is not None
            }
        }