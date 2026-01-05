"""
UEL AI System - Utility Functions Module
"""

import logging
import random
from datetime import datetime
from typing import Dict, Any


def get_logger(name: str = __name__):
    """Get a logger instance"""
    return logging.getLogger(name)


def format_currency(amount: float, currency: str = "GBP") -> str:
    """Format currency amount"""
    symbol_map = {"GBP": "£", "USD": "$", "EUR": "€"}
    symbol = symbol_map.get(currency, currency)
    return f"{symbol}{amount:,.0f}"


def format_date(date_str: str) -> str:
    """Format date string"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y")
    except:
        pass
    return date_str


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_status_color(status: str) -> str:
    """Get color for status badges"""
    color_map = {
        'inquiry': '#3b82f6',
        'application_started': '#f59e0b',
        'under_review': '#6366f1',
        'accepted': '#10b981',
        'rejected': '#ef4444',
        'submitted': '#8b5cf6',
        'draft': '#6b7280',
        'verified': '#10b981',
        'pending': '#f59e0b'
    }
    return color_map.get(status, '#6b7280')


def get_level_color(level: str) -> str:
    """Get color for academic level badges"""
    color_map = {
        'undergraduate': '#3b82f6',
        'postgraduate': '#8b5cf6',
        'masters': '#10b981',
        'phd': '#ef4444'
    }
    return color_map.get(level.lower(), '#6b7280')


def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    try:
        return dictionary.get(key, default) if dictionary else default
    except:
        return default


def generate_sample_data():
    """Generate comprehensive sample data for demonstration"""
    # Sample students
    first_names = ["John", "Emma", "Michael", "Sophia", "William", "Olivia", "James", "Ava"]
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson"]
    countries = ["United Kingdom", "United States", "India", "China", "Nigeria", "Pakistan", "Canada"]
    fields = ["Computer Science", "Business Management", "Engineering", "Psychology", "Medicine", "Law"]
    
    students = []
    for i in range(20):
        first = random.choice(first_names)
        last = random.choice(last_names)
        students.append({
            'id': i + 1,
            'first_name': first,
            'last_name': last,
            'email': f"{first.lower()}.{last.lower()}{random.randint(1, 999)}@example.com",
            'country': random.choice(countries),
            'nationality': random.choice(countries),
            'field_of_interest': random.choice(fields),
            'academic_level': random.choice(['undergraduate', 'postgraduate', 'masters']),
            'status': random.choice(['inquiry', 'application_started', 'under_review', 'accepted']),
            'phone': f"+44 {random.randint(1000000000, 9999999999)}",
            'documents': []
        })
    
    # Sample courses
    course_names = ["Computer Science", "Business Management", "Data Science", "Engineering", "Psychology"]
    departments = ["School of Computing", "Business School", "School of Engineering", "School of Psychology"]
    
    courses = []
    for i, name in enumerate(course_names):
        courses.append({
            'id': i + 1,
            'course_name': name,
            'course_code': f"UEL{random.randint(1000, 9999)}",
            'department': random.choice(departments),
            'level': random.choice(['undergraduate', 'postgraduate', 'masters']),
            'duration': random.choice(['1 year', '2 years', '3 years']),
            'description': f"Comprehensive {name.lower()} program at UEL with excellent career prospects.",
            'fees': {
                'domestic': random.randint(9000, 12000),
                'international': random.randint(13000, 18000)
            }
        })
    
    return {
        'students': students,
        'courses': courses,
        'applications': [],
        'analytics': {
            'total_students': len(students),
            'active_applications': random.randint(15, 25),
            'total_courses': len(courses)
        }
    }