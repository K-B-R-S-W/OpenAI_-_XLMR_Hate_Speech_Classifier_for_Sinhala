"""
Configuration file for AI-Backend Service
Contains all configuration parameters for OpenAI, LangChain, RAG, and backend integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Service Configuration
DEBUG = True
HOST = '0.0.0.0'
AI_BACKEND_PORT = 8001
ROBERTA_BACKEND_PORT = 8002
FRONTEND_PORT = 3000

# Backend URLs
ROBERTA_BACKEND_URL = f'http://localhost:{ROBERTA_BACKEND_PORT}'

# Detection Thresholds
HATE_THRESHOLD = 0.6  # Confidence threshold for hate classification
OPENAI_CONFIDENCE_WEIGHT = 0.4
RAG_CONFIDENCE_WEIGHT = 0.3
CONTEXT_CONFIDENCE_WEIGHT = 0.2
ROBERTA_CONFIDENCE_WEIGHT = 0.1

# OpenAI Configuration
OPENAI_MODEL = "gpt-5-chat-latest"  # Latest available GPT-5 model
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 1500  # Increased for better analysis

# LangChain Configuration
LANGCHAIN_MODEL = "gpt-5-chat-latest"  # Using GPT-5 for enhanced context analysis
LANGCHAIN_TEMPERATURE = 0.2

# RAG Configuration
RAG_EMBEDDINGS_MODEL = "text-embedding-ada-002"
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 5

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEEDBACK_FILE = os.path.join(DATA_DIR, 'feedback.json')
RAG_KNOWLEDGE_BASE = os.path.join(DATA_DIR, 'hate_speech_knowledge.json')

# Hate Speech Patterns (for RAG system)
HATE_SPEECH_CATEGORIES = [
    'racial_slurs',
    'religious_hatred',
    'gender_based_violence',
    'homophobic_language',
    'ethnic_discrimination',
    'body_shaming',
    'threat_language',
    'cyberbullying'
]

# Sinhala/Singlish specific patterns
SINHALA_HATE_PATTERNS = [
    'curse_words',
    'derogatory_terms',
    'threat_expressions',
    'discriminatory_language'
]

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

print("📋 AI-Backend Configuration Loaded")
print(f"   OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Missing'}")
print(f"   AI Backend Port: {AI_BACKEND_PORT}")
print(f"   RoBERTa Backend URL: {ROBERTA_BACKEND_URL}")
print(f"   Hate Threshold: {HATE_THRESHOLD}")
