"""
AI-Backend: Primary OpenAI-based Hate Speech Detection Service
This service handles the main hate speech detection using OpenAI models,
with XLM-RoBERTa backend as fallback/secondary validation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

# Import custom modules
from openai_detector import OpenAIHateDetector
from langchain_integration import LangChainProcessor
from rag_system import RAGHateDetection
from backend_integration import RoBERTaBackendClient
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global instances (lazy initialization)
_openai_detector = None
_langchain_processor = None
_rag_system = None
_roberta_client = None

def get_openai_detector() -> OpenAIHateDetector:
    global _openai_detector
    if _openai_detector is None:
        _openai_detector = OpenAIHateDetector()
        logger.info("OpenAI Hate Detector initialized")
    return _openai_detector

def get_langchain_processor() -> LangChainProcessor:
    global _langchain_processor
    if _langchain_processor is None:
        _langchain_processor = LangChainProcessor()
        logger.info("LangChain Processor initialized")
    return _langchain_processor

def get_rag_system() -> RAGHateDetection:
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGHateDetection()
        logger.info("RAG System initialized")
    return _rag_system

def get_roberta_client() -> RoBERTaBackendClient:
    global _roberta_client
    if _roberta_client is None:
        _roberta_client = RoBERTaBackendClient(config.ROBERTA_BACKEND_URL)
        logger.info("RoBERTa Backend Client initialized")
    return _roberta_client

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ai-backend',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/detect', methods=['POST'])
def detect_hate_speech():
    """
    Primary hate speech detection endpoint
    Uses OpenAI as primary method, XLM-RoBERTa as secondary validation
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Step 1: OpenAI Detection (Primary)
        openai_detector = get_openai_detector()
        openai_result = openai_detector.detect_hate(text)
        
        # Step 2: LangChain Processing for context enhancement
        langchain_processor = get_langchain_processor()
        context_analysis = langchain_processor.analyze_context(text)
        
        # Step 3: RAG System for knowledge-based detection
        rag_system = get_rag_system()
        rag_result = rag_system.query_hate_patterns(text)
        
        # Step 4: XLM-RoBERTa Backend (Secondary/Validation)
        roberta_client = get_roberta_client()
        roberta_result = roberta_client.detect_hate(text)
        
        # Step 5: Aggregate Results
        final_result = aggregate_detection_results(
            text, openai_result, context_analysis, rag_result, roberta_result
        )
        
        logger.info(f"Detection completed for text length: {len(text)}")
        return jsonify(final_result)
        
    except Exception as e:
        logger.error(f"Error in hate detection: {str(e)}")
        return jsonify({
            'error': f'Detection failed: {str(e)}'
        }), 500

def aggregate_detection_results(text: str, openai_result: Dict, 
                              context_analysis: Dict, rag_result: Dict, 
                              roberta_result: Dict) -> Dict:
    """
    Aggregate results from all detection methods
    Priority: OpenAI > RAG > Context > RoBERTa
    """
    # Calculate confidence scores
    openai_confidence = openai_result.get('confidence', 0.0)
    rag_confidence = rag_result.get('confidence', 0.0)
    context_confidence = context_analysis.get('confidence', 0.0)
    roberta_confidence = roberta_result.get('confidence', 0.0)
    
    # Determine primary decision (OpenAI has highest weight)
    weights = {
        'openai': 0.4,
        'rag': 0.3,
        'context': 0.2,
        'roberta': 0.1
    }
    
    # Weighted score calculation
    weighted_score = (
        openai_confidence * weights['openai'] +
        rag_confidence * weights['rag'] +
        context_confidence * weights['context'] +
        roberta_confidence * weights['roberta']
    )
    
    # Final decision
    is_hate = weighted_score > config.HATE_THRESHOLD
    
    return {
        'text': text,
        'is_hate': is_hate,
        'confidence': round(weighted_score, 3),
        'methods': {
            'openai': {
                'is_hate': openai_result.get('is_hate', False),
                'confidence': round(openai_confidence, 3),
                'reason': openai_result.get('reason', ''),
                'flagged_words': openai_result.get('flagged_words', [])
            },
            'rag': {
                'is_hate': rag_result.get('is_hate', False),
                'confidence': round(rag_confidence, 3),
                'matched_patterns': rag_result.get('matched_patterns', [])
            },
            'context': {
                'sentiment': context_analysis.get('sentiment', 'neutral'),
                'confidence': round(context_confidence, 3),
                'context_flags': context_analysis.get('flags', [])
            },
            'roberta': {
                'is_hate': roberta_result.get('sentence_label') == 'HATE',
                'confidence': round(roberta_confidence, 3),
                'hate_words': roberta_result.get('hate_words', []),
                'word_predictions': roberta_result.get('word_predictions', [])
            }
        },
        'timestamp': datetime.utcnow().isoformat()
    }

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """Batch processing endpoint"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid texts format'}), 400
        
        results = []
        for text in texts:
            if text.strip():
                # Process each text through the main detection pipeline
                single_result = detect_hate_speech_internal(text.strip())
                results.append(single_result)
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch detection: {str(e)}")
        return jsonify({'error': f'Batch detection failed: {str(e)}'}), 500

def detect_hate_speech_internal(text: str) -> Dict:
    """Internal method for single text detection (used by batch)"""
    # Similar to detect_hate_speech but returns dict instead of Response
    openai_detector = get_openai_detector()
    langchain_processor = get_langchain_processor()
    rag_system = get_rag_system()
    roberta_client = get_roberta_client()
    
    openai_result = openai_detector.detect_hate(text)
    context_analysis = langchain_processor.analyze_context(text)
    rag_result = rag_system.query_hate_patterns(text)
    roberta_result = roberta_client.detect_hate(text)
    
    return aggregate_detection_results(text, openai_result, context_analysis, rag_result, roberta_result)

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """Collect user feedback for model improvement"""
    try:
        data = request.get_json()
        feedback_data = {
            'text': data.get('text'),
            'predicted_label': data.get('predicted_label'),
            'actual_label': data.get('actual_label'),
            'user_notes': data.get('user_notes', ''),
            'timestamp': datetime.utcnow().isoformat(),
            'ip_address': request.remote_addr
        }
        
        # Save feedback to file/database
        save_feedback(feedback_data)
        
        # Also send to RoBERTa backend for their feedback system
        roberta_client = get_roberta_client()
        roberta_client.send_feedback(feedback_data)
        
        return jsonify({'status': 'feedback_received'})
        
    except Exception as e:
        logger.error(f"Error collecting feedback: {str(e)}")
        return jsonify({'error': f'Feedback collection failed: {str(e)}'}), 500

def save_feedback(feedback_data: Dict):
    """Save feedback to JSON file"""
    feedback_file = config.FEEDBACK_FILE
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    # Load existing feedback
    existing_feedback = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            existing_feedback = json.load(f)
    
    # Append new feedback
    existing_feedback.append(feedback_data)
    
    # Save back
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(existing_feedback, f, ensure_ascii=False, indent=2)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        # Check RoBERTa backend health actively
        roberta_healthy = check_roberta_backend_health()
        
        stats = {
            'openai_status': _openai_detector is not None,
            'langchain_status': _langchain_processor is not None,
            'rag_status': _rag_system is not None,
            'roberta_backend_status': roberta_healthy,
            'config': {
                'hate_threshold': config.HATE_THRESHOLD,
                'roberta_backend_url': config.ROBERTA_BACKEND_URL
            }
        }
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': f'Failed to get stats: {str(e)}'}), 500

def check_roberta_backend_health():
    """Actively check if RoBERTa backend is responding"""
    try:
        import requests
        response = requests.get(f"{config.ROBERTA_BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"RoBERTa backend health check failed: {str(e)}")
        return False

if __name__ == '__main__':
    print("🚀 Starting AI-Backend Service...")
    print("🧠 Primary: OpenAI + LangChain + RAG")
    print("🤖 Secondary: XLM-RoBERTa Backend")
    print("=" * 50)
    
    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.AI_BACKEND_PORT
    )
