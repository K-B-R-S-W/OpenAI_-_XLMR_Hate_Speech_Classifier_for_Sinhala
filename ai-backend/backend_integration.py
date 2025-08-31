"""
Backend Integration Client for XLM-RoBERTa Service
Handles communication with the RoBERTa backend service
"""

import requests
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class RoBERTaBackendClient:
    """Client for communicating with XLM-RoBERTa backend service"""
    
    def __init__(self, backend_url: str = None):
        """
        Initialize backend client
        
        Args:
            backend_url: URL of the RoBERTa backend service
        """
        self.backend_url = backend_url or config.ROBERTA_BACKEND_URL
        self.timeout = 30  # seconds
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Backend-Client/1.0'
        })
        
        logger.info(f"RoBERTa Backend Client initialized: {self.backend_url}")
    
    def health_check(self) -> Dict:
        """Check if RoBERTa backend is healthy"""
        try:
            response = self.session.get(
                f"{self.backend_url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'backend_response': response.json()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'error': response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend health check failed: {str(e)}")
            return {
                'status': 'unreachable',
                'error': str(e)
            }
    
    def detect_hate(self, text: str) -> Dict:
        """
        Send text to RoBERTa backend for hate speech detection
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict containing detection results from RoBERTa
        """
        try:
            payload = {
                'text': text
            }
            
            response = self.session.post(
                f"{self.backend_url}/detect",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._process_roberta_result(result)
            else:
                logger.error(f"Backend detection failed: {response.status_code} - {response.text}")
                return self._create_error_response(text, f"Backend error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend request failed: {str(e)}")
            return self._create_error_response(text, str(e))
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        """
        Send multiple texts to RoBERTa backend for batch processing
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of detection results
        """
        try:
            payload = {
                'texts': texts
            }
            
            response = self.session.post(
                f"{self.backend_url}/batch_detect",
                json=payload,
                timeout=self.timeout * 2  # Longer timeout for batch
            )
            
            if response.status_code == 200:
                result = response.json()
                results = result.get('results', [])
                return [self._process_roberta_result(r) for r in results]
            else:
                logger.error(f"Backend batch detection failed: {response.status_code}")
                return [self._create_error_response(text, "Batch detection failed") for text in texts]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend batch request failed: {str(e)}")
            return [self._create_error_response(text, str(e)) for text in texts]
    
    def send_feedback(self, feedback_data: Dict) -> bool:
        """
        Send feedback to RoBERTa backend
        
        Args:
            feedback_data: User feedback data
            
        Returns:
            True if feedback sent successfully
        """
        try:
            # Determine feedback endpoint based on type
            if feedback_data.get('actual_label') == 'HATE' and feedback_data.get('predicted_label') == 'NOT_HATE':
                endpoint = '/feedback/hate_missed'
                payload = {
                    'text': feedback_data['text'],
                    'words': feedback_data.get('user_notes', '')
                }
            elif feedback_data.get('actual_label') == 'NOT_HATE' and feedback_data.get('predicted_label') == 'HATE':
                endpoint = '/feedback/false_positive'
                payload = {
                    'text': feedback_data['text'],
                    'words': feedback_data.get('user_notes', '')
                }
            else:
                # General feedback
                endpoint = '/feedback/general'
                payload = feedback_data
            
            response = self.session.post(
                f"{self.backend_url}{endpoint}",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info("Feedback sent to RoBERTa backend successfully")
                return True
            else:
                logger.error(f"Feedback sending failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Feedback request failed: {str(e)}")
            return False
    
    def get_backend_stats(self) -> Dict:
        """Get statistics from RoBERTa backend"""
        try:
            response = self.session.get(
                f"{self.backend_url}/stats",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Stats request failed: {response.status_code}'}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend stats request failed: {str(e)}")
            return {'error': str(e)}
    
    def trigger_retrain(self, params: Dict = None) -> Dict:
        """
        Trigger model retraining on RoBERTa backend
        
        Args:
            params: Training parameters (epochs, learning_rate, etc.)
            
        Returns:
            Dict containing retraining result
        """
        try:
            payload = params or {
                'epochs': 2,
                'lr': 2e-5,
                'batch_size': 8
            }
            
            response = self.session.post(
                f"{self.backend_url}/admin/retrain",
                json=payload,
                timeout=300  # 5 minutes timeout for training
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Model retraining triggered successfully")
                return result
            else:
                logger.error(f"Retrain request failed: {response.status_code}")
                return {'error': f'Retrain failed: {response.status_code}'}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Retrain request failed: {str(e)}")
            return {'error': str(e)}
    
    def _process_roberta_result(self, result: Dict) -> Dict:
        """
        Process and normalize RoBERTa backend result
        
        Args:
            result: Raw result from RoBERTa backend
            
        Returns:
            Processed and normalized result
        """
        try:
            # Extract confidence score
            confidence = 0.0
            if 'confidence' in result:
                confidence = float(result['confidence'])
            elif 'word_probabilities' in result:
                # Calculate average confidence from word probabilities
                probs = result['word_probabilities']
                if probs:
                    hate_probs = []
                    for prob in probs:
                        if isinstance(prob, list) and len(prob) > 1:
                            hate_probs.append(prob[1])  # Hate class probability
                        elif isinstance(prob, (int, float)):
                            hate_probs.append(prob)
                    
                    if hate_probs:
                        confidence = sum(hate_probs) / len(hate_probs)
            
            # Normalize result
            normalized = {
                'sentence': result.get('sentence', ''),
                'sentence_label': result.get('sentence_label', 'NOT_HATE'),
                'confidence': round(confidence, 3),
                'hate_words': result.get('hate_words', []),
                'word_predictions': result.get('word_predictions', []),
                'word_probabilities': result.get('word_probabilities', []),
                'highlighted_hate_words': result.get('highlighted_hate_words', []),
                'source': 'roberta_backend'
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error processing RoBERTa result: {str(e)}")
            return self._create_error_response(result.get('sentence', ''), str(e))
    
    def _create_error_response(self, text: str, error_msg: str) -> Dict:
        """Create error response when backend communication fails"""
        return {
            'sentence': text,
            'sentence_label': 'NOT_HATE',
            'confidence': 0.0,
            'hate_words': [],
            'word_predictions': [],
            'word_probabilities': [],
            'highlighted_hate_words': [],
            'source': 'roberta_backend',
            'error': error_msg
        }
    
    def is_backend_available(self) -> bool:
        """Check if backend is available"""
        health = self.health_check()
        return health['status'] == 'healthy'
    
    def get_connection_info(self) -> Dict:
        """Get connection information"""
        return {
            'backend_url': self.backend_url,
            'timeout': self.timeout,
            'available': self.is_backend_available()
        }

# Test function
if __name__ == "__main__":
    # Test the client (requires running RoBERTa backend)
    client = RoBERTaBackendClient()
    
    print("🧪 Testing RoBERTa Backend Client")
    print("=" * 50)
    
    # Health check
    health = client.health_check()
    print(f"Backend Health: {health['status']}")
    
    if health['status'] == 'healthy':
        # Test detection
        test_texts = [
            "Hello, how are you?",
            "You are stupid!",
            "මෝඩයා එතන යන්න"
        ]
        
        for text in test_texts:
            result = client.detect_hate(text)
            print(f"Text: {text}")
            print(f"Result: {result['sentence_label']}, Confidence: {result['confidence']:.3f}")
            print("-" * 30)
    else:
        print("Backend not available for testing")
