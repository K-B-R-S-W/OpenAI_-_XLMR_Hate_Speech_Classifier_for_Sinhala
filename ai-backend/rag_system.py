"""
RAG (Retrieval-Augmented Generation) System for Hate Speech Detection
Uses knowledge base and embeddings for pattern matching
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import config
from sinhala_detector import SinhalaHateDetector

logger = logging.getLogger(__name__)

class RAGHateDetection:
    """RAG-based hate speech detection using knowledge base and embeddings"""
    
    def __init__(self):
        """Initialize RAG system with knowledge base and embeddings"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for embeddings")
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embeddings_model = config.RAG_EMBEDDINGS_MODEL
        self.top_k = config.RAG_TOP_K
        
        # Initialize specialized Sinhala detector
        self.sinhala_detector = SinhalaHateDetector()
        
        # Load or create knowledge base
        self.knowledge_base = self._load_knowledge_base()
        self.embeddings_cache = {}
        
        # Initialize hate patterns
        self._initialize_hate_patterns()
        
        logger.info(f"RAG System initialized with {len(self.knowledge_base)} patterns")
    
    def _load_knowledge_base(self) -> List[Dict]:
        """Load hate speech knowledge base"""
        kb_file = config.RAG_KNOWLEDGE_BASE
        
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default knowledge base
            default_kb = self._create_default_knowledge_base()
            self._save_knowledge_base(default_kb)
            return default_kb
    
    def _create_default_knowledge_base(self) -> List[Dict]:
        """Create default knowledge base with hate speech patterns"""
        patterns = [
            # General hate patterns
            {
                "id": "racial_slur_1",
                "pattern": "racial slurs and ethnic discrimination",
                "category": "racial_hatred",
                "severity": "high",
                "language": "english",
                "examples": ["racial slurs", "ethnic discrimination", "racist language"],
                "confidence": 0.9
            },
            {
                "id": "threat_1",
                "pattern": "violent threats and intimidation",
                "category": "threats",
                "severity": "high",
                "language": "english",
                "examples": ["I will hurt you", "you deserve to die", "kill yourself"],
                "confidence": 0.95
            },
            
            # Sinhala specific patterns
            {
                "id": "sinhala_curse_1",
                "pattern": "සිංහල අපශබ්ද වචන",
                "category": "profanity",
                "severity": "medium",
                "language": "sinhala",
                "examples": ["හුත්තො", "පකයා", "වේසියා"],
                "confidence": 0.8
            },
            {
                "id": "sinhala_insult_1",
                "pattern": "පෞරුෂත්වයට අභියෝග කරන වචන",
                "category": "personal_attacks",
                "severity": "medium",
                "language": "sinhala",
                "examples": ["මෝඩයා", "බල්ලා", "ගහනවා"],
                "confidence": 0.7
            },
            
            # Singlish patterns
            {
                "id": "singlish_1",
                "pattern": "singlish offensive terms",
                "category": "mixed_language_hate",
                "severity": "medium",
                "language": "singlish",
                "examples": ["pakaya mokada", "hora minihek", "ballage puthaa"],
                "confidence": 0.75
            },
            
            # Cyberbullying patterns
            {
                "id": "cyberbully_1",
                "pattern": "cyberbullying and harassment",
                "category": "harassment",
                "severity": "medium",
                "language": "english",
                "examples": ["nobody likes you", "you are worthless", "go kill yourself"],
                "confidence": 0.85
            },
            
            # Gender-based violence
            {
                "id": "gender_hate_1",
                "pattern": "gender-based harassment",
                "category": "gender_violence",
                "severity": "high",
                "language": "english",
                "examples": ["women belong in kitchen", "stupid female", "sexist remarks"],
                "confidence": 0.9
            },
            
            # Religious hatred
            {
                "id": "religious_1",
                "pattern": "religious intolerance",
                "category": "religious_hatred",
                "severity": "high",
                "language": "mixed",
                "examples": ["religious slurs", "infidel", "religious discrimination"],
                "confidence": 0.88
            }
        ]
        
        return patterns
    
    def _save_knowledge_base(self, knowledge_base: List[Dict]):
        """Save knowledge base to file"""
        os.makedirs(os.path.dirname(config.RAG_KNOWLEDGE_BASE), exist_ok=True)
        with open(config.RAG_KNOWLEDGE_BASE, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    def _initialize_hate_patterns(self):
        """Initialize hate pattern embeddings"""
        try:
            # Generate embeddings for all patterns
            for pattern in self.knowledge_base:
                pattern_text = f"{pattern['pattern']} {' '.join(pattern['examples'])}"
                pattern['embedding'] = self._get_embedding(pattern_text)
            
            logger.info("Hate pattern embeddings initialized")
            
        except Exception as e:
            logger.error(f"Error initializing pattern embeddings: {str(e)}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            # Check cache first
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]
            
            response = self.client.embeddings.create(
                model=self.embeddings_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            self.embeddings_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return [0.0] * 1536  # Default embedding size for ada-002
    
    def query_hate_patterns(self, text: str) -> Dict:
        """
        Query hate speech patterns using RAG + Sinhala detector
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict containing RAG analysis results
        """
        try:
            # First, use specialized Sinhala detector for direct pattern matching
            sinhala_result = self.sinhala_detector.detect_hate_patterns(text)
            
            # If Sinhala detector found patterns, use high confidence
            if sinhala_result['is_hate']:
                return {
                    'is_hate': True,
                    'confidence': sinhala_result['confidence'],
                    'matched_patterns': [{
                        'pattern_id': 'sinhala_direct',
                        'pattern': f"Direct Sinhala pattern detection: {sinhala_result['detected_patterns']}",
                        'category': 'sinhala_hate_patterns',
                        'severity': sinhala_result['severity'],
                        'similarity': sinhala_result['confidence'],
                        'confidence': sinhala_result['confidence'],
                        'language': sinhala_result['language'],
                        'flagged_words': sinhala_result['flagged_words']
                    }],
                    'categories': ['sinhala_hate_patterns'],
                    'max_severity': sinhala_result['severity'],
                    'total_matches': len(sinhala_result['detected_patterns']),
                    'significant_matches': len(sinhala_result['detected_patterns']),
                    'detection_method': 'sinhala_direct'
                }
            
            # Fall back to embedding-based RAG if no direct patterns found
            # Get embedding for input text
            text_embedding = self._get_embedding(text)
            
            # Calculate similarities with all patterns
            similarities = []
            for pattern in self.knowledge_base:
                if 'embedding' in pattern:
                    similarity = self._calculate_cosine_similarity(
                        text_embedding, pattern['embedding']
                    )
                    similarities.append((pattern, similarity))
            
            # Sort by similarity and get top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_matches = similarities[:self.top_k]
            
            # Analyze matches
            analysis_result = self._analyze_matches(text, top_matches)
            analysis_result['detection_method'] = 'embedding_rag'
            
            logger.info(f"RAG analysis completed - Matches: {len(top_matches)}, Confidence: {analysis_result['confidence']:.3f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            return self._create_error_response(text, str(e))
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]
        except Exception:
            return 0.0
    
    def _analyze_matches(self, text: str, matches: List[Tuple]) -> Dict:
        """Analyze pattern matches and determine hate speech likelihood"""
        if not matches:
            return {
                'is_hate': False,
                'confidence': 0.0,
                'matched_patterns': [],
                'categories': [],
                'max_severity': 'low'
            }
        
        # Extract match information
        matched_patterns = []
        categories = set()
        severities = []
        total_confidence = 0.0
        
        for pattern, similarity in matches:
            if similarity > 0.7:  # Threshold for significant similarity
                pattern_confidence = similarity * pattern.get('confidence', 0.5)
                matched_patterns.append({
                    'pattern_id': pattern['id'],
                    'pattern': pattern['pattern'],
                    'category': pattern['category'],
                    'severity': pattern['severity'],
                    'similarity': round(similarity, 3),
                    'confidence': round(pattern_confidence, 3),
                    'language': pattern['language']
                })
                
                categories.add(pattern['category'])
                severities.append(pattern['severity'])
                total_confidence += pattern_confidence
        
        # Determine overall results
        is_hate = len(matched_patterns) > 0 and total_confidence > 0.6
        avg_confidence = total_confidence / len(matched_patterns) if matched_patterns else 0.0
        
        # Determine maximum severity
        severity_order = {'low': 1, 'medium': 2, 'high': 3}
        max_severity = max(severities, key=lambda x: severity_order.get(x, 0)) if severities else 'low'
        
        return {
            'is_hate': is_hate,
            'confidence': round(avg_confidence, 3),
            'matched_patterns': matched_patterns,
            'categories': list(categories),
            'max_severity': max_severity,
            'total_matches': len(matches),
            'significant_matches': len(matched_patterns)
        }
    
    def _create_error_response(self, text: str, error_msg: str) -> Dict:
        """Create error response when RAG query fails"""
        return {
            'is_hate': False,
            'confidence': 0.0,
            'matched_patterns': [],
            'categories': [],
            'max_severity': 'low',
            'error': error_msg
        }
    
    def add_pattern(self, pattern: Dict) -> bool:
        """Add new hate speech pattern to knowledge base"""
        try:
            # Validate pattern
            required_fields = ['id', 'pattern', 'category', 'severity', 'examples']
            if not all(field in pattern for field in required_fields):
                return False
            
            # Generate embedding
            pattern_text = f"{pattern['pattern']} {' '.join(pattern['examples'])}"
            pattern['embedding'] = self._get_embedding(pattern_text)
            
            # Add to knowledge base
            self.knowledge_base.append(pattern)
            
            # Save to file
            self._save_knowledge_base(self.knowledge_base)
            
            logger.info(f"Added new pattern: {pattern['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding pattern: {str(e)}")
            return False
    
    def update_from_feedback(self, feedback_data: List[Dict]):
        """Update knowledge base based on user feedback"""
        try:
            new_patterns = []
            
            for feedback in feedback_data:
                if feedback.get('actual_label') == 'HATE' and feedback.get('predicted_label') == 'NOT_HATE':
                    # False negative - add as positive pattern
                    pattern = {
                        'id': f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'pattern': feedback['text'][:100],  # Truncate long texts
                        'category': 'user_feedback',
                        'severity': 'medium',
                        'language': 'mixed',
                        'examples': [feedback['text']],
                        'confidence': 0.8,
                        'source': 'user_feedback'
                    }
                    new_patterns.append(pattern)
            
            # Add new patterns
            for pattern in new_patterns:
                self.add_pattern(pattern)
            
            logger.info(f"Updated knowledge base with {len(new_patterns)} feedback patterns")
            return True
            
        except Exception as e:
            logger.error(f"Error updating from feedback: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        categories = {}
        languages = {}
        severities = {}
        
        for pattern in self.knowledge_base:
            # Count categories
            cat = pattern.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count languages
            lang = pattern.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
            
            # Count severities
            sev = pattern.get('severity', 'unknown')
            severities[sev] = severities.get(sev, 0) + 1
        
        return {
            'total_patterns': len(self.knowledge_base),
            'categories': categories,
            'languages': languages,
            'severities': severities,
            'cache_size': len(self.embeddings_cache),
            'model': self.embeddings_model
        }

# Test function
if __name__ == "__main__":
    rag_system = RAGHateDetection()
    
    test_texts = [
        "Hello, how are you?",
        "You are stupid and worthless",
        "මෝඩයා එතන යන්න",
        "pakaya mokada",
        "I hate all people from that country"
    ]
    
    print("🧪 Testing RAG Hate Detection System")
    print("=" * 50)
    
    for text in test_texts:
        result = rag_system.query_hate_patterns(text)
        print(f"Text: {text}")
        print(f"Hate: {result['is_hate']}, Confidence: {result['confidence']:.3f}")
        print(f"Matches: {result['significant_matches']}")
        if result['matched_patterns']:
            for match in result['matched_patterns'][:2]:  # Show top 2 matches
                print(f"  - {match['pattern']} (sim: {match['similarity']:.3f})")
        print("-" * 30)
