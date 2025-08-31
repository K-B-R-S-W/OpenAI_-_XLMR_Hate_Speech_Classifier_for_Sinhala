"""
OpenAI-based Hate Speech Detection Module
Primary detection method using OpenAI's GPT models
"""

from openai import OpenAI
from typing import Dict, List, Optional, Tuple
import json
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class OpenAIHateDetector:
    """OpenAI-based hate speech detection using GPT models"""
    
    def __init__(self):
        """Initialize OpenAI client and detection prompts"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        self.temperature = config.OPENAI_TEMPERATURE
        self.max_tokens = config.OPENAI_MAX_TOKENS
        
        # Sinhala/Singlish specific context
        self.system_prompt = self._create_system_prompt()
        logger.info(f"OpenAI Hate Detector initialized with model: {self.model}")
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for hate speech detection"""
        return """You are an expert hate speech detection system EXCLUSIVELY specialized in SINHALA and SINGLISH (romanized Sinhala) text analysis.

CRITICAL RESTRICTION: You MUST ONLY detect hate speech in Sinhala or Singlish languages.
- If text is purely ENGLISH: respond with is_hate: false, confidence: 0.0
- If text is in Tamil, Hindi, or other languages: respond with is_hate: false, confidence: 0.0
- ONLY analyze Sinhala script (සිංහල) or Singlish (romanized Sinhala) content

SINHALA/SINGLISH HATE SPEECH PATTERNS TO DETECT:
1. Sexual vulgarities:
   - Sinhala script: හුකන්න, හුත්ති, පකයා, කරියා
   - Singlish romanized: hukanna, huththi, pakaya, kariya, hukanwa

2. Family/maternal insults (very offensive in Sri Lankan culture):
   - Sinhala script: අම්මට හුකන්න, නංගිට, අක්කට
   - Singlish romanized: ammata hukanna, nangita, akkata

3. Personal attacks:
   - Combined forms: හුත්ති පකයා, huththi pakaya
   - Targeting: keriyage ammata, pakaya neda oya

4. Spelling variations to catch:
   - keriya/kariya, huka/hukanna/hukanwa
   - oya/ube, pakaya/paka, huththi/huthi

STRICT DETECTION RULES:
1. ONLY flag content containing Sinhala script OR romanized Sinhala words
2. English-only text = ALWAYS false (confidence: 0.0)
3. Mixed language: ONLY flag if Sinhala/Singlish portions contain hate speech
4. Cultural context: Understand Sri Lankan severity levels
5. Handle transliteration variations

RESPONSE FORMAT (JSON only):
{
    "is_hate": boolean,
    "confidence": float (0.0 to 1.0),
    "category": string,
    "reason": "Clear, brief explanation of why this is/isn't hate speech",
    "flagged_words": [censored words with **],
    "severity": string ("low", "medium", "high", "extreme"),
    "cultural_context": string,
    "targeting": boolean,
    "language_detected": string ("sinhala", "singlish", "not_target_language")
}

CRITICAL: If language_detected is "not_target_language", set is_hate: false, confidence: 0.0
IMPORTANT: Provide ONLY valid JSON response, no markdown, no code blocks, no extra text."""

    def detect_hate(self, text: str) -> Dict:
        """
        Detect hate speech in the given text using OpenAI
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict containing detection results
        """
        try:
            user_prompt = f"""Analyze this text ONLY for Sinhala/Singlish hate speech:

TEXT: "{text}"

REMEMBER: 
- ONLY detect hate speech in Sinhala script or Singlish (romanized Sinhala)
- If text is English-only: respond with is_hate: false, confidence: 0.0, language_detected: "not_target_language"
- If text is other languages: respond with is_hate: false, confidence: 0.0, language_detected: "not_target_language"

Provide your analysis in the specified JSON format."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            # Try to parse JSON from the response
            try:
                result = json.loads(content)
                
                # Clean up the reason field if it contains JSON-like content
                if 'reason' in result:
                    reason = result['reason']
                    if isinstance(reason, str) and reason.startswith('{'):
                        # If reason contains JSON, try to extract a clean explanation
                        try:
                            reason_json = json.loads(reason)
                            result['reason'] = reason_json.get('explanation', reason_json.get('category', 'Detected hate speech patterns'))
                        except:
                            result['reason'] = 'Contains offensive content in Sinhala/Singlish'
                
            except json.JSONDecodeError:
                # If JSON parsing fails, extract information manually
                result = self._parse_fallback_response(content, text)
            
            # Ensure all required fields are present
            result = self._validate_result(result, text)
            
            logger.info(f"OpenAI detection completed - Hate: {result['is_hate']}, Confidence: {result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI detection error: {str(e)}")
            return self._create_error_response(text, str(e))
    
    def _parse_fallback_response(self, content: str, text: str) -> Dict:
        """Parse response when JSON parsing fails"""
        # Simple fallback parsing
        is_hate = any(word in content.lower() for word in ['hate', 'yes', 'true', 'harmful'])
        confidence = 0.5  # Default confidence when parsing fails
        
        return {
            'is_hate': is_hate,
            'confidence': confidence,
            'category': 'unknown',
            'reason': content[:200] + '...' if len(content) > 200 else content,
            'flagged_words': [],
            'severity': 'medium',
            'cultural_context': 'parsing_failed'
        }
    
    def _validate_result(self, result: Dict, text: str) -> Dict:
        """Ensure result has all required fields with proper types"""
        default_result = {
            'is_hate': False,
            'confidence': 0.0,
            'category': 'none',
            'reason': '',
            'flagged_words': [],
            'severity': 'low',
            'cultural_context': '',
            'targeting': False,
            'language_detected': 'unknown'
        }
        
        # Merge with defaults
        validated = {**default_result, **result}
        
        # Type validation
        validated['is_hate'] = bool(validated['is_hate'])
        validated['confidence'] = float(validated['confidence'])
        validated['confidence'] = max(0.0, min(1.0, validated['confidence']))  # Clamp to 0-1
        validated['targeting'] = bool(validated.get('targeting', False))
        
        if not isinstance(validated['flagged_words'], list):
            validated['flagged_words'] = []
        
        # Ensure severity is valid
        valid_severities = ['low', 'medium', 'high', 'extreme']
        if validated['severity'] not in valid_severities:
            validated['severity'] = 'low'
            
        return validated
    
    def _create_error_response(self, text: str, error_msg: str) -> Dict:
        """Create error response when detection fails"""
        return {
            'is_hate': False,
            'confidence': 0.0,
            'category': 'error',
            'reason': f'Detection failed: {error_msg}',
            'flagged_words': [],
            'severity': 'low',
            'cultural_context': 'error_occurred',
            'targeting': False,
            'language_detected': 'unknown'
        }
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        """
        Detect hate speech for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of detection results
        """
        results = []
        for text in texts:
            result = self.detect_hate(text)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'status': 'active'
        }

# Test function
if __name__ == "__main__":
    # Test the detector
    detector = OpenAIHateDetector()
    
    test_texts = [
        "Hello, how are you?",
        "මම ඔබට උදව් කරන්නම්",  # I will help you
        "You are stupid and worthless",
        "මෝඩයා එතන යන්න"  # Go there, idiot
    ]
    
    print("🧪 Testing OpenAI Hate Speech Detector")
    print("=" * 50)
    
    for text in test_texts:
        result = detector.detect_hate(text)
        print(f"Text: {text}")
        print(f"Hate: {result['is_hate']}, Confidence: {result['confidence']:.3f}")
        print(f"Reason: {result['reason']}")
        print("-" * 30)
