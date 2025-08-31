"""
Enhanced Sinhala/Singlish Hate Word Detection
Specialized detector for Sinhala offensive language patterns
"""

import re
import json
from typing import Dict, List, Set, Tuple
import os
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

class SinhalaHateDetector:
    """Specialized detector for Sinhala/Singlish hate speech patterns"""
    
    def __init__(self):
        """Initialize with comprehensive Sinhala hate word patterns"""
        self.hate_patterns = self._load_hate_patterns()
        self.transliteration_map = self._create_transliteration_map()
        self.morphological_variants = self._create_morphological_variants()
        
        logger.info(f"Sinhala Hate Detector initialized with {len(self.hate_patterns)} patterns")
    
    def _load_hate_patterns(self) -> Dict:
        """Load comprehensive hate speech patterns"""
        return {
            # Core offensive terms (censored patterns)
            'sexual_vulgar': {
                'sinhala': ['හු**', 'කැ**', 'කි**'],
                'transliterated': ['hu**', 'huk**', 'kar**', 'kim**'],
                'variants': ['huka*', 'hukanwa', 'hukanna', 'keriya*', 'kariya*'],
                'severity': 'extreme',
                'confidence': 0.95
            },
            
            'personal_insults': {
                'sinhala': ['පක**', 'මෝ**', 'බල්**', 'ගහ**'],
                'transliterated': ['pak**', 'mo**', 'bal**', 'gah**'],
                'variants': ['pakaya', 'modaya', 'modayaa', 'ballaa', 'gahanna'],
                'severity': 'high',
                'confidence': 0.90
            },
            
            'family_insults': {
                'sinhala': ['අම්**', 'නං**', 'තාත්**'],
                'transliterated': ['amm**', 'nan**', 'tath**'],
                'variants': ['ammata', 'ammage', 'nangi', 'nangita', 'thaththata'],
                'severity': 'extreme',
                'confidence': 0.95
            },
            
            'derogatory_female': {
                'sinhala': ['වේ**', 'පොන්**', 'කුණු**'],
                'transliterated': ['we**', 'pon**', 'kun**'],
                'variants': ['wesiya', 'wesige', 'ponni', 'ponnaya', 'kunathu'],
                'severity': 'extreme',
                'confidence': 0.92
            },
            
            'threats_violence': {
                'sinhala': ['මර**', 'කප**', 'හප**', 'ගහ**'],
                'transliterated': ['mar**', 'kap**', 'hap**', 'gah**'],
                'variants': ['maranna', 'kapanna', 'hapanwa', 'gahanawa'],
                'severity': 'extreme',
                'confidence': 0.98
            },
            
            'body_shaming': {
                'sinhala': ['කුඩු**', 'මෝට**', 'කෑල්**'],
                'transliterated': ['kudu**', 'mot**', 'kael**'],
                'variants': ['kudusa', 'motaya', 'kaelaya'],
                'severity': 'medium',
                'confidence': 0.80
            }
        }
    
    def _create_transliteration_map(self) -> Dict:
        """Create mapping between Sinhala and English characters"""
        return {
            # Common Sinhala to English mappings
            'හ': 'h', 'ක': 'k', 'ග': 'g', 'ප': 'p', 'බ': 'b', 'ම': 'm',
            'න': 'n', 'ත': 't', 'ද': 'd', 'ල': 'l', 'ර': 'r', 'ව': 'w',
            'ස': 's', 'ය': 'y', 'ඒ': 'e', 'ඔ': 'o', 'ඉ': 'i', 'උ': 'u',
            'අ': 'a', 'ු': 'u', 'ා': 'aa', 'ි': 'i', 'ී': 'ii', 'ෙ': 'e',
            'ො': 'o', '්': '', 'ං': 'ng', 'ණ': 'n', 'ච': 'ch', 'ජ': 'j'
        }
    
    def _create_morphological_variants(self) -> Dict:
        """Create morphological variants for common patterns"""
        return {
            # Suffix patterns for Sinhala words
            'verb_suffixes': ['nwa', 'nawa', 'nna', 'nne', 'la', 'anna'],
            'noun_suffixes': ['ya', 'yaa', 'ge', 'ta', 'ata', 'age'],
            'adjective_suffixes': ['ka', 'kaa', 'ki', 'kii'],
            
            # Common spelling variations
            'spelling_variants': {
                'keriya': ['kariya', 'keriyaa', 'kariyaa'],
                'pakaya': ['pakayaa', 'pakai', 'pakka'],
                'modaya': ['modayaa', 'moda', 'modda'],
                'huka': ['hukka', 'hookaa', 'hukaa']
            }
        }
    
    def detect_hate_patterns(self, text: str) -> Dict:
        """
        Detect hate speech patterns in Sinhala/Singlish text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict containing detection results
        """
        text_lower = text.lower()
        detected_patterns = []
        flagged_words = []
        max_severity = 'low'
        total_confidence = 0.0
        
        # Check each pattern category
        for category, pattern_data in self.hate_patterns.items():
            matches = self._find_pattern_matches(text_lower, pattern_data)
            
            if matches:
                detected_patterns.append({
                    'category': category,
                    'matches': matches,
                    'severity': pattern_data['severity'],
                    'confidence': pattern_data['confidence']
                })
                
                flagged_words.extend([self._censor_word(match) for match in matches])
                
                # Update max severity
                if self._severity_level(pattern_data['severity']) > self._severity_level(max_severity):
                    max_severity = pattern_data['severity']
                
                total_confidence = max(total_confidence, pattern_data['confidence'])
        
        # Calculate final results
        is_hate = len(detected_patterns) > 0
        confidence = total_confidence if is_hate else 0.0
        
        # Language detection
        language = self._detect_language(text)
        
        return {
            'is_hate': is_hate,
            'confidence': confidence,
            'detected_patterns': detected_patterns,
            'flagged_words': list(set(flagged_words)),  # Remove duplicates
            'severity': max_severity,
            'language': language,
            'pattern_count': len(detected_patterns)
        }
    
    def _find_pattern_matches(self, text: str, pattern_data: Dict) -> List[str]:
        """Find matches for a specific pattern category"""
        matches = []
        
        # Check Sinhala script patterns
        for pattern in pattern_data.get('sinhala', []):
            # Convert pattern to regex (replace ** with wildcard)
            regex_pattern = pattern.replace('**', r'\w*')
            if re.search(regex_pattern, text):
                matches.append(pattern)
        
        # Check transliterated patterns
        for pattern in pattern_data.get('transliterated', []):
            regex_pattern = pattern.replace('**', r'\w*')
            if re.search(regex_pattern, text):
                matches.append(pattern)
        
        # Check variant patterns with fuzzy matching
        for variant in pattern_data.get('variants', []):
            if self._fuzzy_match(variant.replace('*', ''), text):
                matches.append(variant)
        
        return matches
    
    def _fuzzy_match(self, pattern: str, text: str, threshold: float = 0.8) -> bool:
        """Check for fuzzy string matching"""
        words = text.split()
        for word in words:
            if len(word) >= 3:  # Only check words with 3+ characters
                similarity = SequenceMatcher(None, pattern, word).ratio()
                if similarity >= threshold:
                    return True
        return False
    
    def _censor_word(self, word: str) -> str:
        """Censor detected hate word"""
        if len(word) <= 3:
            return word[0] + '**'
        else:
            return word[:2] + '**' + (word[-1] if len(word) > 4 else '')
    
    def _detect_language(self, text: str) -> str:
        """Detect the primary language of the text"""
        sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0D80-\u0DFF]', text))
        
        if total_chars == 0:
            return 'unknown'
        
        sinhala_ratio = sinhala_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if sinhala_ratio > 0.7:
            return 'sinhala'
        elif english_ratio > 0.7:
            return 'english' if not self._has_sinhala_transliteration(text) else 'singlish'
        elif sinhala_ratio > 0.3 and english_ratio > 0.3:
            return 'mixed'
        else:
            return 'singlish'
    
    def _has_sinhala_transliteration(self, text: str) -> bool:
        """Check if text contains transliterated Sinhala"""
        common_sinhala_transliterations = [
            'ka', 'ga', 'pa', 'ba', 'ma', 'na', 'ta', 'da', 'la', 'ra', 'wa',
            'ya', 'sa', 'ha', 'nge', 'nwa', 'kka', 'tta', 'nna'
        ]
        
        text_lower = text.lower()
        transliteration_count = sum(1 for trans in common_sinhala_transliterations if trans in text_lower)
        
        return transliteration_count >= 2
    
    def _severity_level(self, severity: str) -> int:
        """Convert severity to numeric level"""
        levels = {'low': 1, 'medium': 2, 'high': 3, 'extreme': 4}
        return levels.get(severity, 1)
    
    def add_custom_pattern(self, category: str, pattern_data: Dict) -> bool:
        """Add custom hate speech pattern"""
        try:
            if category not in self.hate_patterns:
                self.hate_patterns[category] = pattern_data
            else:
                # Merge with existing category
                for key in pattern_data:
                    if key in self.hate_patterns[category] and isinstance(pattern_data[key], list):
                        self.hate_patterns[category][key].extend(pattern_data[key])
                    else:
                        self.hate_patterns[category][key] = pattern_data[key]
            
            logger.info(f"Added custom pattern for category: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom pattern: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        total_patterns = 0
        severity_counts = {}
        
        for category, data in self.hate_patterns.items():
            pattern_count = (len(data.get('sinhala', [])) + 
                           len(data.get('transliterated', [])) + 
                           len(data.get('variants', [])))
            total_patterns += pattern_count
            
            severity = data.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_categories': len(self.hate_patterns),
            'total_patterns': total_patterns,
            'severity_distribution': severity_counts,
            'languages_supported': ['sinhala', 'singlish', 'english', 'mixed']
        }

# Test function
if __name__ == "__main__":
    detector = SinhalaHateDetector()
    
    # Test with your examples
    test_texts = [
        "උබේ අම්මට හුකන්න දාලා කැරි පොන්න වේසිගේ පුතා",  # Should detect multiple patterns
        "කොල්ලෝ උබ පොන්නයෙක් බන්",  # Should detect 'pon**'
        "ඒකිගේ කැරි කිම්බ බනි",  # Should detect 'ka**', 'kim**'
        "keriyage ammata hukanwa huththege putha",  # Transliterated
        "Hello, how are you today?",  # Safe text
        "මම ඔබට උදව් කරන්නම්"  # Safe Sinhala
    ]
    
    print("🧪 Testing Enhanced Sinhala Hate Detector")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        result = detector.detect_hate_patterns(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Hate: {result['is_hate']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Severity: {result['severity']}")
        print(f"   Language: {result['language']}")
        print(f"   Flagged words: {result['flagged_words']}")
        print(f"   Pattern categories: {[p['category'] for p in result['detected_patterns']]}")
    
    print(f"\n📊 Statistics: {detector.get_statistics()}")
