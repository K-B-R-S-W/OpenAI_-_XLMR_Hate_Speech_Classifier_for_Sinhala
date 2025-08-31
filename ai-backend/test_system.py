#!/usr/bin/env python3
"""
Quick test script for the hate speech detection system
Tests with examples from user's successful ChatGPT conversation
"""

import sys
import os
import asyncio
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)

from sinhala_detector import SinhalaHateDetector
from openai_detector import OpenAIHateDetector
import config

async def test_sinhala_detector():
    """Test the Sinhala detector with known hate words"""
    print("\n=== Testing Sinhala Detector ===")
    
    detector = SinhalaHateDetector()
    
    # Test cases from user's ChatGPT conversation
    test_cases = [
        # Sinhala/Singlish hate speech (should be detected)
        "උබේ අම්මට හුකන්න දාලා",  # Direct hate speech in Sinhala
        "keriyage ammata hukanwa",    # Transliterated Singlish
        "oya hukanna one",           # Simple vulgar Singlish
        "pakaya neda oya",           # Insult in Singlish
        "huththi pakaya",           # Vulgar insult combination
        
        # Clean content (should NOT be detected)
        "hello how are you",         # Pure English
        "mama kohomada",             # Clean Sinhala
        "I hate this weather",       # English with "hate" but not hate speech
        "This is terrible",          # English negative sentiment
        "fuck this shit",           # English profanity (should NOT be flagged by Sinhala detector)
    ]
    
    for text in test_cases:
        result = detector.detect(text)
        print(f"Text: {text}")
        print(f"  Is Hate: {result['is_hate']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Detected patterns: {result.get('detected_patterns', [])}")
        print()

async def test_openai_detector():
    """Test OpenAI detector if API key is available - should ONLY flag Sinhala/Singlish"""
    print("\n=== Testing OpenAI Detector (Sinhala/Singlish Only) ===")
    
    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY not configured, skipping...")
        return
    
    try:
        detector = OpenAIHateDetector()
        
        # Test cases to verify Sinhala/Singlish-only detection
        test_cases = [
            # Should be flagged (Sinhala/Singlish hate speech)
            ("උබේ අම්මට හුකන්න දාලා", "Sinhala hate speech - should be flagged"),
            ("keriyage ammata hukanwa", "Singlish hate speech - should be flagged"),
            ("pakaya neda oya", "Singlish insult - should be flagged"),
            
            # Should NOT be flagged (English content)
            ("fuck you asshole", "English profanity - should NOT be flagged"),
            ("I hate you so much", "English with 'hate' - should NOT be flagged"),
            ("You are stupid", "English insult - should NOT be flagged"),
            ("hello how are you", "Clean English - should NOT be flagged"),
        ]
        
        for text, description in test_cases:
            result = await detector.detect(text)
            
            print(f"Text: {text}")
            print(f"  Description: {description}")
            print(f"  Is Hate: {result['is_hate']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Language Detected: {result.get('language_detected', 'unknown')}")
            print(f"  Reasoning: {result.get('reason', 'N/A')[:100]}...")
            print()
            
    except Exception as e:
        print(f"OpenAI test failed: {e}")

def main():
    """Run all tests"""
    print("Testing Enhanced Sinhala Hate Speech Detection System")
    print("=" * 60)
    
    # Test Sinhala detector (always available)
    asyncio.run(test_sinhala_detector())
    
    # Test OpenAI detector if configured
    asyncio.run(test_openai_detector())
    
    print("Testing complete!")

if __name__ == "__main__":
    main()
