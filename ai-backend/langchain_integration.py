"""
LangChain Integration for Enhanced Context Analysis
Provides advanced text processing and context understanding
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Dict, List, Optional
import logging
import config

logger = logging.getLogger(__name__)

class LangChainProcessor:
    """LangChain-based text processor for context analysis"""
    
    def __init__(self):
        """Initialize LangChain components"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for LangChain")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.LANGCHAIN_MODEL,
            temperature=config.LANGCHAIN_TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP
        )
        
        # Initialize prompts
        self.context_prompt = self._create_context_prompt()
        self.sentiment_prompt = self._create_sentiment_prompt()
        
        # Create chains
        self.context_chain = LLMChain(llm=self.llm, prompt=self.context_prompt)
        self.sentiment_chain = LLMChain(llm=self.llm, prompt=self.sentiment_prompt)
        
        logger.info("LangChain Processor initialized")
    
    def _create_context_prompt(self) -> PromptTemplate:
        """Create prompt template for context analysis"""
        template = """Analyze the following text for contextual indicators of hate speech or harassment.

Consider:
1. Cultural context (Sinhala/Singlish)
2. Implicit meanings and undertones
3. Social and historical context
4. Power dynamics and marginalization
5. Intent and potential harm

Text: {text}

Provide analysis in this format:
CONTEXT_SCORE: [0.0-1.0]
CULTURAL_INDICATORS: [list]
IMPLICIT_MEANINGS: [description]
POTENTIAL_HARM: [assessment]
CONFIDENCE: [0.0-1.0]
"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    def _create_sentiment_prompt(self) -> PromptTemplate:
        """Create prompt template for sentiment analysis"""
        template = """Analyze the sentiment and emotional tone of this text.

Focus on:
1. Primary emotion (anger, joy, sadness, fear, disgust, surprise)
2. Intensity level (low, medium, high)
3. Targeting behavior (is it directed at someone?)
4. Hostile indicators
5. Sinhala/Singlish emotional markers

Text: {text}

Response format:
PRIMARY_EMOTION: [emotion]
INTENSITY: [low/medium/high]
HOSTILITY_SCORE: [0.0-1.0]
TARGETING: [yes/no]
EMOTIONAL_MARKERS: [list]
"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    def analyze_context(self, text: str) -> Dict:
        """
        Analyze text context using LangChain
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict containing context analysis results
        """
        try:
            # Split text if it's too long
            documents = self.text_splitter.split_text(text)
            
            # Analyze each chunk (or just the first if splitting occurred)
            primary_text = documents[0] if documents else text
            
            # Run context analysis
            context_result = self.context_chain.run(text=primary_text)
            sentiment_result = self.sentiment_chain.run(text=primary_text)
            
            # Parse results
            context_analysis = self._parse_context_result(context_result)
            sentiment_analysis = self._parse_sentiment_result(sentiment_result)
            
            # Combine results
            combined_result = {
                **context_analysis,
                **sentiment_analysis,
                'text_length': len(text),
                'chunks_analyzed': len(documents)
            }
            
            # Calculate overall confidence
            combined_result['confidence'] = self._calculate_combined_confidence(combined_result)
            
            logger.info(f"LangChain analysis completed - Confidence: {combined_result['confidence']:.3f}")
            return combined_result
            
        except Exception as e:
            logger.error(f"LangChain analysis error: {str(e)}")
            return self._create_error_response(text, str(e))
    
    def _parse_context_result(self, result: str) -> Dict:
        """Parse context analysis result"""
        try:
            lines = result.strip().split('\n')
            parsed = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    if key == 'context_score' or key == 'confidence':
                        try:
                            parsed[key] = float(value.replace('[', '').replace(']', ''))
                        except ValueError:
                            parsed[key] = 0.5
                    else:
                        parsed[key] = value.replace('[', '').replace(']', '')
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Context parsing error: {str(e)}")
            return {'context_score': 0.5, 'cultural_indicators': '', 'confidence': 0.3}
    
    def _parse_sentiment_result(self, result: str) -> Dict:
        """Parse sentiment analysis result"""
        try:
            lines = result.strip().split('\n')
            parsed = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    if key == 'hostility_score':
                        try:
                            parsed[key] = float(value.replace('[', '').replace(']', ''))
                        except ValueError:
                            parsed[key] = 0.5
                    elif key == 'targeting':
                        parsed[key] = value.lower() in ['yes', 'true', '1']
                    else:
                        parsed[key] = value.replace('[', '').replace(']', '')
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Sentiment parsing error: {str(e)}")
            return {'primary_emotion': 'neutral', 'hostility_score': 0.5, 'targeting': False}
    
    def _calculate_combined_confidence(self, analysis: Dict) -> float:
        """Calculate combined confidence score"""
        context_conf = analysis.get('confidence', 0.5)
        context_score = analysis.get('context_score', 0.5)
        hostility_score = analysis.get('hostility_score', 0.5)
        
        # Weight the different components
        combined = (context_conf * 0.4 + context_score * 0.3 + hostility_score * 0.3)
        return max(0.0, min(1.0, combined))
    
    def _create_error_response(self, text: str, error_msg: str) -> Dict:
        """Create error response when analysis fails"""
        return {
            'confidence': 0.0,
            'context_score': 0.0,
            'hostility_score': 0.0,
            'primary_emotion': 'unknown',
            'targeting': False,
            'error': error_msg,
            'flags': ['analysis_failed']
        }
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that might indicate hate speech"""
        try:
            phrase_prompt = PromptTemplate(
                template="Extract key phrases from this text that might indicate hate speech or harassment: {text}\nKey phrases:",
                input_variables=["text"]
            )
            
            phrase_chain = LLMChain(llm=self.llm, prompt=phrase_prompt)
            result = phrase_chain.run(text=text)
            
            # Parse the result to extract phrases
            phrases = [phrase.strip() for phrase in result.split(',') if phrase.strip()]
            return phrases[:10]  # Limit to top 10 phrases
            
        except Exception as e:
            logger.error(f"Key phrase extraction error: {str(e)}")
            return []
    
    def get_cultural_context(self, text: str) -> Dict:
        """Get Sinhala/Singlish cultural context"""
        try:
            cultural_prompt = PromptTemplate(
                template="""Analyze this text for Sinhala/Singlish cultural context:

Text: {text}

Provide:
1. Language detection (Sinhala, Singlish, English, Mixed)
2. Cultural references
3. Local slang or colloquialisms
4. Regional context
5. Potential cultural sensitivities

Format as: LANGUAGE: [detection], CULTURAL_REFS: [list], SLANG: [list], CONTEXT: [description]""",
                input_variables=["text"]
            )
            
            cultural_chain = LLMChain(llm=self.llm, prompt=cultural_prompt)
            result = cultural_chain.run(text=text)
            
            # Parse result
            return self._parse_cultural_result(result)
            
        except Exception as e:
            logger.error(f"Cultural context error: {str(e)}")
            return {'language': 'unknown', 'cultural_refs': [], 'slang': []}
    
    def _parse_cultural_result(self, result: str) -> Dict:
        """Parse cultural context result"""
        try:
            parsed = {'language': 'unknown', 'cultural_refs': [], 'slang': [], 'context': ''}
            lines = result.strip().split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'language' in key:
                        parsed['language'] = value.replace('[', '').replace(']', '')
                    elif 'cultural' in key:
                        parsed['cultural_refs'] = [ref.strip() for ref in value.replace('[', '').replace(']', '').split(',')]
                    elif 'slang' in key:
                        parsed['slang'] = [s.strip() for s in value.replace('[', '').replace(']', '').split(',')]
                    elif 'context' in key:
                        parsed['context'] = value.replace('[', '').replace(']', '')
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Cultural parsing error: {str(e)}")
            return {'language': 'unknown', 'cultural_refs': [], 'slang': []}

# Test function
if __name__ == "__main__":
    processor = LangChainProcessor()
    
    test_texts = [
        "Hello, how are you today?",
        "මම ඔබට උදව් කරන්නම්",
        "You are so stupid!",
        "pakaya mokada"
    ]
    
    print("🧪 Testing LangChain Processor")
    print("=" * 50)
    
    for text in test_texts:
        result = processor.analyze_context(text)
        print(f"Text: {text}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Context Score: {result.get('context_score', 0):.3f}")
        print(f"Hostility: {result.get('hostility_score', 0):.3f}")
        print("-" * 30)
