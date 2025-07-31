"""
Advanced text processing utilities for Spanish knowledge graph generation.
Implements intelligent entity filtering using NLP techniques.
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
import logging

# Optional imports - graceful fallback if not available
try:
    import spacy
    from spacy.lang.es.stop_words import STOP_WORDS as SPACY_STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_STOP_WORDS = set()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpanishTextProcessor:
    """
    Advanced text processor for Spanish knowledge graphs.
    Combines multiple NLP techniques for intelligent entity filtering.
    """
    
    def __init__(self, use_spacy: bool = True, use_tfidf: bool = True):
        """
        Initialize the text processor.
        
        Args:
            use_spacy: Whether to use spaCy for NLP (requires model installation)
            use_tfidf: Whether to use TF-IDF for term importance
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_tfidf = use_tfidf and SKLEARN_AVAILABLE
        self.nlp = None
        self.tfidf_vectorizer = None
        self.important_terms = set()
        
        # Initialize spaCy if available and requested
        if self.use_spacy:
            self._load_spacy_model()
        
        # Fallback stopwords for when spaCy is not available
        self.basic_stopwords = self._get_basic_spanish_stopwords()
        
        logger.info(f"SpanishTextProcessor initialized - spaCy: {self.use_spacy}, TF-IDF: {self.use_tfidf}")
    
    def _load_spacy_model(self):
        """Load spaCy Spanish model with fallback options."""
        models_to_try = ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"]
        
        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return
            except OSError:
                continue
        
        logger.warning("No spaCy Spanish model found. Install with: python -m spacy download es_core_news_sm")
        self.use_spacy = False
    
    def _get_basic_spanish_stopwords(self) -> Set[str]:
        """Get basic Spanish stopwords as fallback."""
        if SPACY_AVAILABLE:
            return SPACY_STOP_WORDS
        
        # Basic stopwords if spaCy not available
        return {
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "a", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante", 
            "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras",
            "y", "o", "pero", "sino", "aunque", "porque", "que", "si", "como", "cuando", "donde",
            "es", "son", "fue", "fueron", "está", "están", "ha", "han", "hay",
            "no", "ni", "sí", "también", "muy", "más", "menos", "todo", "toda", "todos", "todas"
        }
    
    def analyze_text_importance(self, texts: List[str], min_tfidf: float = 0.1) -> Set[str]:
        """
        Analyze text corpus to identify important terms using TF-IDF.
        
        Args:
            texts: List of text documents
            min_tfidf: Minimum TF-IDF score to consider term important
            
        Returns:
            Set of important terms
        """
        if not self.use_tfidf or not texts:
            return set()
        
        try:
            # Configure TF-IDF vectorizer for Spanish
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=None,  # We'll handle stopwords separately
                ngram_range=(1, 3),  # Include 1-3 word phrases
                min_df=2,  # Term must appear in at least 2 documents
                max_df=0.8,  # Term must appear in less than 80% of documents
                lowercase=True,
                token_pattern=r'\b[a-záéíóúñü]+\b'  # Spanish characters
            )
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get terms with high TF-IDF scores
            important_terms = set()
            for i, feature in enumerate(feature_names):
                max_score = tfidf_matrix[:, i].max()
                if max_score > min_tfidf and not self._is_stopword(feature):
                    important_terms.add(feature)
            
            self.important_terms = important_terms
            logger.info(f"Identified {len(important_terms)} important terms via TF-IDF")
            return important_terms
            
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}")
            return set()
    
    def _is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.lower() in self.basic_stopwords
    
    def extract_meaningful_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract meaningful entities from text using NLP techniques.
        
        Args:
            text: Input text
            
        Returns:
            List of tuples (entity, type, confidence_score)
        """
        entities = []
        
        if self.use_spacy and self.nlp:
            entities.extend(self._extract_with_spacy(text))
        else:
            entities.extend(self._extract_with_rules(text))
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract entities using spaCy NER and POS tagging."""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"]:
                confidence = 0.9  # High confidence for NER
                entities.append((ent.text.strip(), f"NER_{ent.label_}", confidence))
        
        # Extract important nouns and proper nouns
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN"] and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 2 and
                token.text.isalpha()):
                
                # Calculate confidence based on various factors
                confidence = self._calculate_token_confidence(token)
                if confidence > 0.3:  # Minimum confidence threshold
                    entities.append((token.lemma_, f"POS_{token.pos_}", confidence))
        
        return entities
    
    def _extract_with_rules(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract entities using rule-based approach (fallback)."""
        entities = []
        
        # Simple regex patterns for common entities
        patterns = {
            'PERSON': r'\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\b',
            'ORGANIZATION': r'\b(Universidad|Instituto|Ministerio|Empresa|Corporación)\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü\s]+\b',
            'LOCATION': r'\b(España|México|Argentina|Chile|Colombia|Venezuela|Perú|Ecuador|Bolivia)\b',
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append((match.strip(), f"RULE_{entity_type}", 0.7))
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]{2,}\b', text)
        for word in capitalized_words:
            if not self._is_stopword(word):
                entities.append((word, "RULE_PROPER", 0.5))
        
        return entities
    
    def _calculate_token_confidence(self, token) -> float:
        """Calculate confidence score for a token based on various factors."""
        confidence = 0.0
        
        # Base confidence by POS tag
        if token.pos_ == "PROPN":
            confidence += 0.6
        elif token.pos_ == "NOUN":
            confidence += 0.4
        
        # Length bonus
        if len(token.text) > 4:
            confidence += 0.2
        elif len(token.text) > 6:
            confidence += 0.3
        
        # Capitalization bonus
        if token.text[0].isupper():
            confidence += 0.2
        
        # TF-IDF importance bonus
        if token.lemma_.lower() in self.important_terms:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def is_meaningful_entity(self, entity: str, context: str = "") -> bool:
        """
        Determine if an entity is meaningful for knowledge graph inclusion.
        
        Args:
            entity: The entity to evaluate
            context: Optional context for better evaluation
            
        Returns:
            True if entity should be included in knowledge graph
        """
        entity = entity.strip()
        
        # Basic filters
        if len(entity) < 2:
            return False
        
        if entity.isdigit():
            return False
        
        if not any(c.isalpha() for c in entity):
            return False
        
        # Stopword filter (but allow if part of important terms)
        if (self._is_stopword(entity.lower()) and 
            entity.lower() not in self.important_terms):
            return False
        
        # Check against extracted meaningful entities if we have spaCy
        if self.use_spacy and context:
            meaningful_entities = self.extract_meaningful_entities(context)
            entity_lower = entity.lower()
            for ent, _, confidence in meaningful_entities:
                if entity_lower == ent.lower() and confidence > 0.4:
                    return True
        
        # Additional semantic filters
        if self._is_generic_term(entity):
            return False
        
        return True
    
    def _is_generic_term(self, entity: str) -> bool:
        """Check if entity is too generic to be useful."""
        generic_terms = {
            "cosa", "cosas", "algo", "alguien", "persona", "personas",
            "lugar", "lugares", "tiempo", "veces", "forma", "manera",
            "parte", "partes", "tipo", "tipos", "clase", "clases",
            "número", "números", "cantidad", "cantidades"
        }
        
        return entity.lower() in generic_terms
    
    def filter_triples_intelligently(self, triples: List[Dict], 
                                   source_texts: List[str] = None) -> List[Dict]:
        """
        Filter triples using intelligent NLP-based approach.
        
        Args:
            triples: List of triple dictionaries
            source_texts: Optional source texts for TF-IDF analysis
            
        Returns:
            List of filtered triples
        """
        if source_texts and self.use_tfidf:
            self.analyze_text_importance(source_texts)
        
        filtered_triples = []
        
        for triple in triples:
            subject = triple.get("subject", "").strip()
            obj = triple.get("object", "").strip()
            predicate = triple.get("predicate", "").strip()
            
            # Create context for entity evaluation
            context = f"{subject} {predicate} {obj}"
            
            # Check if both subject and object are meaningful
            if (self.is_meaningful_entity(subject, context) and 
                self.is_meaningful_entity(obj, context) and
                subject.lower() != obj.lower()):  # Avoid self-references
                
                filtered_triples.append(triple)
        
        logger.info(f"Filtered {len(triples)} -> {len(filtered_triples)} triples using intelligent filtering")
        return filtered_triples

# Singleton instance for easy access
text_processor = SpanishTextProcessor()

def get_text_processor() -> SpanishTextProcessor:
    """Get the global text processor instance."""
    return text_processor