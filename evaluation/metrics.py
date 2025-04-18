import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class QAMetrics:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
    def exact_match(self, pred: str, ref: str) -> float:
        """Calculate exact match score"""
        return float(pred.strip().lower() == ref.strip().lower())
    
    def f1_score(self, pred: str, ref: str) -> float:
        """Calculate word-level F1 score"""
        vectorizer = TfidfVectorizer().fit_transform([pred, ref])
        pred_tokens = set(vectorizer[0].indices)
        ref_tokens = set(vectorizer[1].indices)
        
        tp = len(pred_tokens & ref_tokens)
        fp = len(pred_tokens - ref_tokens)
        fn = len(ref_tokens - pred_tokens)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    def semantic_similarity(self, pred: str, ref: str) -> float:
        """Calculate cosine similarity between embeddings"""
        pred_embed = self.embeddings.embed_query(pred)
        ref_embed = self.embeddings.embed_query(ref)
        return np.dot(pred_embed, ref_embed) / (
            np.linalg.norm(pred_embed) * np.linalg.norm(ref_embed)
        )
    
    def context_relevance(self, answer: str, keywords: list) -> float:
        """Check if answer contains context-specific keywords"""
        answer_lower = answer.lower()
        return sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords)