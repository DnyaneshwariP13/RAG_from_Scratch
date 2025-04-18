import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from .metrics import QAMetrics

class QAEvaluator:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.metrics = QAMetrics()
        self.results = []
        
    def load_test_cases(self, filepath: str) -> List[Dict]:
        """Load evaluation test cases"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def evaluate(self, test_cases_path: str) -> pd.DataFrame:
        """Run full evaluation pipeline"""
        test_cases = self.load_test_cases(test_cases_path)
        
        for case in tqdm(test_cases):
            try:
                pred_answer = self.qa_chain.invoke(case["question"])
            except Exception as e:
                pred_answer = ""
                
            metrics = {
                "exact_match": self.metrics.exact_match(pred_answer, case["reference_answer"]),
                "f1_score": self.metrics.f1_score(pred_answer, case["reference_answer"]),
                "semantic_similarity": self.metrics.semantic_similarity(
                    pred_answer, case["reference_answer"]
                ),
                "context_relevance": self.metrics.context_relevance(
                    pred_answer, case["context_keywords"]
                )
            }
            
            self.results.append({
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "predicted_answer": pred_answer,
                **metrics
            })
            
        return pd.DataFrame(self.results)
    
    def summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate aggregate metrics"""
        return {
            "num_cases": len(df),
            "avg_exact_match": df["exact_match"].mean(),
            "avg_f1_score": df["f1_score"].mean(),
            "avg_semantic_sim": df["semantic_similarity"].mean(),
            "avg_context_rel": df["context_relevance"].mean()
        }