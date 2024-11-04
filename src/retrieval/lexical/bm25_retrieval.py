import bm25s
import numpy as np
from src.preprocessor.utils.sent_level import is_numerical
from src.preprocessor.preprocessor import TextPreprocessing
from src.preprocessor.post_preprocessor import PostPreprocessing

class BM25sRetrieval:
    # Định nghĩa các phương thức BM25 được hỗ trợ
    SUPPORTED_METHODS = {'robertson', 'bm25+', 'bm25l'}
    
    def __init__(self,
                 query: str,
                 method: str = 'bm25l',
                 k1: float = 1.5,
                 b: float = 0.75,
                 delta: float = 1.5,
                 top_k: int = 5):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
            
        self.query = query
        self.method = method
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.top_k = top_k
        self.text_preprocessor = TextPreprocessing()
        self.text_post_preprocessor = PostPreprocessing()
        
    @staticmethod
    def corpus2token(corpus):
        return bm25s.tokenize(corpus)
    
    def tokenize_query(self):
        processed_query = self.text_post_preprocessor.post_preprocess_text(self.text_preprocessor.preprocess_text(self.query))
        return bm25s.tokenize(processed_query)
    
    def create_retriever(self):
        params = {'method': self.method, 'k1': self.k1, 'b': self.b}
        if self.method != 'robertson':
            params['delta'] = self.delta
        return bm25s.BM25(**params)
    
    def retrieval_module(self, corpus):
        retriever = self.create_retriever()
        
        # Indexing
        corpus_tokens = self.corpus2token(corpus)
        retriever.index(corpus_tokens)
        
        # Query processing and retrieval
        query_tokens = self.tokenize_query()
        results, scores  =  retriever.retrieve(query_tokens=query_tokens, 
                                corpus=corpus, 
                                k=self.top_k)
        
        results = [doc for doc in results[0]]
        scores = [score for score in scores[0]]
        # Reranking 
        query_tokens = list(query_tokens.vocab.keys())
        scores = self.rerank_with_term_coverage(scores, results, query_tokens)
    
        return results, scores
        
    
    def rerank_with_term_coverage(self, scores, results, query_tokens):
        """
        Tái xếp hạng kết quả dựa trên độ phủ của các từ query
        """
        
        corpus_tokens = [self.text_preprocessor.preprocess_text(i) for i in results]
        coverage_scores = []
        for _, doc_tokens in zip(scores, corpus_tokens):
            # Tính tỷ lệ từ query có trong document
            coverage = sum(1 for term in query_tokens if term in doc_tokens and term is not is_numerical(term)) / len(query_tokens)
            # Điều chỉnh điểm số dựa trên coverage
            coverage_scores.append(coverage)
        
        rerank_scores = []
        for score, coverage_score in zip(scores, coverage_scores):
            score = score*np.exp(np.log(coverage_score)/0.38)
            rerank_scores.append(score)
        return rerank_scores