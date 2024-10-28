import bm25s
from src.preprocessor.text_preprocessor import TextPreprocessing


class BM25sRetrieval():
    def __init__(self,
                 query: str,
                 method:str,
                 k1: float,
                 b:float,
                 delta:float,
                 top_k: int):
        assert method in ['robertson', 'bm25+', 'bm25l']
        self.query = query
        self.method = method
        self.k1 = k1 # mặc định nên để 1.2 - 1.5
        self.b = b # mặc định nên để 0.75
        self.delta= delta
        self.top_k = top_k
        
    @staticmethod
    def corpus2token(corpus):
        corpus_token = bm25s.tokenize(corpus)
        return corpus_token
 
    def tokenize_query(self, query):
        query = TextPreprocessing().preprocess(self.query)
        query_tokens = bm25s.tokenize(query)
        return query_tokens

    
    def retrieval_module(self, corpus, query):
        if self.method == 'robertson':
            retriever = bm25s.BM25(method = self.method, k1 = self.k1, b = self.b)
        else:
            retriever = bm25s.BM25(method = self.method, k1 = self.k1, b = self.b , delta= self.delta)

        # Indexing
        corpus_tokens = self.corpus2token(corpus)
        retriever.index(corpus_tokens)
        
        query_tokens = self.tokenize_query(query)
        results, scores = retriever.retrieve(query_tokens = query_tokens, corpus = corpus, k = self.top_k)
        return results, scores