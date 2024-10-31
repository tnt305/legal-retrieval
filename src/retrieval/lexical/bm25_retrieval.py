# import bm25s
# from src.preprocessor.preprocessor import TextPreprocessing


# class BM25sRetrieval:
    
#     def __init__(self,
#                  query: str,
#                  method: str = 'bm25l',
#                  k1: float = 1.2,
#                  b:float = 0.75,
#                  delta:float = 0.5,
#                  top_k: int = 5):
#         assert method in ['robertson', 'bm25+', 'bm25l']
#         self.query = query
#         self.method = method
#         self.k1 = k1 # mặc định nên để 1.2 - 1.5
#         self.b = b # mặc định nên để 0.75
#         self.delta= delta
#         self.top_k = top_k
        
#     @staticmethod
#     def corpus2token(corpus):
#         corpus_token = bm25s.tokenize(corpus)
#         return corpus_token
 
#     def tokenize_query(self, query):
#         query = TextPreprocessing().preprocess_text(self.query)
#         query_tokens = bm25s.tokenize(query)
#         return query_tokens

    
#     def retrieval_module(self, corpus, query):
#         if self.method == 'robertson':
#             retriever = bm25s.BM25(method = self.method, k1 = self.k1, b = self.b)
#         else:
#             retriever = bm25s.BM25(method = self.method, k1 = self.k1, b = self.b , delta= self.delta)

#         # Indexing
#         corpus_tokens = self.corpus2token(corpus)
#         retriever.index(corpus_tokens)
        
#         query_tokens = self.tokenize_query(query)
#         results, scores = retriever.retrieve(query_tokens = query_tokens, corpus = corpus, k = self.top_k)
#         return results, scores


import bm25s
from src.preprocessor.preprocessor import TextPreprocessing
from src.preprocessor.post_preprocessor import PostPreprocessing

class BM25sRetrieval:
    # Định nghĩa các phương thức BM25 được hỗ trợ
    SUPPORTED_METHODS = {'robertson', 'bm25+', 'bm25l'}
    
    def __init__(self,
                 query: str,
                 method: str = 'bm25l',
                 k1: float = 1.2,
                 b: float = 0.75,
                 delta: float = 0.5,
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
        processed_query = self.text_preprocessor.preprocess_text(self.query)
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
        return retriever.retrieve(query_tokens=query_tokens, 
                                corpus=corpus, 
                                k=self.top_k)