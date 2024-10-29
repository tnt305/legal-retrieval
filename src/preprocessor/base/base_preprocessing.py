from abc import ABC, abstractmethod

class BaseTextPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, docs):
        pass
    
    @abstractmethod
    def preprocess_text(self, paragraph: str):
        pass

class BaseTextPostPreprocessor(ABC):
    @abstractmethod
    def post_preprocess(self, docs):
        pass
    
    @abstractmethod
    def post_preprocess_text(self, paragraph: str):
        pass