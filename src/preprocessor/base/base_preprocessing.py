from abc import ABC, abstractmethod

class BaseTextPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, docs):
        pass
    
    @abstractmethod
    def preprocess_text(self, paragraph: str):
        pass