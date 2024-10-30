from abc import ABC, abstractmethod

class BaseTextPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, docs):
        """
        Preprocess text documents.

        Args:
            docs: Input text or pandas Series

        Returns:
            Preprocessed text or Series
    """
        pass
    
    @abstractmethod
    def preprocess_text(self, paragraph: str):
        
        """
        Preprocess a single text paragraph.

        Args:
            paragraph: Input text string

        Returns:
            Preprocessed text string
        """
        pass

class BaseTextPostPreprocessor(ABC):
    @abstractmethod
    def post_preprocess_text(self, paragraph: str):
        """
        Post-preprocess a single text paragraph.

        Args:
            paragraph: Input text string

        Returns:
            Post-processed text string
        """
        pass

    @abstractmethod
    def post_preprocess(self, docs):
        """
        Post-preprocess text documents.

        Args:
            docs: Input text or pandas Series

        Returns:
            Post-processed text or Series
        """
        pass
