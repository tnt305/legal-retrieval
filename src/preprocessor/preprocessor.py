import re
import string
import pandas as pd
from tqdm import tqdm
from typing import Union
from pyvi import ViTokenizer
from underthesea import ner

from src.preprocessor.vocab.stopwords import STOP_WORDS
from src.preprocessor.vocab.legal_dict import LEGAL_DICT
from src.preprocessor.vocab.duties_dict import DUTIES
from src.preprocessor.vocab.special_terms import SPECIAL_TERMS
from src.preprocessor.vocab.numeral_currency import CURRENCY
from src.preprocessor.utils import dupplicated_char_remover, preprocess_pyvi, postprocess_pyvi
from src.preprocessor.legal_processing.legal_terms_tokenize import terms_of_law
from src.preprocessor.legal_processing.duties_tokenize import duties_terms, ner_tokenize
from src.preprocessor.base.base_preprocessing import BaseTextPreprocessor

class TextPreprocessing(BaseTextPreprocessor):
    def __init__(self, 
                 legal_term: dict = None, 
                 stop_words: dict = None, 
                 duty_term: dict = None,
                 special_term: dict = None):
        self.legal_term = LEGAL_DICT if legal_term is None else legal_term
        self.stop_words = STOP_WORDS if stop_words is None else stop_words 
        self.duties = DUTIES if duty_term is None else duty_term
        self.special_terms = SPECIAL_TERMS if special_term is None else special_term
    def preprocess(self,
                  docs: Union[pd.Series, str],
                  url_remover: bool = True,
                  punctuation_remover: bool = True, 
                  line_breaker_remover: bool = True,
                  lowercase_standardizer: bool = False,
                  white_space_remover: bool = True,
                  text_tokenizer: bool = True,
                  law_text_recognizer: bool = True,
                  stop_word_remover: bool = True) -> Union[pd.Series, str]:
        """
        Preprocess text documents.
        
        Args:
            docs: Input text or pandas Series
            [preprocessing flags...]
            
        Returns:
            Preprocessed text or Series
        """
        if isinstance(docs, pd.Series):
            tqdm.pandas(desc="Pre-processing")
            return docs.progress_apply(
                lambda t: self.preprocess_text(
                    str(t), 
                    url_remover,
                    punctuation_remover,
                    line_breaker_remover,
                    lowercase_standardizer, 
                    white_space_remover,
                    text_tokenizer,
                    law_text_recognizer,
                    stop_word_remover
                )
            )
        return self.preprocess_text(
            docs,
            url_remover,
            punctuation_remover, 
            line_breaker_remover,
            lowercase_standardizer,
            white_space_remover, 
            text_tokenizer,
            law_text_recognizer,
            stop_word_remover
        )

    @staticmethod
    def _url_remover(paragraph: str) -> str:
        """Remove URLs within parentheses from text."""
        pattern = r'\([^)]*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[^)]*\)'
        
        def replace_url(match):
            content = match.group(0)
            cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
            return cleaned if cleaned.strip('() ') else ''
            
        return re.sub(pattern, replace_url, paragraph)

    @staticmethod
    def _punctuation_remover(paragraph: str) -> str:
        """Remove punctuation marks from text and handle currency conversions."""
        # Loại bỏ dấu ngoặc khỏi số
        paragraph = re.sub(r"\((\d+)\)", r"\1", paragraph)
        paragraph = re.sub(r"\w+\)", " ", paragraph)

        # Tách các từ trong đoạn văn
        words = paragraph.split()
        updated_words = []

        # Xử lý các từ liên quan đến số và tiền tệ
        for item in words:
            # Nếu từ kết thúc bằng ')' hoặc '.' và bắt đầu bằng số từ 1-9, loại bỏ từ đó
            if item.endswith((')', '.')) and item[0] in '123456789':
                continue

            # Thay thế chính xác từ khớp với từ điển CURRENCY
            if item in CURRENCY:
                updated_words.append(CURRENCY[item])
                continue

            # Tìm và thay thế các trường hợp chứa mẫu trong từ điển CURRENCY
            for key, value in CURRENCY.items():
                if key in item:
                    item = item.replace(key, f" {value}")
                    break

            updated_words.append(item)

        # Kết hợp lại đoạn văn đã được cập nhật
        paragraph = ' '.join(updated_words)

        # Loại bỏ các dấu câu (trừ '/' và '.')
        for punc in string.punctuation:
            if punc == ":":
                paragraph = paragraph.replace(punc, ".")
            elif punc == '-':
                paragraph = paragraph.replace(punc, "")
            elif punc not in ["/", "."]:
                paragraph = paragraph.replace(punc, " ")

        # Loại bỏ khoảng trắng thừa
        return re.sub(r"\s+", " ", paragraph).strip()

    @staticmethod 
    def _line_breaker_remover(paragraph: str) -> str:
        """Remove line breaks from text."""
        para = re.sub(r"\\n+", ". ", paragraph)
        para = re.sub(r"\n+", ". ", paragraph)
        para = re.sub(r"\.\.\.", " ", para)
        para = re.sub(r'\.{1,}', '.', para)
        return para.replace("  ", " ")

    @staticmethod
    def _lowercase_standardizer(paragraph: str) -> str:
        """Convert text to lowercase."""
        return paragraph.lower()

    @staticmethod
    def _white_space_remover(paragraph: str) -> str:
        """Remove extra whitespace from text."""
        para = paragraph.replace("  ", " ")
        return re.sub(r"\s{2,}", " ", para).strip()

    def _legal_text_tokenizer(self, paragraph: str) -> str:
        """Tokenize legal terms in text."""
        
        for phrase, replacement in self.legal_term.items():
            paragraph = paragraph.replace(phrase, replacement)
        
        paragraph = terms_of_law(paragraph)
        paragraph = duties_terms(paragraph)
        paragraph = dupplicated_char_remover(paragraph)
        return paragraph

    def _text_tokenizer(self, paragraph: str) -> str:
        """Tokenize regular text."""
        paragraph = ner_tokenize(paragraph)
        for phrase, replacement in self.special_terms.items():
            paragraph = paragraph.replace(phrase, replacement)
        
        paragraph = preprocess_pyvi(paragraph)
        paragraph = ViTokenizer.tokenize(paragraph)
        paragraph = postprocess_pyvi(paragraph)
        return paragraph

    def _stopword_remover(self, paragraph: str) -> str:
        """Remove stopwords from text."""
        return " ".join([word for word in paragraph.split() if word not in self.stop_words]).strip()

    def preprocess_text(self,
                       paragraph: str,
                       url_remover: bool = True,
                       punctuation_remover: bool = True,
                       line_breaker_remover: bool = True,
                       lowercase_standardizer: bool = False, 
                       white_space_remover: bool = True,
                       text_tokenizer: bool = True,
                       law_text_recognizer: bool = True,
                       stop_word_remover: bool = True) -> str:
        """
        Preprocess a single text paragraph.
        
        Args:
            paragraph: Input text string
            [preprocessing flags...]
            
        Returns:
            Preprocessed text string
        """
        if url_remover:
            paragraph = self._url_remover(paragraph)
        
        if punctuation_remover:
            paragraph = self._punctuation_remover(paragraph)

        if line_breaker_remover:
            paragraph = self._line_breaker_remover(paragraph)

        if lowercase_standardizer:
            paragraph = self._lowercase_standardizer(paragraph)

        if white_space_remover:
            paragraph = self._white_space_remover(paragraph)
            
        if law_text_recognizer:
            paragraph = self._legal_text_tokenizer(paragraph)
            
        if text_tokenizer:
            paragraph = self._text_tokenizer(paragraph)
            
        if stop_word_remover:
            paragraph = self._stopword_remover(paragraph)
        
        return paragraph