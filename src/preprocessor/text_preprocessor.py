import re
import string
import pandas as pd
from tqdm import tqdm
from pyvi import ViTokenizer
from underthesea import ner
from src.preprocessor.vocab.stopwords import STOP_WORDS
from src.preprocessor.vocab.legal_dict import LEGAL_DICT
from src.preprocessor.vocab.duties_dict import DUTIES
from src.preprocessor.legal_processing.legal_terms_tokenize import terms_of_law
from src.preprocessor.legal_processing.duties_tokenize import duties_terms, ner_tokenize

class TextPreprocessing:
    def __init__(self, legal_term: dict =  None, stop_words: dict = None, duty_term: dict = None):
        self.legal_term = LEGAL_DICT if legal_term is None else legal_term
        self.stop_words = STOP_WORDS if stop_words is None else stop_words
        self.duties = DUTIES if duty_term is None else duty_term
    
    def preprocess(self,
                   docs,
                   punctuation_remover: bool = True,
                   line_breaker_remover: bool= True,
                   lowercase_standardizer: bool = False,
                   white_space_remover: bool = True,
                   text_tokenizer: bool = True,
                   law_text_recognizer: bool = True,
                   stop_word_remover: bool = True):
        # Nếu texts là Series, dùng .apply() để xử lý từng phần tử
        if isinstance(docs, pd.Series):
            tqdm.pandas(desc="Pre-processing")  # Sử dụng tqdm để hiển thị tiến trình
            return docs.progress_apply(
                lambda t: self.preprocess_text(
                    str(t), punctuation_remover, line_breaker_remover, 
                    lowercase_standardizer, white_space_remover, text_tokenizer,
                    law_text_recognizer, stop_word_remover
                )
            )
        else:
            # Xử lý danh sách thông thường như cũ
            results = self.preprocess_text(
                docs, punctuation_remover, line_breaker_remover, 
                lowercase_standardizer, white_space_remover, text_tokenizer,
                law_text_recognizer, stop_word_remover)
            return results  # Moved return statement here   
    
    def preprocess_text(self,
                   paragraph: str,
                   punctuation_remover: bool = True,
                   line_breaker_remover: bool= True,
                   lowercase_standardizer: bool = False,
                   white_space_remover: bool = True,
                   text_tokenizer: bool = True,
                   law_text_recognizer: bool = True,
                   stop_word_remover: bool = True,
                   ):
        
        if punctuation_remover:
            paragraph =  self._punctuation_remover(paragraph)

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
    
    def _punctuation_remover(self, paragraph: str):
        
        paragraph = re.sub(r"\((\d+)\)", r"\1", paragraph)
        # loại bỏ các mục có dạng a) b) ...
        paragraph = re.sub(r"\w+\)", " ", paragraph)
        # loại bỏ các mục có dạng 1. 2. 1) 2)
        paragraph = re.sub(r"\b\d+[\.\)]", " ", paragraph)
        for punc in string.punctuation:
            # Bỏ qua dấu / trường hợp là nghị định
            if punc == "/" or punc == ".": 
                continue
            paragraph = paragraph.replace(punc, " ")
        # loại bỏ khoảng trắng thừa
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        return paragraph
    
    def _line_breaker_remover(self, paragraph: str):
        para = re.sub(r"\n+" , ". ",paragraph)
        para = re.sub(r"\.\.\.", " ", paragraph)
        return para.replace("  ", " ")
    
    def _lowercase_standardizer(self,paragraph: str):
        return paragraph.lower()
    
    def _white_space_remover(self, paragraph: str):
        para = paragraph.replace("  ", " ")
        para = re.sub(r"\s{2,}", " ", para).strip()
        return para
    
    def _legal_text_tokenizer(self, paragraph: str):
        '''
        Các nội dung pháp lý để tiền xử lý bao gồm
        - Các từ ngữ chuyên dụng cần được tokenize 
        - Điều khoản
        '''
        
        for phrase, replacement in tqdm(self.legal_term.items(), desc = 'Xử lý các nội dung pháp lý'):
            para = paragraph.replace(phrase, replacement)
        
        para = terms_of_law(para)
        para = duties_terms(para)
            
        return para
            
    def _text_tokenizer(self, paragraph: str):
        '''
        normal tokenizer
        '''
        paragraph = ner_tokenize(paragraph)
        paragraph =  ViTokenizer.tokenize(paragraph)
        return paragraph
    
    def _stopword_remover(self, paragraph: str):
        return " ".join([vnm for vnm in paragraph.split() if vnm not in self.stop_words]).strip()
    
    