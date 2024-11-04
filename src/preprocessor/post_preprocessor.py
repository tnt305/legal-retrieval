import re
from tqdm import tqdm
import pandas as pd
from typing import List, Set, Dict, Union, Tuple, Optional
from src.preprocessor.vocab.stopwords import STOP_WORDS
from src.preprocessor.vocab.legal_dict import LEGAL_DICT
from src.preprocessor.vocab.duties_dict import DUTIES
from src.preprocessor.vocab.special_terms import SPECIAL_TERMS
from src.preprocessor.vocab.acronym import ACRONYMS
from src.preprocessor.vocab.province import PROVINCES
from src.preprocessor.vocab.roman_numerals_dict import ROMAN_DICT
from src.preprocessor.base.base_model import BaseTextPostPreprocessor


class PostPreprocessing(BaseTextPostPreprocessor):
    """Class for post-processing text with various rules and vocabulary handling."""
    
    def __init__(
        self,
        legal_term: Optional[Dict[str, str]] = None,
        stop_words: Optional[Set[str]] = None,
        duty_term: Optional[Dict[str, str]] = None,
        special_term: Optional[Set[str]] = None
    ) -> None:
        """Initialize PostPreprocessing object with vocabularies and terms.
        
        Args:
            legal_term: Dictionary of legal terms and definitions
            stop_words: Set of stop words to be filtered
            duty_term: Dictionary of duties and their definitions
            special_term: Set of special terms
        """
        super().__init__()
        
        self.legal_term = legal_term or LEGAL_DICT
        self.stop_words = stop_words or STOP_WORDS
        self.duties = duty_term or DUTIES
        self.special_terms = special_term or SPECIAL_TERMS
        
        # Create token sets for faster lookup
        self.legal_tokens = set(self.legal_term.values())
        self.stopwords_tokens = {i for i in self.stop_words if i}
        self.duties_tokens = set(self.duties.values())
        self.special_tokens = set(self.special_terms)

    @staticmethod
    def handle_n_items(text: str) -> str:
        """Handle items starting with 'n' followed by numbers.
    
        Args:
            text: Input text to process
            
        Returns:
            Processed text with n-items handled
        """
        numbers = map(str, range(1, 10))
        rewrite_text = []
        words = text.split("_")
        for word in words:
            if word.startswith('n'):
                if len(word) >= 2:
                    if word[1] == word[1].upper():
                        if any(num in word[1] for num in numbers): # word[2]
                            rewrite_text.append(word[2:])
                        else:
                            rewrite_text.append(word[1:])  
                    else:
                        rewrite_text.append(word)
                else:
                    rewrite_text.append(word)
            else:
                rewrite_text.append(word)
        
        rewrite_text = "_".join(rewrite_text)

        return rewrite_text

    @staticmethod
    def handle_rules(text: str):
        """Handle rules based text processing with specific formatting.
        
        Args:
            text: Input text containing rules
            
        Returns:
            Tuple of processed text parts or original text if no rules apply
        """
        numbers = [str(i) for i in range(1, 1000)]
        rules = ['Khoản', 'Điều', 'Điểm', 'Chương', 'Cấp', 'Hạng', 'Mục']
        words = text.split("_")
        
        if len(words) == 4 and any(rule in text for rule in rules):
            return f"{words[0]}_{words[1]}", f"{words[-2]}_{words[-1]}"
        
        elif len(words) == 6 and any(rule in text for rule in rules):
            return (f"{words[0]}_{words[1]}", 
                   f"{words[2]}_{words[3]}", 
                   f"{words[-2]}_{words[-1]}")
        
        elif len(words) > 2 and words[-2] in rules and words[-1] in numbers:
            return (f"{words[-2]}_{words[-1]}",
                f"{words[:-2]}",)
        else:
            return text
        

    @staticmethod
    def handle_punc(text: str) -> str:
        """Remove trailing periods from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with trailing periods removed
        """
        text = text.strip(".")
        text =  text.replace(".", "")
        return text

    def handle_rules_v2(self, text: str) -> Union[Tuple[str, str], str]:
        """Handle additional vocabulary-based rules.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of matched term and remaining text, or single term if no match
        """
        for term in self.duties_tokens:
            if term in text:
                sub_text = text.split(term)[-1].strip("_")
                if (sub_text in self.duties_tokens or 
                    sub_text in self.special_tokens or 
                    sub_text in self.legal_tokens):
                    return term, sub_text
                return term
        return text

    @staticmethod
    def handle_xa0_and_stopwords(text: str) -> str:
        """Remove 'xa0' characters and handle stopwords.
        
        Args:
            text: Input text
            
        Returns:
            Processed text with xa0 removed and stopwords handled
        """
        new_text = text.replace('xa0', '')
        new_text = text.replace('xAA0', '')
        words = text.split("_")
        
        if words[-1].lower() in list(STOP_WORDS):
            new_text = "_".join(words[:-1])
        
        if len(words) > 2 and len(words) % 2 == 1 and 'trừ' in words[-1]:
            new_text = "_".join(words[:-1])
            
        return new_text

    @staticmethod
    def handle_uppercase(text: str) -> str:
        """Normalize text case based on specific rules.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized case
        """
        if isinstance(text, int):
            return ""
        elif isinstance(text, float):
            return ""
        elif text.isupper():
            text = text.lower()
            words = text.split("_")
            new_words = [
                word.lower() if any(char.isupper() for char in word[1:])
                else word for word in words
            ]
            return "_".join(new_words)
        else:
            return text

    @staticmethod
    def handle_acronym(text: str) -> str:
        """Replace acronyms with their full forms if available.
        
        Args:
            text: Input text
            
        Returns:
            Text with acronyms replaced
        """
        return ACRONYMS.get(text, text)

    @staticmethod
    def handle_numerical(text: str) -> Union[int, str]:
        """Convert text to integer if possible.
        
        Args:
            text: Input text
            
        Returns:
            Integer if conversion successful, original text otherwise
        """
        try:
            return int(text)
        except ValueError:
            return text

    @staticmethod
    def handle_places(text: str) -> str:
        pass 
    
    @staticmethod
    def normalize_text(text: str) -> str:
        norm_words = []
        if any(province in text for province in list(PROVINCES.values())) or \
            any(province in text for province in list(DUTIES)) or \
                any(province in text for province in list(ROMAN_DICT)) or \
                    any(province in text for province in ['Khoản', 'Điều', 'Điểm', 'Chương', 'Cấp', 'Hạng', 'Mục']):
            return text
        else:
            words = text.split('_')
            # Chuyển từng từ thành chữ thường
            for word in words:
                if word not in set(PROVINCES.values()) and word not in DUTIES and word not in ROMAN_DICT:
                    word = word.lower()
                else:
                    word = word
                norm_words.append(word)
            
            # Ghép lại thành chuỗi với dấu gạch dưới
            normalized_text = '_'.join(norm_words)
            return normalized_text
   
    def post_preprocess_text(
        self, 
        text: str
    ) -> Union[str, Tuple[str, str], Tuple[str, str, str], int]:
        """Apply all post-processing steps to input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text in various formats depending on applied rules
        """
        final_text = []
        text = self.handle_punc(text)
        text = self.handle_n_items(text)
        text = self.handle_xa0_and_stopwords(text)
        text = self.handle_numerical(text)
        text = self.handle_uppercase(text)
        text = self.handle_acronym(text)
        texts = self.handle_rules(text)
        if isinstance(texts, tuple):
            for i in range(len(texts)):
                txt = self.handle_rules_v2(texts[i])
                txt = self.normalize_text(txt)  
                final_text.append(txt)
        else:
            texts = self.normalize_text(texts)
            final_text.append(texts)
        return " ".join(final_text)
    
    def post_preprocess(self, 
                        docs: Union[List[str], pd.Series, str]):
        """
        DO NOT USE THIS
        """
        if isinstance(docs, str):
            return self.post_preprocess_text(docs)
        elif isinstance(docs, list):
            return [self.post_preprocess_text(doc) for doc in tqdm(docs)]
        else:  # pandas Series
            tqdm.pandas()
            return docs.progress_apply(self.post_preprocess_text)
    
    def post_preprocess_v1(self, docs: List):
        v1_processed = []
        for i in range(len(docs)):
            v1_processed.extend(self.post_preprocess_text(docs[i]))
        return list(set(v1_processed))
        
    def post_preprocess_v2(self, docs: List):
        v2_processed = set()
        v1_processed = set(self.post_preprocess_v1(docs))

        for item in v1_processed:
            item_lower = item.lower()
            
            # Kiểm tra duties
            matching_duties = {duty for duty in DUTIES.values() if duty.lower() in item_lower}
            if matching_duties:
                v2_processed.update(matching_duties)
                continue
            
            # Xử lý item có 4 phần
            parts = item.split("_")
            if len(parts) == 4:
                v2_processed.update([f"{parts[0]}_{parts[1]}", f"{parts[-2]}_{parts[-1]}"])
            else:
                v2_processed.add(ACRONYMS.get(item, item))

        # Áp dụng các regex transformation, loại bỏ stop words và datetime
        v2_processed = {
            "" if (
                item.lower() in STOP_WORDS or 
                re.match(r'\d{1,2}/\d{1,2}/\d{4}', item)
            ) else re.sub(r'\b\w_\w\b', '', item).replace('xad', '')
            for item in v2_processed
        }

        # Loại bỏ chuỗi rỗng khỏi set
        v2_processed.discard("")
        
        return v2_processed