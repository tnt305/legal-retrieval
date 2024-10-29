
import string
import re
from tqdm import tqdm
from typing import List
from src.preprocessor.utils import remove_punc, remove_n_items
from src.preprocessor.vocab.stopwords import STOP_WORDS
from src.preprocessor.vocab.legal_dict import LEGAL_DICT
from src.preprocessor.vocab.duties_dict import DUTIES
from src.preprocessor.vocab.special_terms import SPECIAL_TERMS
from src.preprocessor.vocab.acronym import ACRONYMS
from src.preprocessor.base import BaseTextPostPreprocessor


class PostPreprocessing(BaseTextPostPreprocessor):
    def __init__(self,
                 update_vocab: List,
                 legal_term,
                 stop_words,
                 duty_term,
                 special_term):

        self.update_vocab = update_vocab
        self.legal_term = LEGAL_DICT if legal_term is None else legal_term
        self.stop_words = STOP_WORDS if stop_words is None else stop_words 
        self.duties = DUTIES if duty_term is None else duty_term
        self.special_terms = SPECIAL_TERMS if special_term is None else special_term
        
    @staticmethod
    def handle_n_items(text):
        numbers = [1,2,3,4,5,6,7,8,9]
        for word in text.split("_"):
            for item in word:
                if item[0] == 'n':
                    if any(str(i) in item[1] for i in numbers):
                        if item[2] == item[2].upper():
                            rewrite_text = item[2:]
                        else:
                            rewrite_text = item
                    else:
                        if item[1] == item[1].upper():
                            rewrite_text = item[1:]
                        else:
                            rewrite_text = item
        return rewrite_text
    
    @staticmethod
    def handle_rules(text):
        rules = ['Khoản', "Điều", 'Điểm', 'Chương', 'Cấp', 'Hạng', "Mục"]
        words = text.split("_")
        if len(words) == 4 and any(substring in text for substring in rules):
            word1 = words[0] + "_" + words[1]
            word2 = words[-2] + "_" + words[-1]
            return word1, word2
        elif len(words) == 6 and any(substring in text for substring in rules):
            word1 = words[0] + "_" + words[1]
            word2 = words[2] + "_" + words[3]
            word3 = words[-2] + "_" + words[-1]
            return word1, word2, word3
        elif len(words)%2 == 1 and any(substring in text for substring in rules):
            if words[0] in rules:
                try:
                    int_value  = int(words[1])
                    word1 = words[0] + "_" + int_value
                    word2 = "_".join(words[1:])
                except: 
                    word1 = words[0]
                    word2 = "_".join(words[1:])
            return word1, word2 
    
    @staticmethod
    def handle_punc(text):
        if text.startswith("."):
            text = text.replace(".", "")
        elif text.endswith("."):
            text = text.replace(".", "")
        return text
    
    @staticmethod
    def handle_rules_v2(text):
        """
        Rules liên quan tới rewrite lại từ vựng
        """
        pass
    @staticmethod
    def handle_xa0_and_stopwords(text):
        new_text = text.replace('xa0', '')
        words = text.split("_")
        if words[-1].lower() in ['và', 'là', 'được', 'các', 'của', 'để']:
            new_text  = "_".join(words[:-1])
        
        if len(words)>2 and len(words)%2 == 1 and 'trừ' in words[-1]:
            new_text  = "_".join(words[:-1])
        return new_text
    
    @staticmethod
    def handle_uppercase(text):
        if text == text.upper():
            new_text = text.lower()
        else:
            list_text = []
            words = text.split("_")
            for word in words:
                # Kiểm tra từ ký tự thứ hai trở đi xem có phải chữ hoa không
                if any(char.isupper() for char in word[1:]):
                    word = word.lower()  # Nếu có, chuyển toàn bộ từ đó thành chữ thường
                list_text.append(word)  # Thêm từ vào danh sách kết quả
            new_text = "_".join(list_text)
        return new_text
    
    @staticmethod
    def handle_false_negatives(text):
        """
        Với những văn bản mà không nằm trong các term được định nghĩa, có chiều dài >=3
        Sử dụng pyvi để hiệu chỉnh với 3, 5
        Chia đôi nếu là 4
        """
        pass

    @staticmethod
    def handle_acronym(text):
        try:
            text = ACRONYMS[text]
        except:
            pass
        return text

    @staticmethod
    def handle_numerical(text):
        try:
            text = int(text)
        except:
            pass
        return text

    
            
        