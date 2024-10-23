import string
import re
class TextPreprocessing:
    def __init__(self):
        pass

    def preprocess(self,
                   paragraph: str,
                   punctuation_remover: bool = True,
                   line_breaker_remover: bool= True,
                   lowercase_standardizer: bool = False,
                   white_space_remover: bool = True,
                   text_tokenizer: bool = True,
                   stop_word_remover: bool = True,
                   law_text_recognizer: bool = True,
                   ):
        if punctuation_remover:
            return self._punctuation_remover(paragraph)
        if line_breaker_remover:
            return self._line_breaker_remover(paragraph)
        if lowercase_standardizer:
            return self._lowercase_standardizer(paragraph)
        if white_space_remover:
            pass
        if text_tokenizer:
            pass
        if stop_word_remover:
            pass
        if law_text_recognizer:
            pass
    def _punctuation_remover(self, paragraph):
        for punc in string.punctuation:
            para = paragraph.replace(punc, "")
            para = re.sub(r"\s+", "", para)
        return para
    
    def _line_breaker_remover(self, paragraph):
        para = re.sub(r"\n+" , "",paragraph)
        para = re.sub(r"\.\.\.", "", paragraph)
        return para.replace("  ", " ")
    
    def _lowercase_standardizer(self,paragraph):
        return paragraph.lower()
    
    def _white_space_remover(self, paragraph):
        para = paragraph.replace("  ", " ")
        para = re.sub(r"\s{2,}", " ", para).strip()
        return para
    
    def _text_tokenizer(self, paragraph):
        return tokenize(paragraph)
    
    def law_text_recognizer(self, paragraph):
        pass
    
    def _stopword_remover(self,paragraph):
        pass
    
    