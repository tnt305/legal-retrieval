import re
import string
from pyvi import ViTokenizer
from src.preprocessor.vocab.stopwords import STOP_WORDS
from src.preprocessor.vocab.legal_dict import LEGAL_DICT
from src.preprocessor.vocab.duties_dict import DUTIES
from src.preprocessor.vocab.special_terms import SPECIAL_TERMS
from src.preprocessor.vocab.numeral_currency import CURRENCY


STOP_WORDS = list(STOP_WORDS)
LEGAL_DICT = list(LEGAL_DICT.values())
DUTIES = list(DUTIES.values())
SPECIAL_TERMS = list(SPECIAL_TERMS.values())

def remove_isolated_numbers(text):
    # Tách văn bản thành các từ
    words = text.split()
    filtered_words = []

    for word in words:
        try:
            # Nếu từ có thể chuyển thành số nguyên, bỏ qua từ đó
            if int(word.replace(',', '').replace('.', '')):
                continue
        except ValueError:
            # Nếu từ không thể chuyển thành số nguyên, kiểm tra thêm các điều kiện khác
            # Pattern để nhận diện số có dấu / ở trước hoặc sau
            has_slash_number = re.findall(r'(?<=/)\d+|\d+(?=/)', word)

            # Pattern để nhận diện từ có chứa ký tự khác
            contains_mix = re.search(r'[^\d.,\s]', word)
            
            # Giữ lại từ nếu có ký tự khác hoặc chứa số có dấu /
            if contains_mix or has_slash_number:
                filtered_words.append(word)

    # Kết hợp lại thành văn bản
    return ' '.join(filtered_words)

def remove_n_items(text):
    # Loại bỏ các từ có dạng 'n', 'n1', 'n2', v.v.
    return re.sub(r'\bn\d*\b', '', text)

def reduce_multiple_punctuation(text):
    # Thay thế nhiều dấu câu liên tiếp (bao gồm cả dấu cách) bằng một dấu chấm
    return re.sub(r'(\s*[^\w\s]+\s*)+', '.', text)

def dupplicated_char_remover(text):
    # Thay thế các từ lặp lại và nối với ký tự a-z đứng riêng lẻ sau đó bằng dấu _
    return re.sub(r'\b(\w+)_\1\b\s([a-z])\b', r'\1_\2', text)

def preprocess_pyvi(text):
    # Bao bọc các cụm từ có dạng ký_tự/ký_tự/... bằng {}
    text = re.sub(r'(\S+/\S+(/\S+)*)', r'{\1}', text)
    return text

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

def postprocess_pyvi(text):
    # Khôi phục lại các cụm từ được bao bọc bởi {}
    text = re.sub(r'\{\s*(\S+(?:\s*/\s*\S+)*)\s*\}', lambda m: m.group(1).replace(' ', ''), text)
    
    text_test = []
    for i in text.split(" "):
        i = handle_rules(i)
        if isinstance(i, str):
            text_test.append(i)
        elif isinstance(i, tuple):
            text_test.extend([j for j in i])
            
    final_text = []
    for i in text_test:
        if i not in STOP_WORDS and i not in LEGAL_DICT and i not in DUTIES and i not in SPECIAL_TERMS:
            if len(i.split("_")) == 3:
                before, after = i.split("_")[0] ,"_".join(i.split("_")[1:])
                final_text.append(before)
                final_text.append(after)
            else:
                final_text.append(i)
        else:
            final_text.append(i)

    final_text = " ".join(final_text)
    return final_text

def remove_punc(text):
    words = text.split(" ")
    rewrite_text = []
    for word in words:
        if word in string.punctuation:
            continue
        else:
            rewrite_text.append(word)
    
    return " ".join(rewrite_text)

def remove_punc_v2(text):
    for punc in string.punctuation:
        text = text.replace(punc, ' ')
        text = text.replace('  ', ' ')
    return text

def is_numerical(text):
    try: 
        i = int(text)
        return True
    except:
        i = i
        return False