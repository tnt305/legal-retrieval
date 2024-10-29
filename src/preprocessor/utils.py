import re
import string

    
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

def postprocess_pyvi(text):
    # Khôi phục lại các cụm từ được bao bọc bởi {}
    text = re.sub(r'\{\s*(\S+(?:\s*/\s*\S+)*)\s*\}', lambda m: m.group(1).replace(' ', ''), text)
    return text

def remove_punc(text):
    words = text.split(" ")
    rewrite_text = []
    for word in words:
        if word in string.punctuation:
            continue
        else:
            rewrite_text.append(word)
    
    return " ".join(rewrite_text)