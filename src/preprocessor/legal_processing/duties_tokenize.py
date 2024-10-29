import re
from underthesea import ner
from src.preprocessor.vocab.duties_dict import DUTIES

def duties_terms(text):
    for item in DUTIES:
        # Tạo regex để tìm kiếm từ khóa, không phân biệt hoa thường
        pattern = re.compile(re.escape(item), re.IGNORECASE)
        
        # Thay thế từ khóa bằng dạng nối với dấu gạch dưới
        # Xóa dấu gạch ngang (-) và giữ nguyên chữ hoa ban đầu
        text = pattern.sub(item.replace(" ", "_").replace("-", ""), text)
    
    return text

def ner_tokenize(text):
    ner_results = ner(text)
    for item in ner_results:
        # Kiểm tra nếu thực thể là tên riêng (B-NP với Np hoặc B-PER)
        if (item[2] == 'B-NP' and item[1] == 'Np') or item[-1] == 'B-PER':
            entity = item[0]  # Lấy tên thực thể từ kết quả NER
            # Tạo regex pattern từ tên thực thể (không phân biệt hoa thường)
            pattern = re.compile(re.escape(entity), re.IGNORECASE)
            # Thay thế thực thể bằng dạng nối dấu gạch dưới
            text = pattern.sub(entity.replace(" ", "_").replace("-", ""), text)
    return text