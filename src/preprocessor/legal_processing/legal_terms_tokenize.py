import re
from tqdm import tqdm

def is_valid_roman(numeral):
    valid_roman = re.fullmatch(r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', numeral)
    return valid_roman is not None

def terms_of_law(text):
    # Định nghĩa các quy tắc cho 'Chương', 'Điều', 'Khoản', và 'điểm'
    terms = [
        (r'[Cc]ấp\s+([IVXLCDM]+(?:,\s*[IVXLCDM]+)*(?:\s+và\s+[IVXLCDM]+)?)', 'cấp'),
        (r'[Cc]hương\s+([IVXLCDM]+(?:,\s*[IVXLCDM]+)*(?:\s+và\s+[IVXLCDM]+)?)', 'Chương'),
        (r'[Đđ]iều\s+\d+(?:,\s*\d+)*(?:\s+và\s+\d+)?', 'Điều'),
        (r'[Kk]hoản\s+\d+(?:,\s*\d+)*(?:\s+và\s+\d+)?', 'Khoản'),
        (r'[Đđ]iểm\s+([a-zđ](?:,\s*[a-zđ])*)(?:\s+và\s+([a-zđ]))?', 'điểm')
    ]

    for pattern, term in tqdm(terms, desc='Đang xử lý các quy tắc pháp luật'):
        # Tìm các cụm từ phù hợp với quy tắc
        matches = re.finditer(pattern, text)

        # Danh sách chứa các phần của chuỗi kết quả
        expanded_text_parts = []
        last_end = 0

        for match in matches:
            start, end = match.span()

            # Xử lý các cụm 'Chương' với số La Mã sau từ 'Chương'
            if term == 'Chương' or term == 'cấp':
                # Lấy các số La Mã từ cụm phù hợp
                roman_numbers = re.findall(r'[IVXLCDM]+', match.group(1))
                # Lọc chỉ giữ lại các số La Mã hợp lệ
                valid_romans = [num for num in roman_numbers if is_valid_roman(num)]
                # Tạo danh sách các chương đã mở rộng
                expanded_chapters = [f"{term}_{num}" for num in valid_romans]
                expanded_text = ', '.join(expanded_chapters)

            # Xử lý các cụm 'Điều' và 'Khoản' với số Ả Rập
            elif term in ['Điều', 'Khoản']:
                # Lấy các số từ cụm phù hợp
                numbers = re.findall(r'\d+', match.group(0))
                # Tạo danh sách các cụm đã mở rộng
                expanded_terms = [f"{term}_{num}" for num in numbers]
                expanded_text = ', '.join(expanded_terms)

            # Xử lý các cụm 'điểm' với chữ cái và chữ 'đ'
            elif term == 'điểm':
                letters_text = match.group(1)
                and_letter = match.group(2)

                # Tách các chữ cái riêng lẻ từ cụm tìm được, bao gồm cả 'đ'
                letters = re.findall(r'[a-zđ]', letters_text)

                # Thêm chữ cái sau 'và' nếu có
                if and_letter:
                    letters.append(and_letter)
                
                # Tạo danh sách các điểm đã mở rộng
                expanded_points = [f"điểm_{letter}" for letter in letters]
                
                # Nối lại chuỗi các điểm đã mở rộng với từ 'và' cho chữ cái cuối cùng
                if len(expanded_points) > 1:
                    expanded_text = ', '.join(expanded_points[:-1]) + ' và ' + expanded_points[-1]
                else:
                    expanded_text = expanded_points[0]

            # Thêm phần chưa thay thế vào kết quả
            expanded_text_parts.append(text[last_end:start])
            
            # Thêm phần đã mở rộng vào kết quả
            expanded_text_parts.append(expanded_text)
            
            # Cập nhật vị trí cuối cùng
            last_end = end

        # Thêm phần còn lại của chuỗi
        expanded_text_parts.append(text[last_end:])
        text = ''.join(expanded_text_parts)

    return text