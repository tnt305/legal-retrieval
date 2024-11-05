def qwen_completion_to_prompt(system_message, user_message):
    return f"<|im_start|>{system_message}\n<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"


DEFAULT_SYSTEM_PROMPT = """Bạn là người trợ lý hữu ích, trung thực và tài giỏi về lĩnh vực pháp luật dân sự Việt Nam. 
Bạn hãy giúp tôi phân loại câu hỏi mà người dùng đưa vào để nhân diện chúng thuộc về loại câu hỏi nào trong 7 loại câu hỏi về pháp luật dưới đây:
-   Câu hỏi pháp lý
-   Câu hỏi về giải thích luật 
-   Câu hỏi về áp dụng luật
-   Câu hỏi về quyền và nghĩa vụ 
-   Câu hỏi về tiền lệ pháp lý 
-   Câu hỏi về chứng cứ
-   Câu hỏi tình huống" 

Nếu câu hỏi người dùng đưa vào khó để phân loại, hãy gán nhãn là Unknown.
Hãy đưa câu trả lời dưới dạng json như ví dụ sau với mỗi user prompt
{"label": Câu hỏi tình huống, "question": Câu hỏi được đưa vào để kiểm tra}

"""

# user_message = None

# SUPPORTED_LLM_MODELS = {
#     "Vietnamese": {
#         "qwen2.5-0.5b-instruct": {
#             "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
#             "remote_code": False,
#             "system_prompt": DEFAULT_SYSTEM_PROMPT,
#             "user_prompt": f"{user_message}",
#             "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
#             "completion_to_prompt": qwen_completion_to_prompt,
#         },
#     }
# }