def qwen_completion_to_prompt(system_message, user_message):
    return f"""<|im_start|>{system_message} \n<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"""