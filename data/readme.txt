corpus.csv 
- text: Một đoạn văn bản pháp luật bất kỳ (dạng string)
- cid: Id của đoạn văn bản đó trong corpus (dạng int)

train.csv:
- question: Dạng văn bản của câu hỏi (dạng string)
- qid: Mã id của câu hỏi (viết tắt của question_id, dạng string)
- context: Các đoạn văn bản luật pháp liên quan (dạng list)
- cid: Mã id của các đoạn văn bản pháp luật trong corpus có liên quan tới câu hỏi (viết tắt của context_id, dạng list)

public_test.csv:
- question: Dạng văn bản của câu hỏi (dạng string)
- qid: Mã id của câu hỏi (viết tắt của question_id, dạng string)