from tqdm import tqdm
from sentence_transformers.util import cos_sim as consine
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)

def evaluate(queries: dict, corpus: dict, relevant_docs: dict):
    matryoshka_dimensions = [768, 512, 256, 128, 64] # Important: large to small
    matryoshka_evaluators = []

    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": consine},
        )
        matryoshka_evaluators.append(ir_evaluator)

    # Create a sequential evaluator
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    return evaluator

def prepare_training_dataset(queries: dict, corpus: dict, relevant_docs: dict):
    anchors = []
    positives = []
    
    # Sử dụng tqdm để theo dõi tiến trình của vòng lặp
    for query_id, docs in tqdm(relevant_docs.items(), desc='Processing queries'):
        for doc_id in docs:
            try:
                # Thử truy cập cả query và document
                anchor = queries[str(query_id)]
                positive = corpus[str(doc_id)]

                # Nếu không gặp lỗi, append vào danh sách
                anchors.append(anchor)
                positives.append(positive)

            except KeyError as e:
                # In ra thông báo lỗi và tiếp tục
                print(f"Lỗi KeyError: {e} - Bỏ qua query_id: {query_id}, doc_id: {doc_id}")
                continue

    df = {
        "anchor": anchors,
        "positive": positives
    }

    return Dataset.from_dict(df)