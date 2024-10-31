from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim as consine

def evaluate(queries: dict, corpus: dict, relevant_docs: dict, matryoshka_dimensions: list = [768, 512, 256]):
    matryoshka_dimensions = matryoshka_dimensions # Important: large to small
    matryoshka_evaluators = []
    # Iterate over the different dimensions
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
    
    return SequentialEvaluator(matryoshka_evaluators)