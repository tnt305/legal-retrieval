
import torch
import gc
from datasets import Dataset
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

from setup.settings import setup_training_args, setup_embedding_model
from setup.memory_efficiency import MemoryEfficientTrainer
from src.preprocessor.utils.dataset_level import read_pickle, prepare_training_dataset, read_json
from src.preprocessor.utils.evaluate import evaluate

def setup_training(config_path, model, train_dataset, train_loss, evaluator):
    args = setup_training_args(config_path)
    
    model.to('cuda')
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    trainer = MemoryEfficientTrainer(
        model=model,
        args=args,
        train_dataset= train_dataset,
        loss= train_loss,
        evaluator = evaluator,
    )
    
    return trainer

if __name__ == '__main__':
    
    corpus = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/corpus.pkl')
    queries = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/queries.pkl')
    relevant_docs = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/relevant_docs.pkl')
    new_tokens = read_json('./src/preprocessor/vocab/data/update_vocab_v1.json')
    
    train_dataset = prepare_training_dataset(queries, corpus, relevant_docs)
    train_dataset = Dataset.from_dict(train_dataset[:10000])
    matryoshka_dimensions= [768, 512, 256]
    evaluator = evaluate(corpus, queries, relevant_docs, matryoshka_dimensions = matryoshka_dimensions)
    model, tokenizer = setup_embedding_model('hiieu/halong_embedding', new_tokens = new_tokens)
    
    train_loss = MatryoshkaLoss(model = model, 
                                loss = MultipleNegativesRankingLoss(model), 
                                matryoshka_dims = matryoshka_dimensions)
    torch.cuda.empty_cache()
    gc.collect()
    
    trainer = setup_training(config_path = './setup/config.yaml', model = model)
    try:
        trainer.train()
    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        gc.collect()