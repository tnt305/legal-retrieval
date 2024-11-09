import os
import torch
import gc
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from src.preprocessor.utils.dataset_level import read_pickle, prepare_training_dataset_with_triplet, read_json
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim as consine
from sentence_transformers.losses import MatryoshkaLoss, TripletLoss
from huggingface_hub import login

def setup_embedding_model(model_name, new_tokens=None):
    """
    Set up a sentence transformer model with proper tokenizer handling and pooling
    
    Args:
        model_name (str): HuggingFace model name/path
        new_tokens (list): Optional list of new tokens to add to vocabulary
    
    Returns:
        SentenceTransformer: Properly configured sentence transformer model
    """
    # Set up word embedding model
    word_embedding_model = models.Transformer(model_name)
    tokenizer = word_embedding_model.tokenizer
    
    # Add new tokens if provided
    if new_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens, special_tokens=False)
        print(f"Added {num_added_tokens} new tokens to the vocabulary")
        # Resize model embeddings to account for new tokens
        word_embedding_model.auto_model.resize_token_embeddings(len(tokenizer))
    
    # Create pooling model
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    
    # Create the full SentenceTransformer model
    sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    return sentence_model

@contextmanager
def track_memory():
    torch.cuda.reset_peak_memory_stats()
    yield
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Custom trainer với memory management
class MemoryEfficientTrainer(SentenceTransformerTrainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        
        # Dọn memory Python và CUDA cache sau mỗi step
        gc.collect()
        torch.cuda.empty_cache()
        
        return loss
        
    def on_epoch_end(self):
        # Dọn memory sau mỗi epoch
        gc.collect()
        torch.cuda.empty_cache()
        super().on_epoch_end()

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    corpus = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/corpus.pkl')
    corpus = {i: j.replace("\xa0", "") for i, j in corpus.items()}
    corpus = {i: j for i, j in corpus.items() if len(j.split(" ")) <= 384}
    
    queries = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/queries.pkl')
    relevant_docs = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/relevant_docs.pkl')


    ######## Train dataset #######
    selected_queries = dict(sorted(queries.items(), key=lambda item: item[1])[:10000])
    # Lọc corpus và relevant_docs dựa trên selected_queries
    selected_corpus = {i: corpus[i] for i in selected_queries.keys() if i in corpus}
    selected_relevant_docs = {i: relevant_docs[i] for i in selected_queries.keys() if i in relevant_docs}
    ####### Val dataset #######
    # u_selected_queries = dict(sorted(queries.items(), key=lambda item: item[1])[:5000])
    # # Lọc corpus và relevant_docs dựa trên selected_queries
    # u_selected_corpus = {i: corpus[i] for i in u_selected_queries.keys() if i in corpus}
    # u_selected_relevant_docs = {i: relevant_docs[i] for i in u_selected_queries.keys() if i in relevant_docs}

    # # Chuẩn bị dataset cho training
    # val_dataset = prepare_training_dataset_with_triplet(u_selected_queries, u_selected_corpus, u_selected_relevant_docs)
    # Chuẩn bị dataset cho training
    train_dataset = prepare_training_dataset_with_triplet(selected_queries, selected_corpus, selected_relevant_docs)


    new_tokens = read_json('./src/preprocessor/vocab/data/update_vocab_v2.json')
    model = setup_embedding_model('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', new_tokens= None)

    del corpus, queries, relevant_docs

    matryoshka_dimensions = [384, 256, 128] # Important: large to small
    matryoshka_evaluators = []
    # Iterate over the different dimensions
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries= selected_queries,
            corpus=selected_corpus,
            relevant_docs=selected_relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": consine},
        )
        matryoshka_evaluators.append(ir_evaluator)
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    
    inner_train_loss = TripletLoss(model)
    train_loss = MatryoshkaLoss(
        model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
    )


    args = SentenceTransformerTrainingArguments(
        output_dir="./src/retrieval/dense/embeddings/models/legal_infloat_v2",
        num_train_epochs = 50,
        per_device_train_batch_size= 4,  # Giảm batch size             
        gradient_accumulation_steps= 4,  # Tăng gradient accumulation            
        per_device_eval_batch_size= 4,
        gradient_checkpointing=True,
        warmup_ratio = 0.1,
        learning_rate= 3e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        save_steps= 200,
        logging_steps = 100,
        save_total_limit = 5,
        load_best_model_at_end=True,
        max_grad_norm = 0.5,
        metric_for_best_model="eval_dim_256_cosine_mrr@10",
        # resume_from_checkpoint = "./legal_finetuning_v2/checkpoint-128",
        ddp_find_unused_parameters=False,
        dataloader_num_workers = 40
    )

    # Khởi tạo trainer với custom class
    trainer = MemoryEfficientTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    print("Empty cache ....")
    del selected_queries, selected_corpus, selected_relevant_docs
    # del u_selected_queries, u_selected_corpus, u_selected_relevant_docs
    del model 
    torch.cuda.empty_cache()

    with track_memory():
        trainer.train()
    
    login(token="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", add_to_git_credential=True)
    trainer.model.push_to_hub("infloat_small_legal_triplet")
