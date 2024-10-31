
import wandb
import torch
from src.preprocessor.utils.dataset_level import read_pickle, prepare_training_dataset, read_json
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from huggingface_hub import login
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim as consine


corpus = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/corpus.pkl')
queries = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/queries.pkl')
relevant_docs = read_pickle('/home/thiendc/projects/legal_retrieval/data/processed/relevant_docs.pkl')
train_dataset = prepare_training_dataset(queries, corpus, relevant_docs)



# 2. Create the base model
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
    
    return sentence_model, tokenizer

# Sử dụng hàm:
# 1. Load new tokens
new_tokens = read_json('./src/preprocessor/vocab/data/update_vocab_v1.json')

# 2. Setup model với vocab mới
model, tokenizer = setup_embedding_model('hiieu/halong_embedding', new_tokens=new_tokens)


matryoshka_evaluators = []
matryoshka_dimensions = [768,512, 256] 
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


del corpus
del queries
del relevant_docs

 # Important: large to small
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)

args = SentenceTransformerTrainingArguments(
    output_dir="legal_finetune",
    num_train_epochs= 20,
    per_device_train_batch_size= 8,             
    gradient_accumulation_steps= 32,            
    per_device_eval_batch_size= 8,
    gradient_checkpointing=True,
    warmup_ratio=0.05,
    learning_rate= 1e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    save_steps= 200,
    logging_steps= 10,
    save_total_limit=5,
    load_best_model_at_end=True,
    max_grad_norm= 1.0,
    metric_for_best_model="eval_dim_786_cosine_ndcg@10",
)


trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset= train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)

# Bắt đầu train
trainer.train()
trainer.save_model()


login(token="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", add_to_git_credential=True)
trainer.model.push_to_hub("legal_embeddings")

