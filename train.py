import torch
import os
from transformers import AutoTokenizer, AutoModel
from torch.nn.parallel import DistributedDataParallel as DDP
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from datasets import Dataset
from sentence_transformers.util import cos_sim as consine
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from src.preprocessor.utils.sent_level import remove_bullet_dot, remove_n_items_v2
from src.preprocessor.utils.dataset_level import prepare_training_dataset, download_and_setup_model, read_json

import pandas as pd
import numpy as np
from tqdm import tqdm


import pickle
with open('./data/processed/queries.pkl', 'rb') as f:
    queries = pickle.load(f)
with open('./data/processed/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
with open('./data/processed/relevant_docs.pkl', 'rb') as f:
    relevant_docs = pickle.load(f)

pairs = prepare_training_dataset(queries, corpus, relevant_docs)


custom_save_path = download_and_setup_model("hiieu/halong_embedding", "./embeddings/legal_roberta")
new_tokens = read_json('./data/update_vocab.json')
local_path = "./embeddings/legal_roberta"

# 2. Load existing model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModel.from_pretrained(local_path)
num_added_tokens = tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# model = SentenceTransformer("./embeddings/legal_roberta")
matryoshka_dimensions = [512, 256, 128, 64] # Important: large to small
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

# Create a sequential evaluator
evaluator = SequentialEvaluator(matryoshka_evaluators)


matryoshka_dimensions = [512, 256, 128, 64]  # Important: large to small
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)

import wandb
wandb.login(key="02ba155e26496a78f062f683274330566fefe94c")
wandb.init(
    project="sentence_sim",  # Đặt tên project phù hợp
    name="experiment-v1",
    config={
        # Training hyperparameters
        "model_name": "base_model_name",  # tên model base bạn dùng
        "learning_rate": 2e-5,
        "epochs": 50,
        "per_device_batch_size": 4,
        "effective_batch_size": 4 * 8 * 4,  # batch_size * gradient_accum * num_gpus
        "warmup_ratio": 0.1,
        "optimizer": "adamw_torch_fused",
        
        # Model architecture
        "embedding_dim": 768,  # dựa trên metric của bạn
        
        # Dataset info
        "train_dataset_size": None,  # số lượng training samples
        "eval_dataset_size": None,   # số lượng validation samples
        
        # Hardware config
        "num_gpus": 4,
        "gpu_type": "RTX 2080Ti",
        "gradient_checkpointing": True,
        "fp16": True,
    }
)
# Định nghĩa các đối số training
args = SentenceTransformerTrainingArguments(
    output_dir="output_dir",
    num_train_epochs=50,
    per_device_train_batch_size= 4,             
    gradient_accumulation_steps= 8,            
    per_device_eval_batch_size= 4,
    gradient_checkpointing=True,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    save_steps=500,
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    max_grad_norm=1.0,
    metric_for_best_model="eval_dim_512_cosine_ndcg@10",
    report_to=["wandb"],
    run_name=wandb.run.name
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()
# Tạo trainer và truyền model vào
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=pairs,
    loss=train_loss,
    evaluator=evaluator,
)

# Bắt đầu train
trainer.train()  # Dọn dẹp tiến trình sau khi train
