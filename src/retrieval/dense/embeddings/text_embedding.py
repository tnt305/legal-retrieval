import pickle
import wandb
import random
import datasets from Dataset
from tqdm import tqdm
from retrieval.dense.settings import args
from retrieval.dense.utils import evaluate, prepare_training_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
tokenizer = AutoTokenizer.from_pretrained('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
tokenizer.add_tokens(['custom_token_1', 'custom_token_2', 'tổng_bí_thư'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))


def open_pickle_dataset(root_dir, file_name):
    pass
with open('./data/processed/queries_after.pkl', 'rb') as f:
    queries = pickle.load(f)

with open('./data/processed/corpus_after.pkl', 'rb') as f:
    corpus = pickle.load(f)

with open('./data/processed/relevant_docs.pkl', 'rb') as f:
    relevant_docs = pickle.load(f)
    
    
evaluator = evaluate(queries, corpus, relevant_docs)
pairs = prepare_training_dataset(queries, corpus, relevant_docs)


matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)


trainer = SentenceTransformerTrainer(
    model=model,
    args=args,  # training arguments
    train_dataset=pairs,
    loss=train_loss,
    evaluator=evaluator,
)

trainer.train()