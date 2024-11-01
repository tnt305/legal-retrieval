#!/usr/bin/env python3
"""
Training script for sentence transformer model with matryoshka loss and evaluation.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim

from setup.settings import setup_training_args, setup_embedding_model
from src.preprocessor.utils.dataset_level import read_pickle, prepare_training_dataset, read_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration class for model training parameters."""
    
    def __init__(self):
        self.data_dir = Path('/home/thiendc/projects/legal_retrieval/data/processed')
        self.model_name = 'dangvantuan/vietnamese-embedding'
        self.vocab_path = './src/preprocessor/vocab/data/update_vocab_v1.json'
        self.output_dir = "output_dir"
        self.matryoshka_dimensions = [768, 512, 256]  # Large to small
        self.num_train_samples = 5000
        
        # Training arguments
        self.training_args = {
            'num_train_epochs': 10,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'per_device_eval_batch_size': 8,
            'gradient_checkpointing': True,
            'warmup_ratio': 0.1,
            'learning_rate': 2e-5,
            'lr_scheduler_type': "cosine",
            'optim': "adamw_torch_fused",
            'fp16': True,
            'batch_sampler': BatchSamplers.NO_DUPLICATES,
            'eval_strategy': "steps",
            'save_steps': 500,
            'logging_steps': 10,
            'save_total_limit': 3,
            'load_best_model_at_end': True,
            'max_grad_norm': 1.0,
            'metric_for_best_model': "eval_dim_768_cosine_ndcg@10",
        }

def load_datasets(config: ModelConfig) -> Tuple[Dict, Dict, Dict]:
    """Load and validate datasets from pickle files."""
    try:
        logger.info("Loading datasets...")
        corpus = read_pickle(config.data_dir / 'corpus.pkl')
        queries = read_pickle(config.data_dir / 'queries.pkl')
        relevant_docs = read_pickle(config.data_dir / 'relevant_docs.pkl')
        
        # Validate data
        assert len(corpus) > 0, "Corpus is empty"
        assert len(queries) > 0, "Queries are empty"
        assert len(relevant_docs) > 0, "Relevant docs are empty"
        
        logger.info(f"Loaded {len(corpus)} corpus documents, {len(queries)} queries")
        return corpus, queries, relevant_docs
    
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def create_evaluators(
    queries: Dict,
    corpus: Dict,
    relevant_docs: Dict,
    dimensions: List[int]
) -> SequentialEvaluator:
    """Create sequential evaluator with multiple dimensions."""
    logger.info("Creating evaluators...")
    evaluators = []
    
    for dim in dimensions:
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        evaluators.append(evaluator)
        
    return SequentialEvaluator(evaluators)

def setup_trainer(
    config: ModelConfig,
    model,
    train_dataset: Dataset,
    evaluator: SequentialEvaluator
) -> SentenceTransformerTrainer:
    """Setup trainer with specified configuration."""
    logger.info("Setting up trainer...")
    
    # Setup loss functions
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model,
        inner_train_loss,
        matryoshka_dims=config.matryoshka_dimensions
    )
    
    # Create training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        **config.training_args
    )
    
    return SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

def main():
    """Main training function."""
    try:
        # Initialize configuration
        config = ModelConfig()
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load datasets
        corpus, queries, relevant_docs = load_datasets(config)
        
        # Prepare training dataset
        logger.info("Preparing training dataset...")
        train_dataset = prepare_training_dataset(queries, corpus, relevant_docs)
        subset = Dataset.from_dict(train_dataset[:config.num_train_samples])
        
        # Load model and tokenizer
        logger.info("Setting up model and tokenizer...")
        new_tokens = read_json(config.vocab_path)
        model, _ = setup_embedding_model(config.model_name, new_tokens=new_tokens)
        model.to(device)
        
        # Create evaluators
        evaluator = create_evaluators(
            queries,
            corpus,
            relevant_docs,
            config.matryoshka_dimensions
        )
        
        # Setup and start training
        trainer = setup_trainer(config, model, subset, evaluator)
        
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()