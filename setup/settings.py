import yaml
import os
from src.preprocessor.utils.dataset_level import read_yaml
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, models

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

def setup_training_args(config_path="config.yaml"):
    # Load config values from YAML file
    """
    Set up training arguments from a YAML config file.

    This function loads the configuration from a YAML file and sets up the training
    arguments for the SentenceTransformer model. It also sets environment variables
    for PyTorch.

    Args:
    Returns:
        SentenceTransformerTrainingArguments: Configured training arguments.
    """
    config = read_yaml(config_path)
    
    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["environment"]["PYTORCH_CUDA_ALLOC_CONF"]
    
    # Initialize training arguments from the config file
    args = SentenceTransformerTrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        warmup_ratio=config["training"]["warmup_ratio"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        optim=config["training"]["optim"],
        fp16=config["training"]["fp16"],
        batch_sampler=config["training"]["batch_sampler"],
        eval_strategy=config["training"]["eval_strategy"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        max_grad_norm=config["training"]["max_grad_norm"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        dataloader_pin_memory=config["training"]["dataloader_pin_memory"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"]
    )

    return args