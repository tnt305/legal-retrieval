import wandb
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers


wandb.login(key="02ba155e26496a78f062f683274330566fefe94c")
wandb.init(
    project="sentence_sim",  # Đặt tên project phù hợp
    name="experiment-v1",
    config={
        # Training hyperparameters
        "model_name": "",  # tên model base bạn dùng
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

args = SentenceTransformerTrainingArguments(
    output_dir="output_dir",
    num_train_epochs=50,
    # 2080Ti has 11GB VRAM, reduced batch size for multi-GPU training
    per_device_train_batch_size=4,             # reduced from 8 to 4
    gradient_accumulation_steps=8,             # increased to maintain effective batch size
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,               # enabled to save memory
    warmup_ratio=0.1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",                       # changed from fused to regular adamw for better compatibility
    fp16=True,                                 # keep fp16 for memory efficiency
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    save_steps=500,
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_dim_768_cosine_ndcg@10",
    early_stopping_patience=3,
    early_stopping_threshold=0.0001,
    # Thêm wandb config
    report_to=["wandb"],          # Enable wandb logging
    run_name=wandb.run.name      # Sử dụng tên run từ wandb
)