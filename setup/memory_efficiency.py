import os
import gc
import psutil
import torch
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint_sequential
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

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