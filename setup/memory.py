import os
import gc
import psutil
import torch
import torch.cuda.amp as amp
from torch import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

class MemoryEfficientTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = amp.GradScaler()
        
    def training_step(self, model, inputs):
        # Clear cache before each step
        torch.cuda.empty_cache()
        gc.collect()
        
        # Move inputs to GPU only when needed
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        with amp.autocast():
            outputs = model(**inputs)
            loss = outputs.loss / self.args.gradient_accumulation_steps
            
        self.scaler.scale(loss).backward()
        
        if self.steps % self.args.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        return loss.detach()