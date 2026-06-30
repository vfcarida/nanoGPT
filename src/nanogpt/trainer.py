"""
Isolated training module for nanoGPT.
Contains the Trainer class that encapsulates the optimization loop, validation, and checkpointing.
Refactored for Production-Grade Robustness: NaN Loss Catching, DDP sync optimizations, Memory Cleanup.
"""

import os
import time
import math
import logging
from typing import Dict, Any, Callable, Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import GPT

logger = logging.getLogger(__name__)

class Trainer:
    """
    Class responsible for executing the training loop for the GPT model.
    Encapsulates learning rate control, loss estimation, backward passes, and logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        get_batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
        scaler: torch.cuda.amp.GradScaler,
        config: Dict[str, Any],
        ctx: Any, 
        device: str,
        master_process: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch_fn
        self.scaler = scaler
        self.config = config
        self.ctx = ctx
        self.device = device
        self.master_process = master_process
        
        self.iter_num = config.get('iter_num', 0)
        self.best_val_loss = config.get('best_val_loss', 1e9)
        self.local_iter_num = 0
        self.running_mfu = -1.0
        
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int) -> Dict[str, float]:
        """
        Estimates the loss using multiple batches from both training and validation splits.
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                if X is None or Y is None:
                    continue # Defensive programming: skip broken batches
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def get_lr(self, it: int) -> float:
        """
        Calculates the current learning rate based on cosine decay with warmup.
        """
        lr = self.config['learning_rate']
        warmup_iters = self.config['warmup_iters']
        lr_decay_iters = self.config['lr_decay_iters']
        min_lr = self.config['min_lr']
        
        if it < warmup_iters:
            return lr * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def save_checkpoint(self, val_loss: float):
        """
        Saves the model to disk if the current validation loss is the best achieved so far.
        """
        out_dir = self.config['out_dir']
        always_save_checkpoint = self.config['always_save_checkpoint']
        
        if val_loss < self.best_val_loss or always_save_checkpoint:
            self.best_val_loss = val_loss
            if self.iter_num > 0:
                checkpoint = {
                    'model': self.raw_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'model_args': self.config.get('model_args', {}),
                    'iter_num': self.iter_num,
                    'best_val_loss': self.best_val_loss,
                    'config': self.config,
                }
                os.makedirs(out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                logger.info(f"Saved checkpoint to {out_dir}")

    def train(self):
        """
        Main training loop.
        """
        X, Y = self.get_batch('train')
        if X is None or Y is None:
            raise ValueError("Data loader returned null batches on initialization.")

        t0 = time.time()
        
        gradient_accumulation_steps = self.config['gradient_accumulation_steps']
        max_iters = self.config['max_iters']
        eval_interval = self.config['eval_interval']
        eval_iters = self.config['eval_iters']
        eval_only = self.config['eval_only']
        decay_lr = self.config['decay_lr']
        learning_rate = self.config['learning_rate']
        grad_clip = self.config['grad_clip']
        log_interval = self.config['log_interval']
        batch_size = self.config['batch_size']
        wandb_log = self.config['wandb_log']
        
        if wandb_log and self.master_process:
            import wandb

        while True:
            lr = self.get_lr(self.iter_num) if decay_lr else learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.iter_num % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss(eval_iters)
                
                # Check for NaN validation loss
                if math.isnan(losses['train']) or math.isnan(losses['val']):
                    logger.error(f"NaN loss detected during evaluation at step {self.iter_num}. Aborting training to prevent state corruption.")
                    break

                logger.info(f"Step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                if wandb_log:
                    wandb.log({
                        "iter": self.iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": self.running_mfu * 100,
                    })
                
                self.save_checkpoint(losses['val'])
                
            if self.iter_num == 0 and eval_only:
                break

            for micro_step in range(gradient_accumulation_steps):
                is_last_micro_step = (micro_step == gradient_accumulation_steps - 1)
                
                if isinstance(self.model, DDP):
                    self.model.require_backward_grad_sync = is_last_micro_step
                    
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    
                    if torch.isnan(loss):
                        logger.error(f"NaN Loss detected in forward pass at iter {self.iter_num}. Exiting to prevent model breakdown.")
                        raise RuntimeError("NaN Loss during training.")
                        
                    loss = loss / gradient_accumulation_steps
                    
                X, Y = self.get_batch('train')
                
                self.scaler.scale(loss).backward()
                
            if grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Use set_to_none=True to optimize memory overhead
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            if self.iter_num % log_interval == 0 and self.master_process:
                lossf = loss.item() * gradient_accumulation_steps
                if self.local_iter_num >= 5: 
                    mfu = self.raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
                    
                logger.info(f"Iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {self.running_mfu*100:.2f}%")
                
            self.iter_num += 1
            self.local_iter_num += 1

            if self.iter_num > max_iters:
                break
