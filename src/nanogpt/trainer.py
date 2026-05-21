"""
Módulo de treinamento isolado do nanoGPT.
Contém a classe Trainer que encapsula o loop de otimização, validação e checkpointing.
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
    Classe responsável por executar o loop de treinamento do modelo GPT.
    Encapsula o controle de learning rate, estimativa de loss, backward pass e logging.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        get_batch_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]],
        scaler: torch.cuda.amp.GradScaler,
        config: Dict[str, Any],
        ctx: Any, # Context manager for mixed precision (amp.autocast or nullcontext)
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
        
        # Estado interno
        self.iter_num = config.get('iter_num', 0)
        self.best_val_loss = config.get('best_val_loss', 1e9)
        self.local_iter_num = 0
        self.running_mfu = -1.0
        
        # Unwrap model from DDP se necessário para acesso a métodos específicos
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int) -> Dict[str, float]:
        """
        Estima a loss usando múltiplos batches nos splits de treino e validação.
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def get_lr(self, it: int) -> float:
        """
        Calcula a learning rate atual baseada em cosine decay com warmup.
        """
        lr = self.config['learning_rate']
        warmup_iters = self.config['warmup_iters']
        lr_decay_iters = self.config['lr_decay_iters']
        min_lr = self.config['min_lr']
        
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return lr * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (lr - min_lr)

    def save_checkpoint(self, val_loss: float):
        """
        Salva o modelo no disco se for o melhor loss atingido.
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
                logger.info(f"Saving checkpoint to {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    def train(self):
        """
        Loop principal de treinamento.
        """
        X, Y = self.get_batch('train')
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
        
        # Integration with wandb se habilitado e for master_process
        if wandb_log and self.master_process:
            import wandb

        while True:
            # Update learning rate
            lr = self.get_lr(self.iter_num) if decay_lr else learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Eval phase
            if self.iter_num % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss(eval_iters)
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

            # Forward e backward passes com gradient accumulation
            for micro_step in range(gradient_accumulation_steps):
                is_last_micro_step = (micro_step == gradient_accumulation_steps - 1)
                
                if isinstance(self.model, DDP):
                    self.model.require_backward_grad_sync = is_last_micro_step
                    
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / gradient_accumulation_steps
                    
                # Async prefetch next batch
                X, Y = self.get_batch('train')
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
            if grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Timing e logging local
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            if self.iter_num % log_interval == 0 and self.master_process:
                lossf = loss.item() * gradient_accumulation_steps
                if self.local_iter_num >= 5: # let the loop settle
                    mfu = self.raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
                    
                logger.info(f"Iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {self.running_mfu*100:.2f}%")
                
            self.iter_num += 1
            self.local_iter_num += 1

            # Condição de parada
            if self.iter_num > max_iters:
                break
