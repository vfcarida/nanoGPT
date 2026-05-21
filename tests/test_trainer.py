import pytest
import torch
from unittest.mock import MagicMock
from nanogpt.trainer import Trainer
from nanogpt.model import GPT, GPTConfig

def test_trainer_initialization():
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32
    )
    model = GPT(config)
    optimizer = MagicMock()
    scaler = MagicMock()
    ctx = MagicMock()
    
    trainer_config = {
        'iter_num': 0,
        'best_val_loss': 1e9,
        'learning_rate': 1e-4,
        'warmup_iters': 100,
        'lr_decay_iters': 1000,
        'min_lr': 1e-5,
    }
    
    def mock_get_batch(split):
        return torch.zeros((2, 16), dtype=torch.long), torch.zeros((2, 16), dtype=torch.long)
        
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        get_batch_fn=mock_get_batch,
        scaler=scaler,
        config=trainer_config,
        ctx=ctx,
        device="cpu"
    )
    
    assert trainer.iter_num == 0
    assert trainer.best_val_loss == 1e9
    
    # Test learning rate scheduling
    lr_warmup = trainer.get_lr(50)
    assert lr_warmup < 1e-4
    
    lr_decayed = trainer.get_lr(1100)
    assert lr_decayed == 1e-5
    
def test_trainer_estimate_loss():
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32
    )
    model = GPT(config)
    optimizer = MagicMock()
    scaler = MagicMock()
    from contextlib import nullcontext
    
    trainer_config = {}
    
    def mock_get_batch(split):
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))
        return x, y
        
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        get_batch_fn=mock_get_batch,
        scaler=scaler,
        config=trainer_config,
        ctx=nullcontext(),
        device="cpu"
    )
    
    losses = trainer.estimate_loss(eval_iters=2)
    assert 'train' in losses
    assert 'val' in losses
    assert isinstance(losses['train'], float)
    assert losses['train'] >= 0
