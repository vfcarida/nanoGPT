import pytest
import torch
import math
import os
import shutil
from typing import Tuple

from nanogpt.model import GPT, GPTConfig
from nanogpt.trainer import Trainer

@pytest.fixture
def dummy_model():
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=2, n_embd=32)
    return GPT(config)

@pytest.fixture
def dummy_optimizer(dummy_model):
    return torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)

@pytest.fixture
def default_config():
    return {
        'learning_rate': 1e-3,
        'warmup_iters': 10,
        'lr_decay_iters': 100,
        'min_lr': 1e-4,
        'out_dir': './temp_test_out',
        'always_save_checkpoint': True,
        'gradient_accumulation_steps': 1,
        'max_iters': 5,
        'eval_interval': 2,
        'eval_iters': 2,
        'eval_only': False,
        'decay_lr': True,
        'grad_clip': 1.0,
        'log_interval': 1,
        'batch_size': 2,
        'wandb_log': False,
    }

class DummyContext:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def dummy_get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns a batch of random tokens
    X = torch.randint(0, 100, (2, 64))
    Y = torch.randint(0, 100, (2, 64))
    return X, Y

def broken_get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns null batches to test fault tolerance
    return None, None

def test_trainer_initialization(dummy_model, dummy_optimizer, default_config):
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    trainer = Trainer(
        model=dummy_model,
        optimizer=dummy_optimizer,
        get_batch_fn=dummy_get_batch,
        scaler=scaler,
        config=default_config,
        ctx=DummyContext(),
        device='cpu'
    )
    assert trainer is not None

def test_trainer_learning_rate_schedule(dummy_model, dummy_optimizer, default_config):
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    trainer = Trainer(
        dummy_model, dummy_optimizer, dummy_get_batch, scaler, default_config, DummyContext(), 'cpu'
    )
    
    # Warmup phase
    lr_iter_0 = trainer.get_lr(0)
    assert lr_iter_0 < default_config['learning_rate']
    
    # Post-decay phase
    lr_iter_200 = trainer.get_lr(200)
    assert lr_iter_200 == default_config['min_lr']

def test_trainer_null_batch_fault_injection(dummy_model, dummy_optimizer, default_config):
    """Fault injection: testing how the trainer handles null batches from dataloader."""
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    trainer = Trainer(
        dummy_model, dummy_optimizer, broken_get_batch, scaler, default_config, DummyContext(), 'cpu'
    )
    
    with pytest.raises(ValueError, match="Data loader returned null batches"):
        trainer.train()

def test_trainer_checkpointing(dummy_model, dummy_optimizer, default_config):
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    trainer = Trainer(
        dummy_model, dummy_optimizer, dummy_get_batch, scaler, default_config, DummyContext(), 'cpu'
    )
    
    trainer.iter_num = 1 # Must be > 0 to save
    trainer.save_checkpoint(val_loss=0.5)
    
    assert os.path.exists(default_config['out_dir'])
    assert os.path.exists(os.path.join(default_config['out_dir'], 'ckpt.pt'))
    
    # Cleanup
    shutil.rmtree(default_config['out_dir'])

def test_trainer_training_loop_execution(dummy_model, dummy_optimizer, default_config):
    """Test standard execution of the training loop for a few steps."""
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    trainer = Trainer(
        dummy_model, dummy_optimizer, dummy_get_batch, scaler, default_config, DummyContext(), 'cpu'
    )
    
    try:
        trainer.train()
    except Exception as e:
        pytest.fail(f"Training loop raised an exception: {e}")
