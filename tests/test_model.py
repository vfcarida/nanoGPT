import pytest
import torch
from nanogpt.model import GPT, GPTConfig

def test_gpt_initialization():
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32
    )
    model = GPT(config)
    assert model is not None
    assert model.get_num_params(non_embedding=True) > 0

def test_gpt_forward():
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32
    )
    model = GPT(config)
    
    # Dummy input
    idx = torch.randint(0, 100, (2, 10)) # Batch size 2, Sequence length 10
    
    logits, loss = model(idx)
    assert logits.shape == (2, 1, 100) # Inference optimization returns only the last token logit
    assert loss is None
    
    # With targets
    targets = torch.randint(0, 100, (2, 10))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 10, 100)
    assert loss is not None
    assert loss.item() >= 0
