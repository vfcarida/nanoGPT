import pytest
import torch
import math
from typing import Optional

from nanogpt.model import GPT, GPTConfig, apply_rotary_emb, precompute_freqs_cis

def test_model_initialization():
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    assert model is not None
    assert model.get_num_params() > 0

def test_rope_embeddings_shapes():
    """Test RoPE output shapes and correctness."""
    batch_size, seq_len, n_head, head_dim = 2, 64, 4, 16
    xq = torch.randn(batch_size, seq_len, n_head, head_dim)
    xk = torch.randn(batch_size, seq_len, n_head, head_dim)
    
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len * 2)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cos[:seq_len], freqs_sin[:seq_len])
    
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape
    assert not torch.isnan(xq_out).any()

def test_forward_pass_no_targets():
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    
    idx = torch.randint(0, 100, (2, 64))
    logits, loss = model(idx)
    
    # Check that logits are restricted to the last sequence step
    assert logits.shape == (2, 1, 100)
    assert loss is None

def test_forward_pass_with_targets():
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    
    idx = torch.randint(0, 100, (2, 64))
    targets = torch.randint(0, 100, (2, 64))
    
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 64, 100)
    assert loss is not None
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_out_of_bounds_block_size():
    """Boundary test: sequence length exceeds block size."""
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    
    idx = torch.randint(0, 100, (2, 128)) # Exceeds block_size 64
    with pytest.raises(ValueError, match="Cannot forward sequence of length 128"):
        model(idx)

def test_invalid_config_initialization():
    """Injection test: missing vocab size."""
    with pytest.raises(ValueError):
        config = GPTConfig(vocab_size=None, block_size=128, n_layer=2, n_head=2, n_embd=64)
        model = GPT(config)

def test_grouped_query_attention():
    """Boundary test for GQA where kv heads < q heads."""
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    model = GPT(config)
    idx = torch.randint(0, 100, (2, 32))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, 100)

def test_null_pointers_and_empty_tensors():
    """Fault injection: testing empty inputs."""
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    
    idx = torch.empty(2, 0, dtype=torch.long)
    
    # Should work even with 0 sequence length technically, but in this implementation, 
    # the behavior depends on the embedding and layer norm.
    # The current transformer design will pass this through, but lm_head on x[:, [-1], :] will fail.
    with pytest.raises(IndexError):
        logits, loss = model(idx)

def test_generate():
    config = GPTConfig(vocab_size=100, block_size=128, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)
    idx = torch.randint(0, 100, (2, 10))
    out = model.generate(idx, max_new_tokens=5, top_k=2)
    assert out.shape == (2, 15)
