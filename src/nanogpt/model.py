"""
GPT Model implementation with State-of-the-Art improvements:
- RMSNorm
- RoPE (Rotary Position Embeddings)
- SwiGLU Activations
- Grouped-Query Attention (GQA)
"""

import math
import inspect
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for RoPE (Rotary Position Embeddings).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    """
    Apply Rotary Position Embeddings to query and key tensors.
    xq, xk: (B, T, n_head, head_dim)
    """
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = freqs_cos.view(1, freqs_cos.shape[0], 1, freqs_cos.shape[1])
    freqs_sin = freqs_sin.view(1, freqs_sin.shape[0], 1, freqs_sin.shape[1])

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization. 
    Faster and more memory efficient than standard LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention mechanism with Grouped-Query Attention (GQA).
    """
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"Embedding dim ({config.n_embd}) not divisible by heads ({config.n_head})")
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        # Projections
        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias", 
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos[:T], freqs_sin[:T])
        
        # Repeat KV heads for GQA
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(B, T, self.n_head, self.head_dim)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(B, T, self.n_head, self.head_dim)
            
        xq = xq.transpose(1, 2) # (B, nh, T, hs)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (xq @ xk.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ xv
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit. Better asymptotic performance than GELU.
    """
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3) # Scale down due to 3 matrices instead of 2
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """
    Transformer Block: Self-Attention followed by SwiGLU MLP.
    """
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), freqs_cos, freqs_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    GPT architecture configuration parameters.
    """
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None # Grouped-Query Attention
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # Standardized to False for modern architectures


class GPT(nn.Module):
    """
    Complete GPT Model leveraging RoPE, RMSNorm, GQA, and SwiGLU.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.vocab_size is None or config.block_size is None:
            raise ValueError("vocab_size and block_size must be specified in config.")
            
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        num_params = self.get_num_params() / 1e6
        logger.info(f"Number of parameters: {num_params:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        # Removed wpe subtraction since we no longer have absolute positional embeddings
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}")
            
        freqs_cos = self.freqs_cos.to(device)
        freqs_sin = self.freqs_sin.to(device)

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        
        for block in self.transformer.h:
            x = block(x, freqs_cos, freqs_sin)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) 
        flops_promised = 312e12 
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
