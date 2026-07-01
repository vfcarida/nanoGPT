"""
Simplified version of training specifically for benchmarking purposes.
"""

import os
import sys
import time
import logging
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from nanogpt.model import GPTConfig, GPT
from nanogpt.utils.configurator import update_config_from_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'batch_size': 12,
    'block_size': 1024,
    'bias': False,
    'real_data': True,
    'seed': 1337,
    'device': 'cuda',
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    'compile': True,
    'profile': False,
    'gradient_checkpointing': False,
}

def main():
    config = update_config_from_args(DEFAULT_CONFIG)
    
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    
    device = config['device']
    device_type = 'cuda' if 'cuda' in device else 'cpu' 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    dataloader = None
    if config['real_data']:
        dataset = 'openwebtext'
        data_dir = os.path.join('data', dataset)
        try:
            from nanogpt.utils import PrefetchDataLoader
            dataloader = PrefetchDataLoader(
                data_dir=data_dir,
                batch_size=config['batch_size'],
                block_size=config['block_size'],
                device=device
            )
            get_batch = dataloader.get_batch
        except FileNotFoundError:
            logger.warning("Real data not found. Falling back to synthetic data.")
            config['real_data'] = False
            
    if not config['real_data']:
        x = torch.randint(50304, (config['batch_size'], config['block_size']), device=device)
        y = torch.randint(50304, (config['batch_size'], config['block_size']), device=device)
        get_batch = lambda split: (x, y)

    gptconf = GPTConfig(
        block_size = config['block_size'], 
        n_layer = 12, n_head = 12, n_embd = 768,
        dropout = 0, 
        bias = config['bias'],
        gradient_checkpointing = config['gradient_checkpointing'],
    )
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

    if config['compile']:
        logger.info("Compiling model...")
        model = torch.compile(model) 

    try:
        if config['profile']:
            wait, warmup, active = 5, 5, 5
            num_steps = wait + warmup + active
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
                record_shapes=False,
                profile_memory=False,
                with_stack=False, 
                with_flops=True,
                with_modules=False, 
            ) as prof:
                X, Y = get_batch('train')
                for k in range(num_steps):
                    with ctx:
                        logits, loss = model(X, Y)
                    X, Y = get_batch('train')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    lossf = loss.item()
                    logger.info(f"{k}/{num_steps} loss: {lossf:.4f}")
                    prof.step() 
        else:
            torch.cuda.synchronize()
            for stage, num_steps in enumerate([10, 20]): 
                t0 = time.time()
                X, Y = get_batch('train')
                for k in range(num_steps):
                    with ctx:
                        logits, loss = model(X, Y)
                    X, Y = get_batch('train')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    lossf = loss.item()
                    logger.info(f"{k}/{num_steps} loss: {lossf:.4f}")
                torch.cuda.synchronize()
                t1 = time.time()
                dt = t1-t0
                mfu = model.estimate_mfu(config['batch_size'] * 1 * num_steps, dt)
                if stage == 1:
                    logger.info(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")

    finally:
        if dataloader is not None:
            dataloader.close()

if __name__ == '__main__':
    main()
