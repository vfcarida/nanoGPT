"""
Script de treinamento principal.
Pode ser executado em uma GPU (debug mode) ou com Distributed Data Parallel (DDP).

Exemplo de uso (single GPU):
$ python scripts/train.py --batch_size=32 --compile=False

Exemplo com DDP (4 GPUs):
$ torchrun --standalone --nproc_per_node=4 scripts/train.py
"""

import os
import sys
import pickle
from contextlib import nullcontext
import logging

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Fallback temporário caso o pacote nanogpt não esteja instalado via pip
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nanogpt.model import GPTConfig, GPT
from nanogpt.trainer import Trainer
from nanogpt.utils.configurator import update_config_from_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configurações Padrão
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    'out_dir': 'out',
    'eval_interval': 2000,
    'log_interval': 1,
    'eval_iters': 200,
    'eval_only': False,
    'always_save_checkpoint': True,
    'init_from': 'scratch', # 'scratch' or 'resume' or 'gpt2*'
    
    'wandb_log': False,
    'wandb_project': 'owt',
    'wandb_run_name': 'gpt2',
    
    'dataset': 'openwebtext',
    'gradient_accumulation_steps': 5 * 8,
    'batch_size': 12,
    'block_size': 1024,
    
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'dropout': 0.0,
    'bias': False,
    
    'learning_rate': 6e-4,
    'max_iters': 600000,
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    
    'decay_lr': True,
    'warmup_iters': 2000,
    'lr_decay_iters': 600000,
    'min_lr': 6e-5,
    
    'backend': 'nccl',
    'device': 'cuda',
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    'compile': True,
}

def main():
    config = update_config_from_args(DEFAULT_CONFIG)
    
    # Iniciação do DDP (Distributed Data Parallel)
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        
        assert config['gradient_accumulation_steps'] % ddp_world_size == 0
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config['device']
        
    tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
    if master_process:
        logger.info(f"Tokens per iteration will be: {tokens_per_iter:,}")
        os.makedirs(config['out_dir'], exist_ok=True)
        
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Carregamento de dados (Dataloader rudimentar)
    data_dir = os.path.join('data', config['dataset'])
    
    def get_batch(split: str):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
            
        ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
        x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
        
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            
        return x, y

    # Recupera metadados do vocabulário
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        if master_process:
            logger.info(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Setup do Modelo
    model_args = dict(
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_embd=config['n_embd'], 
        block_size=config['block_size'],
        bias=config['bias'], 
        vocab_size=None, 
        dropout=config['dropout']
    )
    
    if config['init_from'] == 'scratch':
        if master_process: logger.info("Initializing a new model from scratch")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif config['init_from'] == 'resume':
        if master_process: logger.info(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
            
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        config['iter_num'] = checkpoint['iter_num']
        config['best_val_loss'] = checkpoint['best_val_loss']
        
    elif config['init_from'].startswith('gpt2'):
        if master_process: logger.info(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
        override_args = dict(dropout=config['dropout'])
        model = GPT.from_pretrained(config['init_from'], override_args)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
            
    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']
        
    config['model_args'] = model_args
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
    
    if config['init_from'] == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])

    if config['compile']:
        if master_process: logger.info("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
        
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        
    if wandb_log and master_process:
        import wandb
        wandb.init(project=config['wandb_project'], name=config['wandb_run_name'], config=config)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        get_batch_fn=get_batch,
        scaler=scaler,
        config=config,
        ctx=ctx,
        device=device,
        master_process=master_process
    )
    
    trainer.train()

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()
