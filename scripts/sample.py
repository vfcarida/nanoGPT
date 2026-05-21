"""
Script para amostragem/geração a partir de um modelo treinado.
"""

import os
import sys
import pickle
import logging
from contextlib import nullcontext

import torch
import tiktoken

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from nanogpt.model import GPTConfig, GPT
from nanogpt.utils.configurator import update_config_from_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'init_from': 'resume', 
    'out_dir': 'out', 
    'start': '\n', 
    'num_samples': 10,
    'max_new_tokens': 500,
    'temperature': 0.8,
    'top_k': 200,
    'seed': 1337,
    'device': 'cuda',
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    'compile': False
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

    # model
    if config['init_from'] == 'resume':
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif config['init_from'].startswith('gpt2'):
        model = GPT.from_pretrained(config['init_from'], dict(dropout=0.0))
    else:
        raise ValueError(f"Invalid init_from: {config['init_from']}")

    model.eval()
    model.to(device)
    if config['compile']:
        model = torch.compile(model)

    # Decode meta
    load_meta = False
    if config['init_from'] == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: 
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        
    if load_meta:
        logger.info(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        logger.info("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    start = config['start']
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
            
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    logger.info("Starting generation...")
    with torch.no_grad():
        with ctx:
            for k in range(config['num_samples']):
                y = model.generate(x, config['max_new_tokens'], temperature=config['temperature'], top_k=config['top_k'])
                print(decode(y[0].tolist()))
                print('---------------')

if __name__ == '__main__':
    main()
