"""
Utilitário de configuração para carregar hiperparâmetros de arquivos e CLI.
Refatorado para não usar side-effects em variáveis globais do sistema.
"""

import sys
import logging
from ast import literal_eval
from typing import Dict, Any

logger = logging.getLogger(__name__)

def update_config_from_args(default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lê os argumentos da linha de comando e atualiza o dicionário de configurações base.
    Pode carregar arquivos .py (executando-os em um namespace isolado) ou argumentos --key=value.
    
    Args:
        default_config: Dicionário contendo as configurações padrão.
        
    Returns:
        Um novo dicionário com as configurações atualizadas.
    """
    config = default_config.copy()
    
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            if arg.startswith('--'):
                raise ValueError(f"Invalid config file format: {arg}")
            config_file = arg
            logger.info(f"Overriding config with {config_file}")
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                # Exec in isolated namespace
                exec(code, {}, config)
            except Exception as e:
                logger.error(f"Failed to load config file {config_file}: {e}")
                raise
        else:
            # assume it's a --key=value argument
            if not arg.startswith('--'):
                raise ValueError(f"Invalid argument format: {arg}")
            key, val = arg.split('=', 1)
            key = key[2:]
            
            if key in config:
                try:
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = val
                
                # Check type and attempt cast if needed
                expected_type = type(config[key])
                if not isinstance(attempt, expected_type) and config[key] is not None:
                    try:
                        # Allow automatic cast like '1' to int
                        attempt = expected_type(attempt)
                    except (ValueError, TypeError):
                        logger.warning(f"Type mismatch for {key}: expected {expected_type}, got {type(attempt)}")
                        
                logger.info(f"Overriding: {key} = {attempt}")
                config[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
                
    return config
