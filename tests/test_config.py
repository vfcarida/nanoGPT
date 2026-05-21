import sys
import pytest
from unittest.mock import patch
from nanogpt.utils.configurator import update_config_from_args

def test_update_config_from_args_key_value():
    default_config = {
        'batch_size': 12,
        'learning_rate': 1e-4,
        'flag': False,
        'name': 'test'
    }
    
    test_args = ['script_name', '--batch_size=32', '--learning_rate=2e-4', '--flag=True', '--name=production']
    
    with patch.object(sys, 'argv', test_args):
        new_config = update_config_from_args(default_config)
        
    assert new_config['batch_size'] == 32
    assert new_config['learning_rate'] == 2e-4
    assert new_config['flag'] is True
    assert new_config['name'] == 'production'

def test_update_config_from_args_invalid_key():
    default_config = {'batch_size': 12}
    test_args = ['script_name', '--invalid_key=10']
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(ValueError):
            update_config_from_args(default_config)
