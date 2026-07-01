import os
import pytest
import numpy as np
import torch

from nanogpt.utils.dataloader import PrefetchDataLoader

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    # Create temporary train.bin and val.bin
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Write 1000 tokens
    train_data = np.random.randint(0, 1000, size=1000, dtype=np.uint16)
    val_data = np.random.randint(0, 1000, size=500, dtype=np.uint16)
    
    train_data.tofile(data_dir / "train.bin")
    val_data.tofile(data_dir / "val.bin")
    
    return str(data_dir)

def test_dataloader_initialization_and_prefetch(dummy_dataset_dir):
    batch_size = 4
    block_size = 8
    
    dataloader = PrefetchDataLoader(
        data_dir=dummy_dataset_dir,
        batch_size=batch_size,
        block_size=block_size,
        device='cpu',
        prefetch_factor=2
    )
    
    try:
        # Fetch batches
        x, y = dataloader.get_batch('train')
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        
        assert (x >= 0).all() and (x < 1000).all()
        
        x_val, y_val = dataloader.get_batch('val')
        assert x_val.shape == (batch_size, block_size)
        assert y_val.shape == (batch_size, block_size)
    finally:
        dataloader.close()

def test_dataloader_invalid_directory():
    with pytest.raises(FileNotFoundError):
        # Invalid directory
        dataloader = PrefetchDataLoader(
            data_dir="./non_existent_directory_xyz",
            batch_size=4,
            block_size=8,
            device='cpu'
        )

def test_dataloader_close_cleanup(dummy_dataset_dir):
    dataloader = PrefetchDataLoader(
        data_dir=dummy_dataset_dir,
        batch_size=2,
        block_size=4,
        device='cpu'
    )
    
    # Verify worker threads are active
    for thread in dataloader.threads.values():
        assert thread.is_alive()
        
    dataloader.close()
    
    # Verify worker threads are stopped or joining
    for thread in dataloader.threads.values():
        thread.join(timeout=1.0)
        assert not thread.is_alive()
