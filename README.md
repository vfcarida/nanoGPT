<div align="center">

# 🌌 nanoGPT - Enterprise Edition

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking](https://img.shields.io/badge/type_checker-mypy-blue.svg)](http://mypy-lang.org/)

The **simplest and fastest repository for training/fine-tuning medium-sized GPT models**. 
Based on the original masterpiece by [Andrej Karpathy](https://github.com/karpathy/nanoGPT), this version has been rigorously **refactored to Enterprise standards**, introducing a modular architecture, Type Hinting, structured native logging, full PyPI packaging support, and strict separation of concerns (SOLID).

</div>

---

## 📖 Overview

`nanoGPT` is a repository aimed at researching and training GPT-2 style language models (LLMs). It is designed to be readable, hackable, and highly optimized (using PyTorch 2.0+ `torch.compile` and `Flash Attention`).

This version brings profound structural improvements over the original script-heavy code:
- **Clean Architecture:** The core of the model (`GPT`, transformer blocks, trainer) has been isolated in the `src/nanogpt/` package.
- **Maintainability:** 100% typed and documented code (Docstrings).
- **No Global Side-effects:** Configurations load dictionaries or safe classes instead of using `exec(open(...))` which pollutes the local namespace.
- **Test-Ready:** Main flow coverage is guaranteed with `pytest`.

## 📂 Project Structure

Below is the clean and organized directory structure:

```text
nanoGPT/
├── src/
│   └── nanogpt/              # Core Package
│       ├── model.py          # Transformer and GPT Architecture
│       ├── trainer.py        # Decoupled training loop
│       └── utils/
│           └── configurator.py # Safe configuration parser
├── scripts/                  # CLI Entry Points
│   ├── train.py
│   ├── sample.py
│   └── bench.py
├── tests/                    # Test Suite (PyTest)
├── config/                   # Python-based configs (inherited)
├── data/                     # Data scripts and raw datasets
├── pyproject.toml            # PEP 621 manifests and build-system
└── requirements.txt          # Dependency lock file
```

## 🚀 Installation & Prerequisites

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/nanoGPT.git
   cd nanoGPT
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies with module support:**
   ```bash
   pip install -e .
   ```

## 🛠 Usage Examples

### 1. Training a Baby GPT Model
To train a tiny GPT on the Shakespeare dataset:

First, prepare the dataset:
```bash
python data/shakespeare_char/prepare.py
```

Then, run the training script pointing to a custom configuration file in the `config/` folder:
```bash
python scripts/train.py config/train_shakespeare_char.py
```

### 2. DDP (Distributed Data Parallel)
`nanoGPT` supports transparent asynchronous execution across multiple GPUs using `torchrun`. For example, to train on 4 GPUs in a single node:
```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py
```

### 3. Text Generation (Inference)
To sample generated tokens from your model after training (assuming weights are saved in `out/`):
```bash
python scripts/sample.py --out_dir=out
```

## 🤝 Contributing

Interested in scaling this infrastructure? Check out our [CONTRIBUTING.md](./CONTRIBUTING.md) guide for commit standards, branching workflows (Git Flow), and linting.

## 📜 License & Credits

This repository is licensed under the **MIT License**.

- Original author of nanoGPT: [Andrej Karpathy](https://github.com/karpathy/nanoGPT)
