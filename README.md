<div align="center">

# 🌌 nanoGPT - Enterprise Edition

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking](https://img.shields.io/badge/type_checker-mypy-blue.svg)](http://mypy-lang.org/)

O **repositório mais rápido e simples para treinar/afinar modelos GPT** em médias escalas. 
Baseado na obra-prima original de [Andrej Karpathy](https://github.com/karpathy/nanoGPT), esta versão foi rigorosamente **refatorada para padrões Enterprise**, introduzindo arquitetura modular, Type Hinting, logging nativo estruturado, suporte total a empacotamento PyPI e separação estrita de responsabilidades (SOLID).

</div>

---

## 📖 Visão Geral

`nanoGPT` é um repositório voltado para pesquisa e treinamento de modelos de linguagem (LLMs) estilo GPT-2, desenhado para ser legível, modificável e altamente otimizado (usando PyTorch 2.0+ `torch.compile` e `Flash Attention`).

Esta versão traz melhorias estruturais profundas em relação ao código original focado em scripts soltos:
- **Clean Architecture:** O núcleo do modelo (`GPT`, blocos transformer, trainer) foi isolado no pacote `src/nanogpt/`.
- **Manutenibilidade:** Códigos 100% tipados e documentados (Docstrings).
- **Sem Side-effects Globais:** Configurações carregam dicionários ou classes seguras no lugar de `exec(open(...))` poluindo o namespace local.
- **Testes Prontos:** Cobertura de fluxo principal garantida com `pytest`.

## 📂 Estrutura do Projeto

Abaixo a visão de diretórios de forma limpa e organizada:

```text
nanoGPT/
├── src/
│   └── nanogpt/              # Pacote Core
│       ├── model.py          # Arquitetura Transformer e GPT
│       ├── trainer.py        # Loop de treinamento desacoplado
│       └── utils/
│           └── configurator.py # Parser de config seguro
├── scripts/                  # Pontos de entrada CLI
│   ├── train.py
│   ├── sample.py
│   └── bench.py
├── tests/                    # Suíte de testes (PyTest)
├── config/                   # Configs baseadas em Python (herdadas)
├── data/                     # Scripts de dados e datasets brutos
├── pyproject.toml            # Manifestos PEP 621 e build-system
└── requirements.txt          # Bloqueio de dependências
```

## 🚀 Instalação e Pré-requisitos

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/nanoGPT.git
   cd nanoGPT
   ```

2. **Crie um ambiente virtual (Recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # No Windows use: venv\Scripts\activate
   ```

3. **Instale as dependências com suporte a módulos:**
   ```bash
   pip install -e .
   ```

## 🛠 Exemplos de Uso

### 1. Treinamento de um Modelo Baby GPT
Para treinar um GPT minúsculo no conjunto de dados de textos do Shakespeare:

Primeiro, prepare o dataset:
```bash
python data/shakespeare_char/prepare.py
```

Em seguida, execute o script de treino apontando para um arquivo de configuração customizado na pasta `config/`:
```bash
python scripts/train.py config/train_shakespeare_char.py
```

### 2. DDP (Distributed Data Parallel)
O `nanoGPT` suporta execução assíncrona transparente em múltiplas GPUs através do `torchrun`. Exemplo para treinar com 4 GPUs em um único nó:
```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py
```

### 3. Geração de Textos (Inferência)
Para amostrar tokens gerados pelo seu modelo após o treino (assume que os pesos foram salvos em `out/`):
```bash
python scripts/sample.py --out_dir=out
```

## 🤝 Contribuição

Interessado em escalar esta infraestrutura? Veja o nosso guia [CONTRIBUTING.md](./CONTRIBUTING.md) para padrões de commit, fluxos de branch (Git Flow) e linting.

## 📜 Licença e Créditos

Este repositório é licenciado através da **MIT License**.

- Autor original do nanoGPT: [Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- Refatoração Enterprise/Sênior: **Vinicius**
