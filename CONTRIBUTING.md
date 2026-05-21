# Guia de Contribuição - nanoGPT Enterprise Edition

Primeiramente, obrigado pelo interesse em contribuir com a versão Enterprise do **nanoGPT**!

Este repositório visa manter um padrão elevado de Engenharia de Software. Siga os passos abaixo para submeter as suas contribuições.

## 1. Relatando Bugs ou Sugerindo Melhorias

- Certifique-se de que o problema ou funcionalidade não foi relatado anteriormente (busque nas issues).
- Crie uma issue descrevendo detalhadamente o contexto, como reproduzir o bug (se aplicável), e qual o comportamento esperado.

## 2. Padrões de Código (Clean Code e SOLID)

- **Tipagem Estática:** Use `typing` para 100% das assinaturas das funções (Type Hinting).
- **Docstrings:** Use o padrão Google ou NumPy para documentar a assinatura, retorno e descrição lógica de todos os métodos e classes públicas.
- **Princípio da Responsabilidade Única:** Evite inflar arquivos. Se uma função ou classe ficou grande demais e abriga múltiplas responsabilidades lógicas, refatore e modularize.

## 3. Ambiente de Desenvolvimento

Certifique-se de possuir o ambiente isolado (virtual environment) instalado. Nós gerenciamos dependências usando `pyproject.toml`.
Na raiz, execute:
```bash
pip install -e ".[dev]"
```

## 4. Testes (Pytest)

Não são aceitos Pull Requests sem a devida cobertura de testes para novas lógicas adicionadas, nem Pull Requests que quebrem os testes atuais.
Para executar a suíte de testes:
```bash
pytest tests/
```

## 5. Submetendo Pull Requests

1. Crie um fork do repositório.
2. Crie uma branch baseada no tipo de sua contribuição, ex: `feature/nova_camada`, `bugfix/fix-DDP`.
3. Verifique a formatação do código executando o Black (e Flake8, opcionalmente).
4. Submeta o PR apontando para a branch `main`.

Agradecemos o seu esforço para manter este repositório elegante, limpo e profissional!
