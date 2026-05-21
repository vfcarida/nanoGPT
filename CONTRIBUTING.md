# Contribution Guide - nanoGPT Enterprise Edition

First of all, thank you for your interest in contributing to the Enterprise version of **nanoGPT**!

This repository aims to maintain a high standard of Software Engineering. Follow the steps below to submit your contributions.

## 1. Reporting Bugs or Suggesting Improvements

- Make sure the issue or feature hasn't been reported previously (search the issues).
- Create an issue detailing the context, how to reproduce the bug (if applicable), and what the expected behavior is.

## 2. Code Standards (Clean Code and SOLID)

- **Static Typing:** Use `typing` for 100% of function signatures (Type Hinting).
- **Docstrings:** Use the Google or NumPy standard to document the signature, return type, and logical description of all public methods and classes.
- **Single Responsibility Principle:** Avoid bloating files. If a function or class becomes too large and houses multiple logical responsibilities, refactor and modularize it.

## 3. Development Environment

Make sure you have an isolated environment (virtual environment) installed. We manage dependencies using `pyproject.toml`.
In the root directory, run:
```bash
pip install -e ".[dev]"
```

## 4. Tests (Pytest)

Pull Requests will not be accepted without proper test coverage for newly added logic, nor will Pull Requests that break existing tests.
To run the test suite:
```bash
pytest tests/
```

## 5. Submitting Pull Requests

1. Fork the repository.
2. Create a branch based on the type of your contribution, e.g., `feature/new_layer`, `bugfix/fix-DDP`.
3. Check code formatting by running Black (and optionally Flake8).
4. Submit the PR pointing to the `main` branch.

Thank you for your effort in keeping this repository elegant, clean, and professional!
