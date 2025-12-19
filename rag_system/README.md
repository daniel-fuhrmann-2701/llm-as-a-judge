# RAG System Module

Alternative import path for the evaluation framework, mirroring the `evaluation_system` module structure.

## Overview

This module provides the same evaluation functionality as `evaluation_system/` but under a different namespace. It was created to support legacy import paths and alternative package naming conventions.

## Module Structure

```
rag_system/
├── __init__.py           # Package exports
├── config.py             # Configuration and logging setup
├── enums.py              # SystemType, EvaluationDimension
├── models.py             # Data models (AnswerEvaluation, EvaluationRubric)
├── evaluation.py         # Core evaluation logic
├── llm_client.py         # LLM API client
├── statistics.py         # Statistical analysis
├── templates.py          # Jinja2 prompt templates
├── excel_processor.py    # Excel I/O utilities
├── report.py             # Report generation
├── utils.py              # Utility functions
└── main.py               # CLI interface
```

## Usage

```python
from rag_system import (
    EvalConfig,
    evaluate_answers,
    perform_statistical_comparison,
    SystemType,
    EvaluationDimension
)

# Configure and run evaluation
config = EvalConfig()
results = evaluate_answers(query, answers, config)
```

## Note

For new implementations, prefer using the `evaluation_system` module directly, as it contains the most up-to-date code and documentation.

See [evaluation_system/README.md](../evaluation_system/README.md) for detailed documentation.
