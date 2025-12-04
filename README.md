# LLM-as-a-Judge: Comparative Evaluation of RAG vs Agentic AI Systems

> Master's Thesis Research Repository — Automated evaluation framework for comparing Retrieval-Augmented Generation and Agentic AI architectures in enterprise knowledge management.

## Research Overview

This repository contains the implementation artifacts for a Master's thesis investigating the comparative performance of **RAG (Retrieval-Augmented Generation)** systems versus **Agentic AI** systems in enterprise question-answering contexts. The research employs an **LLM-as-a-Judge** methodology where a language model (GPT-4o-mini) serves as an automated evaluator, scoring system responses across multiple quality dimensions.

### Research Questions
1. How do RAG and Agentic AI systems compare in factual accuracy, relevance, and completeness for enterprise knowledge retrieval?
2. What are the trade-offs between source grounding (RAG) and autonomous reasoning (Agentic) approaches?
3. How do regulatory compliance dimensions (GDPR, EU AI Act) factor into enterprise AI system evaluation?

---

## Repository Structure

```
llm-as-a-judge/
├── evaluation_system/      # Core LLM-as-a-Judge evaluation framework
│   ├── llm_client.py       # Azure OpenAI / OpenAI API abstraction
│   ├── evaluation.py       # Evaluation orchestration with snippet support
│   ├── statistics.py       # Statistical analysis (Cohen's d, t-tests)
│   ├── templates.py        # Jinja2 evaluation prompt templates
│   └── ...                 # See evaluation_system/README.md
│
├── agentic_system/         # Agentic AI implementation under evaluation
│   ├── agents/             # Agent implementations (RAG, Autonomous, Compliance)
│   ├── core/               # Base classes and agent registry
│   ├── tools/              # Web search, content synthesis, source validation
│   └── audit/              # Audit logging for traceability
│
├── rag_system/             # RAG system configuration (mirrors evaluation_system)
│
├── Q&A evaluated/          # Evaluation results and aggregated analysis
│   ├── agentic/            # Agentic system evaluation outputs
│   ├── rag/                # RAG system evaluation outputs
│   └── aggregated_results.ipynb  # Cross-system statistical comparison
│
├── run_evaluation_suite.py # Master orchestrator for batch evaluations
├── Reliability_Study.ipynb # Inter-rater reliability analysis
└── test/                   # Unit and integration tests
```

---

## System Components

### 1. Evaluation System (`evaluation_system/`)

The **LLM-as-a-Judge** framework that automatically evaluates AI system responses. See [evaluation_system/README.md](evaluation_system/README.md) for detailed documentation.

**Key Capabilities**:
- Multi-dimensional scoring (8 dimensions with academic weights)
- Source snippet grounding analysis for RAG systems
- Statistical comparison between system types
- Multiple output formats (JSON, CSV, LaTeX, academic reports)

### 2. Agentic System (`agentic_system/`)

The **Agentic AI** system being evaluated. Implements autonomous reasoning with tool use.

**Architecture**:
```
agentic_system/
├── agents/
│   ├── autonomous_agent.py     # Reasoning, planning, and tool execution
│   ├── rag_agent.py            # ChromaDB-based knowledge retrieval
│   ├── compliance_agent.py     # Regulatory compliance checking
│   └── topic_identification_agent.py  # Query routing
├── core/
│   └── base.py                 # BaseAgent, Task, AgentResponse abstractions
└── tools/
    ├── web_search.py           # External web search capability
    ├── content_synthesizer.py  # Multi-source content synthesis
    └── source_validator.py     # Source credibility assessment
```

**Agent Types**:
- **AutonomousAgent**: Plans multi-step workflows, executes tools, synthesizes results
- **RAGAgent**: Searches ChromaDB vector databases (Confluence, NewHQ, IT Governance, Gifts & Entertainment)
- **TopicIdentificationAgent**: Routes queries to appropriate knowledge bases

### 3. RAG System (`rag_system/`)

Configuration and utilities for the GCP-hosted RAG system being compared against the Agentic system. Uses similar evaluation interfaces as `evaluation_system/`.

---

## Authentication Architecture

This framework uses **service account authentication** rather than API keys, enabling secure enterprise deployment.

### Azure OpenAI (Primary)
```python
from azure.identity import ClientSecretCredential, get_bearer_token_provider

# Service Principal credentials from environment
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET")
)

# Token-based authentication (no API key exposure)
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview"
)
```

### Environment Variables

```bash
# Azure Service Principal
AZURE_TENANT_ID=<tenant-id>
AZURE_CLIENT_ID=<service-principal-client-id>
AZURE_CLIENT_SECRET=<service-principal-secret>

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Fallback (optional)
OPENAI_API_KEY=<openai-api-key>
```

---

## Evaluation Methodology

### Dimensions & Weights

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Factual Accuracy | 0.25 | Correctness of claims against authoritative sources |
| Relevance | 0.20 | Query-response alignment |
| Completeness | 0.15 | Coverage of query aspects |
| Clarity | 0.13 | Organization and readability |
| Citation Quality | 0.07 | Source attribution quality |
| GDPR Compliance | 0.12 | Data protection adherence |
| EU AI Act Alignment | 0.12 | AI regulatory compliance |
| Audit Trail Quality | 0.08 | Decision traceability |

### Scoring Scale (1-5)

| Score | Label | Description |
|-------|-------|-------------|
| 5 | Excellent | Exceeds academic standards (90-100%) |
| 4 | Good | Meets high academic standards (80-89%) |
| 3 | Satisfactory | Acceptable academic quality (70-79%) |
| 2 | Needs Improvement | Below academic standards (60-69%) |
| 1 | Inadequate | Significantly deficient (<60%) |

### Statistical Analysis

- **Effect Size**: Cohen's d for practical significance
- **Significance Testing**: Independent t-tests per dimension
- **Confidence Intervals**: 95% CI for mean differences
- **Inter-rater Reliability**: Cohen's kappa for validation studies

---

## Running Evaluations

### Full Evaluation Suite

```bash
# Run all configured evaluation jobs
python run_evaluation_suite.py
```

This script processes all Q&A datasets defined in `EVALUATION_JOBS`:
- RAG system evaluations (GCP-based)
- Agentic system evaluations

### Single Evaluation

```python
from evaluation_system.config import EvalConfig
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.models import EvaluationInput
from evaluation_system.enums import SystemType

config = EvalConfig()

# Create evaluation input
eval_input = EvaluationInput(
    query="What are the parking options?",
    answer="There are 115 internal parking spaces...",
    system_type=SystemType.RAG,
    source_snippets=[...]
)

# Run evaluation
results = evaluate_answers_with_snippets([eval_input], config)
```

### Statistical Comparison

```python
from evaluation_system.statistics import perform_statistical_comparison

comparison = perform_statistical_comparison(rag_evaluations, agentic_evaluations)
print(f"Winner: {comparison.overall_winner}")
print(f"Effect sizes: {comparison.effect_sizes}")
```

---

## Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│  Q&A Dataset    │     │  Q&A Dataset    │
│  (RAG System)   │     │  (Agentic)      │
│  Excel Files    │     │  Excel Files    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│       run_evaluation_suite.py           │
│  (Batch orchestration with profiles)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         evaluation_system/              │
│  ┌─────────────────────────────────┐    │
│  │  excel_processor.py             │    │
│  │  (Load Q&A + snippets)          │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  evaluation.py                  │    │
│  │  (Orchestrate LLM evaluation)   │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  llm_client.py                  │    │
│  │  (Azure OpenAI API calls)       │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  statistics.py                  │    │
│  │  (Cohen's d, t-tests, CI)       │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  report.py                      │    │
│  │  (Academic, LaTeX, CSV, JSON)   │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Q&A evaluated/                  │
│  - Evaluation results (Excel)           │
│  - aggregated_results.ipynb             │
│  - Statistical comparison               │
└─────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **Service Principal Authentication**: Enterprise security requirement; no API keys in code or environment files.

2. **Multi-Dimensional Evaluation**: Eight dimensions cover both information retrieval quality (TREC-inspired) and regulatory compliance (EU-specific).

3. **Snippet Grounding Analysis**: RAG systems are evaluated on how well answers are grounded in retrieved source snippets, not just overall quality.

4. **Profile-Based Configuration**: Different evaluation profiles (academic research, RAG-focused, agentic-focused) allow fair comparison by adjusting dimension weights.

5. **JSON Output Format**: LLM evaluator uses structured JSON output mode (`response_format={"type": "json_object"}`) for reliable parsing.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `openai>=1.0.0` - OpenAI/Azure OpenAI client
- `azure-identity>=1.15.0` - Azure authentication
- `langchain>=0.1.0` - RAG agent orchestration
- `chromadb>=0.4.0` - Vector database
- `pandas>=2.0.0` - Data processing
- `scipy>=1.10.0` - Statistical analysis
- `sentence-transformers>=2.2.0` - Embeddings

---

## Reproducibility

All experiments can be reproduced by:

1. Setting up environment variables (see Authentication Architecture)
2. Placing Q&A datasets in appropriate directories
3. Running `python run_evaluation_suite.py`
4. Analyzing results in `Q&A evaluated/aggregated_results.ipynb`

Evaluation metadata (timestamps, model versions, token counts) is logged for audit purposes.

---

## Academic Citation

If using this framework for research, please cite:

```bibtex
@mastersthesis{fuhrmann2025llmjudge,
  title={Comparative Evaluation of RAG vs Agentic AI Systems Using LLM-as-a-Judge Methodology},
  author={Fuhrmann, [Name]},
  year={2025},
  school={[University]},
  type={Master's Thesis}
}
```

---

## License

Academic research use. See LICENSE file for details.
