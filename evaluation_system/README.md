# Evaluation System: LLM-as-a-Judge Framework

> Academic evaluation framework for comparative analysis of RAG vs Agentic AI systems using LLM-based automated assessment.

## Research Context

This evaluation system implements an **LLM-as-a-Judge** methodology for assessing AI system responses. It serves as the core evaluation component for comparing Retrieval-Augmented Generation (RAG) systems against Agentic AI architectures in enterprise knowledge management contexts.

**Research Objective**: Provide reproducible, statistically rigorous evaluation of AI system outputs across multiple quality dimensions relevant to enterprise compliance and information retrieval.

---

## Architecture Overview

```
evaluation_system/
├── llm_client.py        # Multi-cloud LLM API abstraction (Azure OpenAI primary)
├── evaluation.py        # Core evaluation orchestration with snippet support
├── models.py            # Data structures (AnswerEvaluation, SourceSnippet, etc.)
├── enums.py             # Type definitions (SystemType, EvaluationDimension)
├── advanced_config.py   # Configuration profiles and rubric management
├── templates.py         # Jinja2-based evaluation prompt templates
├── statistics.py        # Statistical comparison (Cohen's d, t-tests)
├── excel_processor.py   # Excel I/O for Q&A datasets
├── report.py            # Academic report generation (LaTeX, CSV, JSON)
├── utils.py             # Helper functions (snippet parsing, validation)
└── semantic_similarity.py # Token overlap and embedding-based similarity
```

---

## Core Components

### 1. LLM Client (`llm_client.py`)

**Purpose**: Abstracts LLM API calls with multi-cloud support and enterprise authentication.

**Authentication Flow**:
```python
# Primary: Azure OpenAI with Service Principal (no API keys in code)
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET")
)
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    azure_ad_token_provider=token_provider,  # Token-based, not API key
    api_version=api_version
)
```

**Design Decision**: Service principal authentication enables secure deployment in enterprise environments without exposing API keys. Fallback to standard OpenAI API is available if Azure credentials are missing.

### 2. Evaluation Engine (`evaluation.py`)

**Purpose**: Orchestrates LLM-based evaluation of AI responses with source snippet analysis.

**Key Function**: `evaluate_answers_with_snippets()`
- Accepts `EvaluationInput` objects containing query, answer, and optional source snippets
- Calls LLM evaluator with structured prompt templates
- Calculates snippet grounding scores for RAG systems
- Returns `AnswerEvaluation` objects with scores, justifications, and confidence levels

**RAG-Specific Analysis**:
```python
# For RAG systems, calculate how well the answer is grounded in retrieved snippets
evaluation.snippet_grounding_score = calculate_snippet_grounding_score(
    eval_input.answer, 
    eval_input.source_snippets
)
```

### 3. Evaluation Dimensions (`enums.py`, `advanced_config.py`)

**5 Evaluation Dimensions** (academic weights normalized to ~1.0):

| Dimension           | Description                  |
|---------------------|------------------------------|
| Factual Accuracy    | Correctness of claims        |
| Relevance           | Query-response alignment     |
| Completeness        | Coverage of query aspects    |
| Clarity             | Organization and readability |
| Citation Quality    | Source attribution           |


### 4. Prompt Templates (`templates.py`)

**Three Template Variants**:

| Template | Use Case | Key Features |
|----------|----------|--------------|
| `STANDARD_EVALUATION_PROMPT` | Generic evaluation | Basic dimension scoring |
| `SNIPPET_EVALUATION_PROMPT` | RAG with snippets | Source grounding analysis |
| `ENHANCED_SNIPPET_EVALUATION_PROMPT` | Advanced RAG | Separation of content quality vs source utilization |

**Note**: Final evaluations in this research used the `ENHANCED_SNIPPET_EVALUATION_PROMPT` template.

**Enhanced Template Principles**:
- **Content Quality**: Factual accuracy evaluated independently of snippet support
- **Source Utilization**: Separate assessment of how well snippets are used
- **System-Aware Evaluation**: Does not penalize systems for architectural design choices (e.g., no explicit citations)
- **Dual Assessment**: Content dimensions scored on response quality; source dimensions on snippet grounding

**Template Structure** (Jinja2):
```jinja2
## EVALUATION FRAMEWORK
{% for dimension in dimensions %}
{{ loop.index }}. **{{ dimension.value }}**: {{ rubrics[dimension].description }}
{% endfor %}

## INPUT DATA
**User Query**: {{ query }}
**AI Response**: {{ answer }}
{% if snippets %}
**Source Snippets**: {{ snippets|length }} total
{% endif %}

## OUTPUT FORMAT (JSON)
{
    "scores": { ... },           // 1-5 per dimension
    "justifications": { ... },   // Evidence-based reasoning
    "confidence_scores": { ... }, // 0.0-1.0 per dimension
    "snippet_analysis": { ... }  // Grounding and citation assessment
}
```

### 5. Data Models (`models.py`)

**Core Data Structures**:

```python
@dataclass
class SourceSnippet:
    """Represents a retrieved document chunk from RAG system."""
    content: str
    snippet_id: Optional[str]
    source_document: Optional[str]
    relevance_score: Optional[float]

@dataclass
class EvaluationInput:
    """Input for evaluation with query, answer, and optional snippets."""
    query: str
    answer: str
    system_type: Optional[SystemType]
    source_snippets: Optional[List[SourceSnippet]]

@dataclass
class AnswerEvaluation:
    """Complete evaluation result with scores and metadata."""
    scores: Dict[EvaluationDimension, int]  # 1-5 scale
    justifications: Dict[EvaluationDimension, str]
    weighted_total: float
    snippet_grounding_score: Optional[float]  # RAG-specific
```

---

## Authentication & Environment Setup

### Required Environment Variables

```bash
# Azure Service Principal (primary authentication)
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-service-principal-client-id>
AZURE_CLIENT_SECRET=<your-service-principal-secret>

# Azure OpenAI Endpoint
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Fallback (optional): Standard OpenAI
OPENAI_API_KEY=<your-openai-key>
```

**Security Note**: This framework uses service principal authentication with Azure AD tokens rather than API keys, enabling secure deployment in enterprise environments with proper access control and audit logging.

---

## Usage

### Programmatic Evaluation

```python
from evaluation_system.config import EvalConfig
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.models import EvaluationInput, SourceSnippet
from evaluation_system.enums import SystemType

# Configure evaluation
config = EvalConfig()

# Prepare evaluation input (RAG system with snippets)
eval_input = EvaluationInput(
    query="What are the parking options at the new office?",
    answer="There are 115 internal parking spaces bookable via ParkHere app...",
    system_type=SystemType.RAG,
    source_snippets=[
        SourceSnippet(content="Parking spaces at Überseering...", snippet_id="1")
    ]
)

# Run evaluation
evaluations = evaluate_answers_with_snippets([eval_input], config)

# Access results
for eval_result in evaluations:
    print(f"Weighted Score: {eval_result.weighted_total:.2f}")
    print(f"Grounding Score: {eval_result.snippet_grounding_score:.3f}")
```

### Batch Evaluation from Excel

```python
from evaluation_system.excel_processor import load_excel_with_snippets
from evaluation_system.evaluation import evaluate_answers_with_snippets

# Load Q&A dataset
inputs = load_excel_with_snippets(
    "path/to/qa_dataset.xlsx",
    question_col="Question",
    answer_col="Answer",
    snippet_col="Snippets"
)

# Run evaluation suite
results = evaluate_answers_with_snippets(inputs, config)
```

### Statistical Comparison

```python
from evaluation_system.statistics import perform_statistical_comparison

# Separate evaluations by system type
rag_evals = [e for e in results if e.system_type == SystemType.RAG]
agentic_evals = [e for e in results if e.system_type == SystemType.AGENTIC]

# Perform statistical analysis
comparison = perform_statistical_comparison(rag_evals, agentic_evals)

print(f"Overall Winner: {comparison.overall_winner}")
print(f"95% CI: {comparison.confidence_interval}")
```

---

## Configuration Profiles

The `AdvancedEvalConfig` class supports pre-defined profiles for different evaluation contexts:

| Profile | Use Case | Key Settings |
|---------|----------|--------------|
| `ACADEMIC_RESEARCH` | Default for thesis research | Balanced weights, full statistical analysis |
| `RAG_FOCUSED` | RAG system evaluation | Higher citation quality weight (0.15) |
| `AGENTIC_FOCUSED` | Agentic system evaluation | Higher completeness/relevance weights |
| `COMPARATIVE_STUDY` | A/B comparison studies | Stricter significance level (α=0.01) |

---

## Output Formats

### Report Generation (`report.py`)

- **Academic Report**: Publication-ready text with proper citations (APA style)
- **LaTeX Tables**: Ready for inclusion in academic papers
- **CSV Export**: For external statistical analysis (SPSS, R)
- **JSON Export**: Programmatic access to full evaluation data

---

## Dependencies

Core dependencies (see `requirements.txt`):
```
openai>=1.0.0
azure-identity>=1.15.0
google-genai>=1.0.0
pandas>=2.0.0
scipy>=1.10.0
numpy>=1.24.0
jinja2>=3.1.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
```

---

## Research Methodology Notes

1. **LLM-as-a-Judge Approach**: Uses GPT-4o-mini as the evaluation model with structured JSON output format to ensure consistent, parseable responses.

2. **Snippet Grounding**: For RAG systems, the framework calculates how well generated answers are grounded in retrieved source snippets using token overlap and semantic similarity methods.

3. **Dimension Calibration**: Evaluation dimensions and weights are based on established IR/NLP evaluation literature (TREC, BLEU/ROUGE) augmented with regulatory compliance dimensions specific to enterprise AI deployment.

4. **Reproducibility**: All evaluation configurations, rubrics, and prompts are version-controlled. Evaluation metadata includes timestamps, model versions, and token usage for audit purposes.
