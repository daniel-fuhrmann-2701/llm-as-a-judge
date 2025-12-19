# LLM-as-a-Judge Evaluation Framework

Automated evaluation framework for comparing RAG and Agentic AI system outputs using LLM-based judges, developed as part of a Master's thesis at Universität Münster / Professional School Münster.

## Repository Structure

```
llm-as-a-judge/
├── evaluation_system/          # Core LLM-as-a-Judge framework
│   ├── evaluation.py           # Evaluation orchestration
│   ├── llm_client.py           # Azure OpenAI / OpenAI API client
│   ├── models.py               # Data models (EvaluationInput, AnswerEvaluation)
│   ├── enums.py                # SystemType, EvaluationDimension enums
│   ├── templates.py            # Jinja2 evaluation prompts
│   ├── statistics.py           # Statistical analysis (Cohen's d, t-tests)
│   ├── excel_processor.py      # Excel I/O utilities
│   └── agents/                 # Specialized evaluation agents (GDPR, EU AI Act)
│
├── agentic_system/             # Agentic AI implementation
│   ├── agents/                 # Agent implementations
│   │   ├── autonomous_agent.py # Planning and tool execution
│   │   ├── rag_agent.py        # ChromaDB knowledge retrieval
│   │   └── topic_identification_agent.py
│   ├── core/                   # Base classes (BaseAgent, Task, AgentResponse)
│   ├── tools/                  # Web search, content synthesis, validation
│   └── audit/                  # Audit trail logging
│
├── rag_system/                 # RAG system configuration
│
├── Statistical Evaluation/     # Reliability analysis
│   ├── Reliability_Study.ipynb       # Inter-rater reliability analysis
│   ├── Reliability_Study_v2.ipynb    # Updated analysis
│   ├── pairwise_study.ipynb          # Pairwise preference analysis
│   └── aggregated_results.xlsx       # Evaluation results (Q&A redacted)
│
├── test/                       # Unit and integration tests
│
├── run_evaluation_suite.py     # Batch evaluation orchestrator
├── run_end_to_end_evaluation.py # Full pipeline execution
├── create_chroma_db.py         # Vector database creation
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Azure subscription with OpenAI service (or OpenAI API key)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd llm-as-a-judge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Azure Service Principal Authentication (recommended)
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-service-principal-client-id>
AZURE_CLIENT_SECRET=<your-service-principal-secret>

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Alternative: Standard OpenAI API
OPENAI_API_KEY=<your-openai-key>
```

## Usage

### Running Evaluations

```bash
# Run the full evaluation suite
python run_evaluation_suite.py

# Run end-to-end evaluation with regulatory agents
python run_end_to_end_evaluation.py
```

### Programmatic Usage

```python
from evaluation_system.config import EvalConfig
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.models import EvaluationInput, SourceSnippet
from evaluation_system.enums import SystemType

# Configure evaluation
config = EvalConfig()

# Create evaluation input
eval_input = EvaluationInput(
    query="What are the parking options?",
    answer="There are 115 internal parking spaces...",
    system_type=SystemType.RAG,
    source_snippets=[
        SourceSnippet(content="Parking information...", snippet_id="1")
    ]
)

# Run evaluation
results = evaluate_answers_with_snippets([eval_input], config)

for result in results:
    print(f"Weighted Score: {result.weighted_total:.2f}")
```

### Statistical Analysis

The `Statistical Evaluation/` folder contains Jupyter notebooks for reliability analysis:

```bash
# Start Jupyter
jupyter notebook "Statistical Evaluation/"
```

Key notebooks:
- `Reliability_Study_v2.ipynb` - Inter-rater reliability metrics (Spearman's rho, Weighted Kappa)
- `pairwise_study.ipynb` - Pairwise preference agreement analysis

### Creating Vector Databases

```bash
# Create ChromaDB for specific domains
python create_chroma_db.py
python create_it_governance_chroma_db.py
python create_gifts_entertainment_chroma_db.py
```

## Evaluation Dimensions

The framework evaluates responses across 8 dimensions:

| Dimension           | Description                  |
|---------------------|------------------------------|
| Factual Accuracy    | Correctness of claims        |
| Relevance           | Query-response alignment     |
| Completeness        | Coverage of query aspects    |
| Clarity             | Organization and readability |
| Citation Quality    | Source attribution           |

## Data Availability

The evaluation datasets contain proprietary Q&A content that has been redacted. The `Question` and `Answer` columns in the Excel files are not included in this repository.

**For thesis evaluation purposes**: The complete datasets can be provided to the main examiner or members of the evaluation committee upon request.

## Project Structure Details

### evaluation_system/

Core framework for LLM-based evaluation:
- Multi-dimensional scoring (1-5 scale)
- Source snippet grounding analysis for RAG systems
- Statistical comparison utilities
- Multiple output formats (JSON, CSV, Excel)

### agentic_system/

Autonomous agent implementation:
- `AutonomousAgent`: Multi-step planning and tool execution
- `RAGAgent`: ChromaDB-based knowledge retrieval with topic routing
- `TopicIdentificationAgent`: Query classification for database selection
- Audit logging for traceability

### Statistical Evaluation/

Jupyter notebooks for analyzing inter-rater reliability between LLM judges (GPT-4o-mini and Gemini Flash).

## Authentication

The framework uses Azure AD service principal authentication for enterprise deployment:

```python
from azure.identity import ClientSecretCredential, get_bearer_token_provider

credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET")
)

token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default"
)
```

## Author

Daniel Fuhrmann
Master's Thesis
Universität Münster / Professional School Münster
2025
