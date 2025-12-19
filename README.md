# LLM-as-a-Judge Evaluation Framework

Automated evaluation framework for comparing RAG and Agentic AI system outputs using LLM-based judges, developed as part of a Master's thesis at Universität Münster / Professional School Münster.

## Repository Structure

```
llm-as-a-judge/
├── run_evaluation_suite.py        # Main entry point: batch evaluation
├── run_end_to_end_evaluation.py   # Main entry point: full pipeline
├── run_llm_evaluation.py          # Main entry point: single evaluation
├── requirements.txt               # Python dependencies
│
├── evaluation_system/             # Core LLM-as-a-Judge framework
│   ├── evaluation.py              # Evaluation orchestration
│   ├── llm_client.py              # Multi-provider LLM client (Azure, Gemini)
│   ├── advanced_config.py         # Provider configuration management
│   ├── models.py                  # Data models
│   ├── statistics.py              # Statistical analysis (Cohen's d, t-tests)
│   └── agents/                    # Regulatory agents (GDPR, EU AI Act, Audit)
│
├── agentic_system/                # Agentic AI implementation
│   ├── agents/                    # Agent implementations
│   ├── core/                      # Base classes
│   ├── tools/                     # Web search, content synthesis
│   └── audit/                     # Audit trail logging
│
├── rag_system/                    # RAG system configuration
│
├── Statistical Evaluation/        # Reliability analysis notebooks
│   ├── Reliability_Study_v2.ipynb # Inter-rater reliability analysis
│   ├── pairwise_study.ipynb       # Pairwise preference analysis
│   └── aggregated_results.xlsx    # Evaluation results (Q&A redacted)
│
├── scripts/                       # Setup and utility scripts
│   ├── create_chroma_db.py        # Vector database creation
│   ├── create_*_chroma_db.py      # Domain-specific databases
│   ├── process_excel_for_evaluation.py
│   └── ...                        # Other utilities
│
├── test/                          # Unit and integration tests
│
└── debug/                         # Debug and troubleshooting scripts
```

## Installation

### Prerequisites

- Python 3.10 or higher
- One of the following LLM providers:
  - Azure OpenAI service with service principal authentication
  - Google Cloud project with Vertex AI enabled

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

Create a `.env` file in the project root. Configure either Azure OpenAI or Google Vertex AI:

**Option 1: Azure OpenAI**

```bash
# LLM Provider Selection
LLM_PROVIDER=AZURE

# Azure Service Principal Authentication
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-service-principal-client-id>
AZURE_CLIENT_SECRET=<your-service-principal-secret>

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

**Option 2: Google Vertex AI (Gemini)**

```bash
# LLM Provider Selection
LLM_PROVIDER=GEMINI

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=<your-project-id>
GOOGLE_CLOUD_LOCATION=europe-central2
GOOGLE_GENAI_USE_VERTEXAI=True
```

Before using Vertex AI, authenticate with Google Cloud:

```bash
gcloud auth application-default login
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
from evaluation_system.advanced_config import create_default_config
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.models import EvaluationInput, SourceSnippet
from evaluation_system.enums import SystemType, LLMProvider

# Configure evaluation with specific provider
config = create_default_config()
config.llm_provider = LLMProvider.GEMINI  # or LLMProvider.AZURE

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
python scripts/create_chroma_db.py
python scripts/create_it_governance_chroma_db.py
python scripts/create_gifts_entertainment_chroma_db.py
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

## LLM Provider Configuration

The framework supports multiple LLM providers through a modular client system.

### Supported Providers

| Provider | Model | Authentication |
|----------|-------|----------------|
| Azure OpenAI | GPT-4o-mini | Service Principal |
| Google Vertex AI | Gemini 2.0 Flash | Application Default Credentials |

### Switching Providers

**Via environment variable:**

```bash
export LLM_PROVIDER=GEMINI  # or AZURE
```

**Programmatically:**

```python
from evaluation_system.advanced_config import create_default_config
from evaluation_system.enums import LLMProvider

config = create_default_config()
config.llm_provider = LLMProvider.GEMINI
```

### Google Vertex AI Setup

1. Install the Google Cloud SDK
2. Authenticate with your Google Cloud account:
   ```bash
   gcloud auth application-default login
   ```
3. Set the required environment variables:
   ```bash
   export GOOGLE_CLOUD_PROJECT=<your-project-id>
   export GOOGLE_CLOUD_LOCATION=europe-central2
   export GOOGLE_GENAI_USE_VERTEXAI=True
   ```

### Azure OpenAI Setup

The framework uses Azure AD service principal authentication:

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
