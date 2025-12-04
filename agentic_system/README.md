# Agentic System: Autonomous AI Agent Architecture

> Multi-agent system implementation for enterprise knowledge management with autonomous reasoning, tool use, and compliance awareness.

## Overview

This module implements the **Agentic AI** system being evaluated in the comparative study. Unlike traditional RAG systems that directly retrieve and present information, this system employs autonomous agents capable of planning, reasoning, and executing multi-step workflows.

**Key Differentiator**: The agentic approach allows for dynamic query routing, multi-source synthesis, and iterative refinement of responses.

---

## Architecture

```
agentic_system/
├── agents/
│   ├── autonomous_agent.py         # Planning and tool execution
│   ├── rag_agent.py                # ChromaDB knowledge retrieval
│   ├── compliance_agent.py         # Regulatory compliance checking
│   └── topic_identification_agent.py # Query routing
├── core/
│   └── base.py                     # BaseAgent, Task, AgentResponse
├── tools/
│   ├── web_search.py               # External search capability
│   ├── content_synthesizer.py      # Multi-source synthesis
│   ├── source_validator.py         # Credibility assessment
│   └── chromadb_client.py          # Vector database interface
├── audit/
│   └── audit_log.py                # Audit trail logging
└── enums.py                        # AgentType, TaskStatus, Priority
```

---

## Agent Types

### 1. AutonomousAgent (`agents/autonomous_agent.py`)

**Purpose**: Orchestrates complex tasks through planning, reasoning, and tool execution.

**Workflow**:
```
1. Receive Task → 2. Understand Task → 3. Create Plan → 4. Execute Plan → 5. Return Response
```

**Key Methods**:
```python
async def process(self, task: Task) -> AgentResponse:
    """
    Process a complex task by:
    1. Deconstructing and understanding the task
    2. Creating a step-by-step plan
    3. Executing each step with appropriate tools
    4. Synthesizing final response
    """

async def _execute_plan(self, task: Task) -> Any:
    """
    Execute plan steps sequentially:
    - web_search: Gather external information
    - validate_sources: Assess credibility
    - synthesize_content: Combine findings
    """
```

**Tools Available**:
- `WebSearchTool`: External web search for current information
- `ContentSynthesizer`: Combines multiple sources into coherent response
- `SourceValidator`: Assesses source credibility and trustworthiness

### 2. RAGAgent (`agents/rag_agent.py`)

**Purpose**: Retrieves information from ChromaDB vector databases with intelligent routing.

**Knowledge Bases**:
| Database | Content Domain | Collection |
|----------|---------------|------------|
| Confluence | Project documentation, AI/RPA | Default |
| NewHQ | Office facilities, parking | `berenberg_newhq_docs` |
| IT Governance | Policies, security procedures | `berenberg_it_governance` |
| Gifts & Entertainment | Compliance, ethics | `berenberg_gifts_entertainment` |

**Intelligent Routing**:
```python
# Topic-based database selection
topic_to_database_mapping = {
    'confluence': ['project', 'baia', 'begochat', 'rpa', 'ai', 'development'],
    'newhq': ['office', 'building', 'parking', 'facilities', 'workspace'],
    'it_governance': ['governance', 'policy', 'security', 'compliance', 'audit'],
    'gifts_entertainment': ['gifts', 'entertainment', 'corruption', 'ethics']
}
```

**LLM Integration**:
- Uses **Azure OpenAI** with `DefaultAzureCredential` for token-based authentication
- Embeddings via **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2`
- QA chains built with **LangChain** `RetrievalQA`

### 3. TopicIdentificationAgent (`agents/topic_identification_agent.py`)

**Purpose**: Analyzes queries to determine optimal database routing.

**Output**:
```python
{
    'topics': ['parking', 'office', 'facilities'],
    'metadata': {
        'domain': 'facilities',
        'confidence': 0.85
    }
}
```

---

## Core Abstractions (`core/base.py`)

### BaseAgent

Abstract base class for all agents:

```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.is_active = False
    
    @abstractmethod
    async def process(self, task: Task) -> AgentResponse:
        """Process a task and return a response."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the agent gracefully."""
        pass
```

### Task

Standard task representation:

```python
@dataclass
class Task:
    id: str
    name: str
    description: str
    status: TaskStatus  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    priority: Priority  # LOW, MEDIUM, HIGH, CRITICAL
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    assigned_agent: Optional[str]
```

### AgentResponse

Standard response format:

```python
@dataclass
class AgentResponse:
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str]
    confidence_score: float
    execution_time: float
    sources: List[str]
```

---

## Authentication

The agentic system uses **Azure AD authentication** via `DefaultAzureCredential`:

```python
from azure.identity import DefaultAzureCredential

# Automatic credential chain (managed identity, service principal, CLI, etc.)
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

# Set for Azure OpenAI client
os.environ["AZURE_OPENAI_API_KEY"] = token.token
```

This enables deployment in Azure environments with managed identity or local development with Azure CLI authentication.

---

## Audit Logging

All agent actions are logged for traceability:

```python
await audit_logger.log_agent_action(
    agent_id=self.agent_id,
    agent_type=self.agent_type,
    action="task_received",
    task=task,
    log_level=LogLevel.INFO
)
```

Logged events include:
- Agent initialization/shutdown
- Task processing start/completion/failure
- Tool invocations
- Database queries
- Error conditions

---

## Usage

### Standalone Agent Execution

```python
import asyncio
from agentic_system.agents.autonomous_agent import AutonomousAgent
from agentic_system.core.base import Task

async def main():
    # Create and initialize agent
    agent = AutonomousAgent(agent_id="agent-001")
    await agent.initialize()
    
    # Create task
    task = Task(
        id="task-123",
        name="Research Query",
        description="Search for information about parking options",
        input_data={"query": "Where can I park my car?"}
    )
    
    # Process task
    response = await agent.process(task)
    
    if response.success:
        print(f"Answer: {response.data.get('result')}")
        print(f"Sources: {response.sources}")
    
    await agent.shutdown()

asyncio.run(main())
```

### RAG Agent with Topic Routing

```python
from agentic_system.agents.rag_agent import RAGAgent
from agentic_system.core.base import Task, Priority

async def main():
    # Initialize RAG agent (loads ChromaDB connections)
    rag_agent = RAGAgent(agent_id="rag-001")
    await rag_agent.initialize()
    
    # Get knowledge base statistics
    stats = rag_agent.get_knowledge_base_stats()
    print(f"Confluence docs: {stats['confluence_db'].get('document_count')}")
    print(f"NewHQ docs: {stats['newhq_db'].get('document_count')}")
    
    # Process query with automatic routing
    task = Task(
        id="query-001",
        name="Parking Query",
        description="Get parking information",
        input_data={"query": "How do I book a parking space?"}
    )
    
    response = await rag_agent.process(task)
    print(f"Database used: {response.data.get('database_used')}")
    print(f"Answer: {response.data.get('answer')}")
    
    await rag_agent.shutdown()

asyncio.run(main())
```

---

## ChromaDB Configuration

Vector databases are stored at configured paths:

```python
# Database paths
confluence_db_path = ".../confluence_chroma_db"
newhq_db_path = ".../newHQ_chroma_db"
it_governance_db_path = ".../it_governance_chroma_db"
gifts_entertainment_db_path = ".../gifts_entertainment_chroma_db"

# RAG parameters
max_retrieval_results = 5
similarity_threshold = 0.7
max_context_length = 8000
temperature = 0  # Deterministic for evaluation
```

---

## Dependencies

```
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-huggingface>=0.0.1
langchain-chroma>=0.1.0
chromadb>=0.4.0
azure-identity>=1.15.0
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
```

---

## Design Decisions

1. **Async Architecture**: All agent operations are async to support concurrent task processing and non-blocking I/O.

2. **Topic-Based Routing**: Queries are analyzed to determine the most relevant knowledge base, reducing noise from irrelevant sources.

3. **Fallback Strategy**: If topic identification fails, agents fall back to keyword-based routing, ensuring robustness.

4. **Audit Trail**: Comprehensive logging enables post-hoc analysis of agent decisions, important for compliance evaluation.

5. **Modular Tools**: Tool classes (`WebSearchTool`, `ContentSynthesizer`, `SourceValidator`) are loosely coupled, enabling easy extension.
