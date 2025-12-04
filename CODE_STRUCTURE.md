# Code Structure and File Descriptions

This document gives an overview of the repository structure and explains the purpose of the main files and folders. It is intended to make the code understandable and traceable for the master's thesis.

## Main Directories

### `agentic_system/`
Contains the implementation of the agent-based system.
- **`agents/`**: Definitions of the different agents (for example `TopicIdentificationAgent`, `AutonomousAgent`, `RAGAgent`).
- **`core/`**: Base classes and core logic for the agent system.
- **`tools/`**: Tools used by the agents (for example web search).

### `evaluation_system/`
Contains the framework for evaluating the AI systems ("LLM-as-a-Judge").
- **`evaluation.py`**: Core logic for evaluating answers.
- **`models.py`**: Data models for evaluations and metrics.
- **`agents/`**: Specialized agents for evaluation (for example `GDPRComplianceAgent`, `EUAIActAgent`).

### `rag_system/`
Contains the implementation of the classical RAG system (Retrieval-Augmented Generation) used for comparison.

### `Q&A evaluated/`
Contains the evaluation results and aggregated data.

## Main Execution Scripts

These scripts control the main evaluation processes and the comparison between systems.

- **`run_end_to_end_evaluation.py`**: Runs the complete evaluation pipeline, including the specialized regulatory agents. This is the main script for the final evaluation.
- **`run_evaluation_suite.py`**: Orchestrates a suite of evaluations across multiple datasets.
- **`run_llm_evaluation.py`**: Wrapper script to start the LLM-based evaluation with specific configurations.

## Setup and Data Preparation

Scripts for setting up the environment and preparing the data.

- **`create_chroma_db.py`**: Creates the ChromaDB vector database from the source documents for the RAG system.
- **`create_regulatory_chroma_dbs.py`**: Creates specific vector databases for regulatory documents (GDPR, EU AI Act).
- **`create_gifts_entertainment_chroma_db.py`**: Creates the database for the "Gifts and Entertainment" domain.
- **`create_it_governance_chroma_db.py`**: Creates the database for the "IT Governance" domain.
- **`process_excel_for_evaluation.py`**: Processes Excel files with questions and answers and prepares them for the evaluation framework.

## Configuration

Files for managing system configuration.

- **`eval_config_manager.py`**: CLI tool for creating and validating evaluation configurations.
- **`snippets_evaluation_config.py`**: Configuration file for evaluation with source snippets.
- **`snippets_evaluation_config.py`**: Contains profiles and settings for different evaluation scenarios.

## Tests and Debugging

Scripts for testing components and debugging.

- **`dynamic_pipeline_test.py`**: Interactive test for the dynamic routing pipeline of the agents.
- **`improved_dynamic_pipeline.py`**: Extended version of the pipeline test.
- **`debug_azure_evaluation.py`**: Tests the connection and evaluation with Azure OpenAI.
- **`debug_llm_response.py`**: Analyses raw LLM responses for debugging.
- **`simple_azure_test.py`**: Minimal connection test for Azure.
- **`test_audit_trail_agent.py`**: Unit test for the audit trail agent.
- **`test_eu_ai_act_agent.py`**: Unit test for the EU AI Act agent.
- **`test_gdpr_agent.py`**: Unit test for the GDPR agent.

## Utility Scripts

Small scripts for specific helper tasks.

- **`check_excel_structure.py`**: Checks whether Excel input files have the correct structure.
- **`fix_system_types.py`**: Fixes system-type classifications in result files after evaluation.
- **`simple_excel_evaluator.py`**: Simplified evaluator for quick tests without the full framework.

## Documentation

- **`README.md`**: Main project documentation.
- **`ACADEMIC_WEIGHT_UPDATE_SUMMARY.md`**: Documents changes to the academic weighting scheme.
- **`CODE_STRUCTURE.md`**: (This file) Overview of the file and folder structure.
