# Configuration file for evaluate_with_snippets.py
# Senior Data Scientist approach for easy file switching

# =============================================================================
# EVALUATION CONFIGURATION - EDIT THESE VALUES FOR DIFFERENT EVALUATIONS
# =============================================================================

# Current evaluation settings - CHANGE THESE FOR DIFFERENT FILES
CURRENT_SETTINGS = {
    "file": "newHQ",  # Change to: "it_governance", "gifts_entertainment", "newHQ", or add your own
    "description": "New HQ Q&A Evaluation",
    # NEW: Evaluation profile selection
    "evaluation_profile": "agentic_focused"  # Options: "rag_focused", "agentic_focused", "academic_research", "comparative_study"
}

# File configurations - ADD YOUR FILES HERE
EXCEL_CONFIGURATIONS = {
    "it_governance": {
        "file_path": "Q&A/Agentic q&a/it_governance.xlsx",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Database_Used",  # IT Governance ChromaDB
        "description": "IT Governance Q&A evaluation with database context",
        # Additional columns for enhanced analysis
        "metadata_cols": {
            "question_id": "Question_ID",
            "search_method": "Search_Method", 
            "status": "Status",
            "fallback_used": "Fallback_Used",
            "web_fallback_status": "Web_Fallback_Status",
            "error": "Error"
        }
    },
    "gifts_entertainment": {
        "file_path": "Q&A/Agentic q&a/gifts_entertainment.xlsx",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Database_Used",
        "description": "Gifts and Entertainment evaluation",
        "metadata_cols": {
            "question_id": "Question_ID",
            "search_method": "Search_Method",
            "status": "Status", 
            "fallback_used": "Fallback_Used",
            "web_fallback_status": "Web_Fallback_Status",
            "error": "Error"
        }
    },
    "newHQ": {
        "file_path": "Q&A/Agentic q&a/newHQ.xlsx",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Database_Used",
        "description": "New HQ evaluation",
        "metadata_cols": {
            "question_id": "Question_ID",
            "search_method": "Search_Method",
            "status": "Status", 
            "fallback_used": "Fallback_Used",
            "web_fallback_status": "Web_Fallback_Status",
            "error": "Error"
        }
    },
    # ADD YOUR OWN CONFIGURATIONS HERE
    "custom": {
        "file_path": "your_file.xlsx",
        "question_col": "Question",
        "answer_col": "Answer", 
        "snippet_col": "Snippet",
        "description": "Custom evaluation template",
        "metadata_cols": {}
    }
}

# Advanced evaluation options
EVALUATION_OPTIONS = {
    "include_snippet_grounding": True,
    "include_citation_analysis": True,
    "generate_detailed_reports": True,
    "save_intermediate_results": True,
    "verbose_logging": True,
    # New options based on your data structure
    "analyze_fallback_performance": True,  # Compare ChromaDB vs Web Search performance
    "track_search_method_effectiveness": True,  # Analyze which search methods work best
    "evaluate_error_patterns": True,  # Look for patterns in errors
    "assess_source_citation_quality": True,  # Check how well sources are cited
    "compare_database_vs_web_answers": True,  # Compare answers from different sources
}

# =============================================================================
# EVALUATION PROFILE CONFIGURATIONS
# =============================================================================

# Available evaluation profiles with descriptions
EVALUATION_PROFILES = {
    "rag_focused": {
        "name": "RAG-Focused Evaluation",
        "description": "Optimized for Retrieval-Augmented Generation systems",
        "advanced_config_profile": "RAG_FOCUSED",
        "emphasis": [
            "Citation Quality (15% weight)",
            "Factual Accuracy (30% weight)", 
            "Source Grounding",
            "Retrieval Effectiveness"
        ],
        "ideal_for": "Systems that retrieve documents and generate responses based on them"
    },
    "agentic_focused": {
        "name": "Agentic AI Evaluation", 
        "description": "Optimized for autonomous agentic AI systems",
        "advanced_config_profile": "AGENTIC_FOCUSED",
        "emphasis": [
            "Completeness (25% weight)",
            "Relevance (25% weight)",
            "Problem-solving capability",
            "Multi-step reasoning"
        ],
        "ideal_for": "Systems that plan, reason, and use tools autonomously"
    },
    "academic_research": {
        "name": "Academic Research Standard",
        "description": "Balanced academic evaluation with statistical rigor",
        "advanced_config_profile": "ACADEMIC_RESEARCH", 
        "emphasis": [
            "Statistical significance",
            "Inter-rater agreement",
            "Evidence-based scoring",
            "Confidence intervals"
        ],
        "ideal_for": "Research papers, academic studies, comparative analysis"
    },
    "comparative_study": {
        "name": "Comparative Study",
        "description": "Designed for comparing different AI systems",
        "advanced_config_profile": "COMPARATIVE_STUDY",
        "emphasis": [
            "Statistical significance (α=0.01)",
            "Larger sample sizes",
            "Confidence intervals", 
            "LaTeX table generation"
        ],
        "ideal_for": "Head-to-head system comparisons, benchmark studies"
    },
    "production_evaluation": {
        "name": "Production Evaluation",
        "description": "Production-ready evaluation with high confidence thresholds",
        "advanced_config_profile": "PRODUCTION_EVALUATION",
        "emphasis": [
            "High confidence threshold (0.8)",
            "Multiple retries (5x)",
            "Statistical analysis",
            "Robust error handling"
        ],
        "ideal_for": "Production system monitoring, quality assurance"
    }
}

# Evaluation criteria weights (customize based on your priorities)
EVALUATION_WEIGHTS = {
    "accuracy": 0.3,        # How factually correct is the answer?
    "relevance": 0.25,      # How well does it answer the question?
    "completeness": 0.2,    # Does it cover all aspects of the question?
    "citation_quality": 0.15,  # Are sources properly cited?
    "clarity": 0.1          # Is the answer clear and well-structured?
}

# Analysis categories for your specific use case
ANALYSIS_CATEGORIES = {
    "database_effectiveness": {
        "description": "How well does the ChromaDB perform vs web search",
        "metrics": ["success_rate", "answer_quality", "response_time"]
    },
    "fallback_analysis": {
        "description": "When and why does the system fall back to web search",
        "metrics": ["fallback_trigger_rate", "fallback_success_rate", "quality_comparison"]
    },
    "search_method_performance": {
        "description": "Effectiveness of different search methods",
        "metrics": ["topic_identification_success", "retrieval_accuracy"]
    },
    "source_grounding": {
        "description": "How well answers are grounded in retrieved sources",
        "metrics": ["citation_accuracy", "source_relevance", "hallucination_rate"]
    }
}

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
"""
To evaluate a different file:

1. Update CURRENT_SETTINGS["file"] to one of:
   - "it_governance" (default)
   - "gdpr"
   - "eu_ai_act" 
   - "test_snippets"
   - "custom" (update the file_path first)

2. If needed, add your own configuration to EXCEL_CONFIGURATIONS

3. Update column names if your Excel structure is different

4. Run: python evaluate_with_snippets.py

The script will automatically use the correct configuration and generate
reports with snippet grounding analysis!
"""