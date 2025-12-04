"""
================================================================================
RAG vs. Agentic AI - Evaluation Suite Orchestrator
================================================================================

Purpose:
    This script serves as the master orchestrator for running a comprehensive suite
    of evaluations for both RAG (Retrieval-Augmented Generation) and Agentic AI
    systems.

Methodology:
    This script employs a configuration-driven approach. All evaluation jobs are
    defined in the `EVALUATION_JOBS` data structure. The script iterates through
    these jobs, automatically discovering all relevant Excel files in the specified
    directories and executing the evaluation framework for each one with the
    appropriate profile and parameters.
"""

import os
import sys
from pathlib import Path

# Directly import the necessary components from your evaluation system
import sys
from evaluation_system.main import setup_logging
from evaluation_system.config import EvalConfig
from evaluation_system.excel_processor import load_excel_with_snippets, export_evaluation_results_to_excel
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.enums import SystemType

# ==============================================================================
# CONFIGURATION - Define all evaluation jobs here
# ==============================================================================
# This list defines the evaluation jobs to be run. Each dictionary represents
# a batch of evaluations with a specific profile.
#
# Now expanded to handle all your files with the appropriate profiles.
# ==============================================================================
EVALUATION_JOBS = [
    # RAG System Evaluations (rag_focused profile)
    {
        "name": "RAG System - Gifts & Entertainment",
        "profile": "rag_focused",
        "directory": "Q&A/gcp q&a",
        "file": "gifts_entertainment_questions_answered.xlsx",
        "system_type": "rag",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Snippets",
    },
    {
        "name": "RAG System - IT Governance",
        "profile": "rag_focused",
        "directory": "Q&A/gcp q&a",
        "file": "it_governance_questions_answered.xlsx",
        "system_type": "rag",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Snippets",
    },
    {
        "name": "RAG System - Input Questions v0",
        "profile": "rag_focused",
        "directory": "Q&A/gcp q&a",
        "file": "input_questions - v0.xlsx",
        "system_type": "rag",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Snippets",
    },
    {
        "name": "RAG System - Input Questions v1",
        "profile": "rag_focused",
        "directory": "Q&A/gcp q&a",
        "file": "input_questions - v1.xlsx",
        "system_type": "rag",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Snippets",
    },
    {
        "name": "RAG System - Input Questions v2",
        "profile": "rag_focused",
        "directory": "Q&A/gcp q&a",
        "file": "input_questions - v2.xlsx",
        "system_type": "rag",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Snippets",
    },
    {
        "name": "RAG System - Input Questions v3 (Topic Cluster)",
        "profile": "rag_focused",
        "directory": "Q&A/gcp q&a",
        "file": "input_questions - v3 (topic cluster).xlsx",
        "system_type": "rag",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Snippets",
    },
    
    # Agentic System Evaluations (agentic_focused profile)
    {
        "name": "Agentic System - Gifts & Entertainment",
        "profile": "agentic_focused",
        "directory": "Q&A/Agentic q&a",
        "file": "gifts_entertainment.xlsx",
        "system_type": "agentic",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Database_Used",
    },
    {
        "name": "Agentic System - IT Governance",
        "profile": "agentic_focused",
        "directory": "Q&A/Agentic q&a",
        "file": "it_governance.xlsx",
        "system_type": "agentic",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Database_Used",
    },
    {
        "name": "Agentic System - New HQ",
        "profile": "agentic_focused",
        "directory": "Q&A/Agentic q&a",
        "file": "newHQ.xlsx",
        "system_type": "agentic",
        "question_col": "Question",
        "answer_col": "Answer",
        "snippet_col": "Database_Used",
    },
]

# ==============================================================================
# ORCHESTRATION LOGIC - Do not modify below this line
# ==============================================================================

def run_evaluation(job: dict):
    """
    Executes the evaluation framework for a single file by directly calling
    the evaluation functions with proper system type assignment.
    """
    file_path = Path(job["directory"]) / job["file"]
    profile = job["profile"]
    system_type_str = job["system_type"]
    
    print(f"    ▶️  Running evaluation for: {file_path.name}")
    print(f"        Profile: {profile}")
    print(f"        System Type: {system_type_str}")

    try:
        # 1. Create a basic EvalConfig object
        config = EvalConfig()
        
        # 2. Set up logging for the evaluation run
        setup_logging()

        # 3. Load Excel data using the evaluation system's loader
        print(f"        Loading Excel data...")
        evaluation_inputs = load_excel_with_snippets(
            str(file_path),
            question_col=job["question_col"],
            answer_col=job["answer_col"],
            snippet_col=job["snippet_col"],
            system_type_col=None  # We'll set this manually
        )

        # 3a. Fallback answer merger: if internal answer is "I don't know." but a web fallback exists,
        # replace the answer with synthesized web fallback content so evaluation reflects actual retrieval.
        fallback_header = "--- Web Search Fallback ---"
        for eval_input in evaluation_inputs:
            original_answer = eval_input.answer
            lower_answer = original_answer.lower()
            if fallback_header.lower() in lower_answer and "i don't know" in lower_answer:
                # Split at header
                parts = original_answer.split(fallback_header, 1)
                internal_part = parts[0].strip()
                web_part = parts[1].strip() if len(parts) > 1 else ""
                # Extract the portion after the line "Based on web search results ..." if present
                synthesized = web_part
                for marker in ["Based on web search results", "Web search did not return relevant results"]:
                    idx = web_part.lower().find(marker.lower())
                    if idx != -1:
                        synthesized = web_part[idx:]
                        break
                # Clean enumerated source lines like "1. " etc to form a cohesive answer
                cleaned_lines = []
                for line in synthesized.splitlines():
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    # Skip pure source listing lines that look like enumerated references without sentences
                    if line_stripped[0:2].isdigit() and "http" not in line_stripped and len(line_stripped.split()) < 6:
                        continue
                    cleaned_lines.append(line_stripped)
                merged_answer = "\n".join(cleaned_lines).strip()
                if merged_answer:
                    eval_input.metadata = eval_input.metadata or {}
                    eval_input.metadata.update({
                        "fallback_used": True,
                        "fallback_merge_applied": True,
                        "original_internal_answer_length": len(internal_part),
                        "web_fallback_length": len(merged_answer),
                    })
                    eval_input.answer = merged_answer
                else:
                    # Mark fallback attempted but nothing usable
                    eval_input.metadata = eval_input.metadata or {}
                    eval_input.metadata.update({
                        "fallback_used": True,
                        "fallback_merge_applied": False,
                        "original_internal_answer_length": len(internal_part),
                        "web_fallback_length": 0,
                    })
            else:
                # No fallback scenario
                eval_input.metadata = eval_input.metadata or {}
                eval_input.metadata.setdefault("fallback_used", False)
        
        # 4. CRITICAL: Set the system type for each evaluation input
        system_type_enum = SystemType(system_type_str)
        for eval_input in evaluation_inputs:
            eval_input.system_type = system_type_enum
        
        print(f"        Loaded {len(evaluation_inputs)} questions")
        print(f"        Starting evaluations...")
        
        # 5. Run the evaluation with snippets
        evaluations = evaluate_answers_with_snippets(evaluation_inputs, config)
        
        if not evaluations:
            print(f"    ❌ No evaluations completed successfully")
            return 1
        
        # 6. Save results to Excel format
        output_dir = Path(f"{file_path.stem}_evaluation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Convert evaluations to dictionary format for Excel export
        evaluation_dicts = []
        for i, eval_result in enumerate(evaluations):
            eval_dict = {
                "question": evaluation_inputs[i].query,
                "answer": eval_result.answer,
                "system_type": eval_result.system_type.value if eval_result.system_type else system_type_str,
                "scores": {dim.value: score for dim, score in eval_result.scores.items()},
                "weighted_total": eval_result.weighted_total,
                "raw_total": eval_result.raw_total,
                "snippet_grounding_score": eval_result.snippet_grounding_score,
                "source_snippets": eval_result.source_snippets or [],
                "overall_assessment": eval_result.overall_assessment,
                # Export fallback metadata if present
                "fallback_used": eval_result.evaluation_metadata.get("fallback_used"),
                "fallback_merge_applied": eval_result.evaluation_metadata.get("fallback_merge_applied"),
                "original_internal_answer_length": eval_result.evaluation_metadata.get("original_internal_answer_length"),
                "web_fallback_length": eval_result.evaluation_metadata.get("web_fallback_length"),
            }
            evaluation_dicts.append(eval_dict)
        
        # Save Excel results
        excel_output_path = output_dir / f"{file_path.stem}_evaluation_results.xlsx"
        export_evaluation_results_to_excel(evaluation_dicts, str(excel_output_path))
        
        print(f"    ✅ Evaluation successful!")
        print(f"        Processed: {len(evaluations)} questions")
        print(f"        Results saved to: {excel_output_path}")
        
        return 0

    except Exception as e:
        print(f"    ❌ An unexpected error occurred during evaluation for {file_path.name}:")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main function to orchestrate the evaluation suite."""
    print("======================================================")
    print("  Starting Comprehensive Evaluation Suite             ")
    print("======================================================")
    
    if not EVALUATION_JOBS:
        print("\nNo evaluation jobs configured. Exiting.")
        return

    total_jobs = len(EVALUATION_JOBS)
    
    for i, job in enumerate(EVALUATION_JOBS):
        print(f"\n--- Starting Job {i+1}/{total_jobs}: {job['name']} ---")
        
        file_to_check = Path(job["directory"]) / job["file"]
        if not file_to_check.exists():
            print(f"    ❌ Error: File not found: {file_to_check}")
            print("    Skipping job.")
            continue
            
        run_evaluation(job)
            
    print("\n======================================================")
    print(f"  Evaluation Suite Completed.                     ")
    print("======================================================")


if __name__ == "__main__":
    # Add the current directory to sys.path to ensure modules can be found
    # This is a robust way to handle the ModuleNotFoundError
    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    main()
