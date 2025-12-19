#!/usr/bin/env python3
"""
Fix System Type Classification in Existing Results

This script reads existing evaluation results and applies proper system type 
classification without re-running the expensive Azure OpenAI evaluation.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def classify_system_type(answer, has_snippets=True):
    """Classify system type based on answer content."""
    rag_indicators = 0
    agentic_indicators = 0
    
    # RAG indicators
    if 'Sources:' in answer or '#' in answer:
        rag_indicators += 2
    if 'Based on my search' in answer or 'database' in answer.lower():
        rag_indicators += 2
    if 'According to' in answer or 'search of the' in answer:
        rag_indicators += 1
    if has_snippets:
        rag_indicators += 1
    
    # Agentic indicators
    if 'Web Search' in answer or 'web search' in answer.lower():
        agentic_indicators += 2
    if 'I don\'t know' in answer or 'cannot find' in answer.lower():
        agentic_indicators += 1  # Uncertainty suggests autonomous reasoning
    if len(answer) > 500:  # Longer responses suggest more reasoning
        agentic_indicators += 1
    if any(phrase in answer.lower() for phrase in ['i recommend', 'you should', 'best approach', 'steps to take']):
        agentic_indicators += 1
    
    # Classify
    if rag_indicators > agentic_indicators:
        return "rag_system"
    elif agentic_indicators > rag_indicators:
        return "agentic_system"
    else:
        return "hybrid_system"

def fix_csv_system_types(csv_path, force_type=None, excel_file=None):
    """Fix system types in CSV file."""
    print(f"Processing CSV: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    if force_type:
        print(f"Forcing all items to be classified as: {force_type}")
        system_types = [force_type] * len(df)
    else:
        # Auto-detect Excel file if not provided
        if excel_file is None:
            # Try to determine from directory name
            dir_name = csv_path.parent.name
            if "it_governance" in dir_name:
                excel_file = "Q&A/Agentic q&a/it_governance.xlsx"
            elif "gifts_entertainment" in dir_name:
                excel_file = "Q&A/Agentic q&a/gifts_entertainment.xlsx"
            elif "newHQ" in dir_name:
                excel_file = "Q&A/Agentic q&a/newHQ.xlsx"
            else:
                print(f"Cannot auto-detect Excel file for directory: {dir_name}")
                print("Please specify the Excel file manually")
                return
        
        # Read the original Excel to get the answers
        excel_path = Path(__file__).parent / excel_file
        if not excel_path.exists():
            print(f"Error: Excel file not found: {excel_path}")
            return
        
        print(f"Using Excel file: {excel_path}")
        excel_df = pd.read_excel(excel_path)
        
        # Apply classification
        system_types = []
        for i, row in df.iterrows():
            if i < len(excel_df):
                answer = str(excel_df.iloc[i]['Answer'])
                snippet_count = row.get('Snippet_Count', 0)
                system_type = classify_system_type(answer, snippet_count > 0)
                system_types.append(system_type)
                
                if i < 5:  # Debug first few
                    print(f"Item {i+1}: {system_type}")
            else:
                system_types.append("unknown")
    
    # Update the DataFrame
    df['System_Type'] = system_types
    
    # Save the updated CSV
    if force_type:
        output_path = csv_path.parent / f"{csv_path.stem}_{force_type}{csv_path.suffix}"
    else:
        output_path = csv_path.parent / f"{csv_path.stem}_fixed{csv_path.suffix}"
    df.to_csv(output_path, index=False)
    print(f"Fixed CSV saved to: {output_path}")
    
    # Print summary
    type_counts = pd.Series(system_types).value_counts()
    print(f"\nSystem Type Classification Summary:")
    for sys_type, count in type_counts.items():
        print(f"  {sys_type}: {count} ({count/len(system_types)*100:.1f}%)")

def fix_json_system_types(json_path, force_type=None):
    """Fix system types in JSON file."""
    print(f"Processing JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Fix system types in detailed results
    for item in data.get('detailed_results', []):
        if force_type:
            system_type = force_type
        else:
            answer = item.get('answer', '')
            snippet_count = item.get('snippet_count', 0)
            system_type = classify_system_type(answer, snippet_count > 0)
        item['system_type'] = system_type
    
    # Save the updated JSON
    if force_type:
        output_path = json_path.parent / f"{json_path.stem}_{force_type}{json_path.suffix}"
    else:
        output_path = json_path.parent / f"{json_path.stem}_fixed{json_path.suffix}"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed JSON saved to: {output_path}")

def main():
    """Main function to fix system types in existing results."""
    
    # Check command line arguments
    force_type = None
    target_dir = None
    
    # Parse arguments
    args = sys.argv[1:]
    
    for arg in args:
        if arg.lower() in ['agentic_system', 'rag_system', 'hybrid_system']:
            force_type = arg.lower()
        elif arg.endswith('_evaluation_results_') or 'evaluation_results' in arg:
            target_dir = arg
    
    if force_type:
        print(f"🎯 Forcing all items to be classified as: {force_type}")
    else:
        print("🔍 Using automatic classification based on content")
    
    # Find evaluation results directories
    if target_dir:
        # User specified a specific directory
        if not target_dir.startswith('\\'):
            # If it's just the directory name, find it
            results_dirs = list(Path(__file__).parent.glob(f"*{target_dir}*"))
        else:
            # Full path provided
            results_dirs = [Path(target_dir)]
    else:
        # Find all evaluation directories
        patterns = [
            "*_evaluation_results_*",
            "it_governance_evaluation_results_*",
            "gifts_entertainment_evaluation_results_*", 
            "newHQ_evaluation_results_*"
        ]
        results_dirs = []
        for pattern in patterns:
            results_dirs.extend(Path(__file__).parent.glob(pattern))
    
    if not results_dirs:
        print("No evaluation results directories found!")
        print("Available patterns:")
        print("  - it_governance_evaluation_results_*")
        print("  - gifts_entertainment_evaluation_results_*")
        print("  - newHQ_evaluation_results_*")
        return
    
    # Get the most recent directory if multiple found
    if len(results_dirs) > 1:
        if target_dir:
            print(f"Multiple directories match '{target_dir}':")
            for i, d in enumerate(results_dirs):
                print(f"  {i+1}. {d.name}")
            latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Using most recent: {latest_dir.name}")
        else:
            latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    else:
        latest_dir = results_dirs[0]
    
    print(f"Processing results directory: {latest_dir}")
    
    # Detect evaluation type from directory name
    dir_name = latest_dir.name
    if "gifts_entertainment" in dir_name:
        eval_type = "gifts_entertainment"
    elif "it_governance" in dir_name:
        eval_type = "it_governance"
    elif "newHQ" in dir_name:
        eval_type = "newHQ"
    else:
        eval_type = "unknown"
    
    print(f"Detected evaluation type: {eval_type}")
    
    # Fix CSV file
    csv_path = latest_dir / "evaluation_summary.csv"
    if csv_path.exists():
        fix_csv_system_types(csv_path, force_type)
    else:
        print(f"CSV file not found: {csv_path}")
    
    # Fix JSON file
    json_path = latest_dir / "detailed_results.json"
    if json_path.exists():
        fix_json_system_types(json_path, force_type)
    else:
        print(f"JSON file not found: {json_path}")

if __name__ == "__main__":
    main()