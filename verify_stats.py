import pandas as pd
import numpy as np

file_path = r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\Q&A evaluated\aggregated_results.xlsx"

sheet_patterns = {
    # OpenAI (OAI) sheets
    'OAI Agentic G+E': {'System_Type': 'Agentic', 'UseCase': 'Gifts_Entertainment'},
    'OAI Agentic IT-G': {'System_Type': 'Agentic', 'UseCase': 'IT_Governance'},
    'OAI Agentic NHQ': {'System_Type': 'Agentic', 'UseCase': 'New_HQ'},
    'OAI RAG G+E': {'System_Type': 'RAG', 'UseCase': 'Gifts_Entertainment'},
    'OAI RAG IT-G': {'System_Type': 'RAG', 'UseCase': 'IT_Governance'},
    'OAI RAG NHQ v0': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    'OAI RAG NHQ v1': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    'OAI RAG NHQ v2': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    'OAI RAG NHQ v3': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    # Gemini sheets
    'Gemini Agentic G+E': {'System_Type': 'Agentic', 'UseCase': 'Gifts_Entertainment'},
    'Gemini Agentic IT-G': {'System_Type': 'Agentic', 'UseCase': 'IT_Governance'},
    'Gemini Agentic NHQ': {'System_Type': 'Agentic', 'UseCase': 'New_HQ'},
    'Gemini RAG G+E': {'System_Type': 'RAG', 'UseCase': 'Gifts_Entertainment'},
    'Gemini RAG IT-G': {'System_Type': 'RAG', 'UseCase': 'IT_Governance'},
    'Gemini RAG NHQ v0': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    'Gemini RAG NHQ v1': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    'Gemini RAG NHQ v2': {'System_Type': 'RAG', 'UseCase': 'New_HQ'},
    'Gemini RAG NHQ v3': {'System_Type': 'RAG', 'UseCase': 'New_HQ'}
}

try:
    xls = pd.ExcelFile(file_path)
    processed_data = []

    for sheet_name in xls.sheet_names:
        matched_pattern = None
        for pattern_key, metadata in sheet_patterns.items():
            if pattern_key in sheet_name:
                matched_pattern = metadata
                break
        
        if not matched_pattern:
            continue
            
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Assign judge and metadata
        df['Judge'] = 'OpenAI' if 'OAI' in sheet_name else 'Gemini'
        df['Sheet'] = sheet_name # Keep track of sheet name
        for key, value in matched_pattern.items():
            df[key] = value
        
        processed_data.append(df)

    full_dataset = pd.concat(processed_data, ignore_index=True)
    
    # Filter for Balanced Subset
    # Exclude RAG variants v0, v1, v2
    # Keep Agentic (all)
    # Keep RAG G+E, RAG IT-G, RAG NHQ v3
    
    def is_balanced(row):
        if row['System_Type'] == 'Agentic':
            return True
        if row['System_Type'] == 'RAG':
            sheet = row['Sheet']
            if 'v0' in sheet or 'v1' in sheet or 'v2' in sheet:
                return False
            return True
        return False
        
    balanced_df = full_dataset[full_dataset.apply(is_balanced, axis=1)]

    # Filter for Relevance
    # Note: Column name might be 'Relevance' or similar. The notebook used 'Relevance'.
    if 'Relevance' in balanced_df.columns:
        # Ensure numeric
        balanced_df['Relevance'] = pd.to_numeric(balanced_df['Relevance'], errors='coerce')
        
        # Group by System_Type and calculate mean/std
        stats = balanced_df.groupby('System_Type')['Relevance'].agg(['mean', 'std', 'count'])
        print("Relevance Stats by System Type (Balanced Subset):")
        print(stats)
        
        # Also group by Judge and System_Type
        stats_judge = balanced_df.groupby(['Judge', 'System_Type'])['Relevance'].agg(['mean', 'std', 'count'])
        print("\nRelevance Stats by Judge and System Type (Balanced Subset):")
        print(stats_judge)
        
    else:
        print("Column 'Relevance' not found.")


except Exception as e:
    print(f"Error: {e}")
