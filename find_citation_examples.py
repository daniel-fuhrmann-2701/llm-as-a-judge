import pandas as pd
import numpy as np
import os

file_path = r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\Q&A evaluated\aggregated_results.xlsx"

def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_patterns = {
        'OAI': 'OpenAI',
        'Gemini': 'Gemini'
    }
    
    all_data = []
    
    for sheet_name in xls.sheet_names:
        judge = None
        if 'OAI' in sheet_name:
            judge = 'OpenAI'
        elif 'Gemini' in sheet_name:
            judge = 'Gemini'
        
        if not judge:
            continue
            
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if len(all_data) == 0:
            print(f"Columns in first sheet ({sheet_name}): {df.columns.tolist()}")
            
        df['Judge'] = judge
        df['Sheet'] = sheet_name
        
        # Try to identify system type and use case from sheet name
        if 'Agentic' in sheet_name:
            df['System_Type'] = 'Agentic'
        elif 'RAG' in sheet_name:
            df['System_Type'] = 'RAG'
            
        if 'G+E' in sheet_name:
            df['UseCase'] = 'Gifts_Entertainment'
        elif 'IT-G' in sheet_name:
            df['UseCase'] = 'IT_Governance'
        elif 'NHQ' in sheet_name:
            df['UseCase'] = 'New_HQ'
            
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

try:
    df = load_data(file_path)
    
    # Create a unique identifier for the question/response pair
    # Assuming 'Question' is the question text. 
    # We need to match OpenAI and Gemini evaluations for the SAME response.
    # However, the notebook logic suggests they evaluated the SAME outputs?
    # "The N=750 system outputs, each evaluated by both judges"
    # So for a given (UseCase, System_Type, Question), there is ONE output.
    # And TWO evaluations (one from OpenAI, one from Gemini).
    
    # Let's create a key
    df['Key'] = df['UseCase'].astype(str) + "|" + df['System_Type'].astype(str) + "|" + df['Question'].astype(str)
    
    # Pivot to get side-by-side scores
    # We want columns like Citation Quality_OpenAI, Citation Quality_Gemini
    
    # Check if 'Citation Quality' column exists
    cols = [c for c in df.columns if 'Citation' in c]
    print(f"Citation columns found: {cols}")
    
    citation_col = 'Citation Quality'
    if citation_col not in df.columns:
        # Maybe it has a different name
        if cols:
            citation_col = cols[0]
        else:
            print("No Citation Quality column found")
            exit()

    # Also look for Justification columns
    just_cols = [c for c in df.columns if 'Justification' in c or 'Reasoning' in c or 'Explanation' in c]
    print(f"Justification columns found: {just_cols}")
    just_col = just_cols[0] if just_cols else None

    # Pivot
    pivot_df = df.pivot_table(
        index=['Key', 'Question', 'UseCase', 'System_Type', 'Answer'], 
        columns='Judge', 
        values=[citation_col, 'Overall_Assessment'],
        aggfunc='first'
    )
    
    # Flatten columns
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    
    # Calculate difference
    openai_score_col = f"{citation_col}_OpenAI"
    gemini_score_col = f"{citation_col}_Gemini"
    
    if openai_score_col not in pivot_df.columns or gemini_score_col not in pivot_df.columns:
        print("Could not find both judge scores")
        print(pivot_df.columns)
        exit()
        
    pivot_df['Diff'] = pivot_df[openai_score_col] - pivot_df[gemini_score_col]
    pivot_df['AbsDiff'] = pivot_df['Diff'].abs()
    
    # Filter for high divergence (>= 3)
    high_div = pivot_df[pivot_df['AbsDiff'] >= 3].sort_values('AbsDiff', ascending=False)
    
    print(f"Found {len(high_div)} high divergence examples (>= 3 points)")
    
    for i, row in high_div.iloc[5:10].iterrows():
        print("-" * 50)
        print(f"Question: {row['Question']}")
        print(f"System: {row['System_Type']} ({row['UseCase']})")
        print(f"Answer: {row['Answer'][:200]}...")
        print(f"OpenAI Score: {row[openai_score_col]}")
        print(f"Gemini Score: {row[gemini_score_col]}")
        print(f"OpenAI Assessment: {row['Overall_Assessment_OpenAI']}")
        print(f"Gemini Assessment: {row['Overall_Assessment_Gemini']}")

except Exception as e:
    print(f"Error: {e}")
