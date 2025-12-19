# Statistical Evaluation

This folder contains the reliability analysis for comparing LLM judges (GPT-4o-mini and Gemini Flash).

## Files Overview

| File | Description |
|------|-------------|
| `Reliability_Study_v2.ipynb` | Main analysis notebook with inter-rater reliability metrics |
| `pairwise_study.ipynb` | Pairwise preference agreement analysis |
| `aggregated_results.xlsx` | All evaluation results (Q&A columns redacted) |
| `verify_stats.py` | Helper script for statistical calculations |

## What You Can Find

### Reliability_Study_v2.ipynb
- Spearman correlation between judges
- Weighted Kappa agreement scores
- Breakdown by evaluation dimension
- Visualizations of judge agreement

### pairwise_study.ipynb
- Pairwise preference analysis (RAG vs Agentic)
- Agreement rates between judges

### aggregated_results.xlsx
- Raw evaluation scores from both judges
- Results grouped by system type and topic
- Note: Question and Answer columns are removed for confidentiality

## How to Run

```bash
jupyter notebook "Statistical Evaluation/"
```

## Data Availability

The complete datasets with Q&A content can be provided to the evaluation committee upon request.
