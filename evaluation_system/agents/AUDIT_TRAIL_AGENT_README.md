# Audit Trail Agent

A specialized agent for evaluating audit trails based on completeness, quality, and compliance with best practices.

## Overview

The `AuditTrailAgent` analyzes audit logs to assess their effectiveness in providing a clear and comprehensive record of system activities. Unlike other agents, it directly processes log files rather than querying a vector database, using LLM capabilities to evaluate the content.

## Key Features

### Core Assessment Areas

1. **Completeness**
   - Verifies that all critical system events are logged
   - Checks for the presence of essential data fields (timestamp, user, action)
   - Identifies potential gaps in the audit trail

2. **Quality and Consistency**
   - Assesses the clarity and detail of log messages
   - Evaluates the consistency of the log format
   - Checks for excessive noise or irrelevant entries

3. **Security and Integrity**
   - Looks for signs of log tampering or unauthorized access
   - Assesses if logs are stored securely

### Assessment Capabilities

- **Direct Log Analysis**: Reads and evaluates raw log files from a specified directory
- **Scoring System**: Provides a 0-100 score for each assessment area and an overall score
- **Automated Recommendations**: Generates actionable suggestions for improving the audit trail

## Usage

### Basic Usage

```python
from evaluation_system.agents import AuditTrailAgent

# Initialize the agent with the path to your audit logs
audit_log_directory = "path/to/your/audit_logs"
agent = AuditTrailAgent(audit_log_path=audit_log_directory)

if agent.initialize():
    # Perform a comprehensive assessment
    assessment_results = agent.assess_audit_trail()

    print(f"Overall Audit Trail Score: {assessment_results['overall_score']}/100")

    # Generate and print the report
    report = agent.generate_assessment_report(assessment_results)
    print(report)
```

### Specific Area Assessment

```python
# Focus on a specific area, like 'completeness' or 'quality'
quality_assessment = agent.assess_audit_trail(focus_area="quality")

# Print the detailed analysis for the focused area
print(quality_assessment['detailed_assessments']['quality']['analysis'])
```

## API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `__init__(audit_log_path, agent_name="Audit Trail Agent")` | Initialize with path to audit logs |
| `initialize()` | Initialize the agent's LLM (required before assessments) |
| `assess_audit_trail(focus_area=None)` | Perform comprehensive assessment |
| `generate_assessment_report(results)` | Generate formatted text report |

### Assessment Result Structure

```python
{
    "overall_assessment": {
        "summary_statement": "The audit trail is adequate but has several areas...",
        "key_strengths": ["Completeness: Strong performance."],
        "critical_weaknesses": ["Security: Needs significant improvement."],
        "priority_actions": ["Implement log integrity checks.", ...]
    },
    "detailed_assessments": {
        "completeness": {
            "area": "completeness",
            "assessment_criteria": ["all_events_logged", ...],
            "analysis": "The logs cover most critical events...",
            "score": 85,
            "recommendations": ["Add logging for user role changes."]
        },
        ...
    },
    "recommendations": ["Implement log integrity checks.", ...],
    "overall_score": 65.0
}
```

## Testing

```bash
python test_audit_trail_agent.py
```

## Limitations

- **LLM Dependency**: Assessment quality depends on the configured LLM's analytical capabilities
- **Context Sensitivity**: Assessment is based solely on provided log files and may lack broader system context
- **Parsing Accuracy**: The agent expects JSON-formatted responses from the LLM; parsing failures can affect results
- **Scalability**: For very large log volumes, content is truncated, which may affect assessment completeness
