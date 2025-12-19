# GDPR Compliance Agent

A specialized RAG agent for assessing GDPR compliance within the evaluation framework.

## Overview

The `GDPRComplianceAgent` extends the base `RAGAgent` to provide comprehensive assessment capabilities for GDPR compliance across key regulatory requirements and data protection principles.

## Key Features

### Core GDPR Assessment Areas

1. **Lawfulness of Processing (Article 6)**
   - Validates legal basis for processing
   - Checks consent mechanisms
   - Evaluates legitimate interests

2. **Data Minimization (Article 5(1)(c))**
   - Assesses necessity of data collection
   - Evaluates relevance to purpose
   - Checks limitation principles

3. **Purpose Limitation (Article 5(1)(b))**
   - Validates specified purposes
   - Ensures explicit communication
   - Confirms legitimate use

4. **Right to Explanation (Recital 71)**
   - Evaluates automated decision-making
   - Checks explanation provisions
   - Validates human intervention options

5. **Data Subject Rights (Articles 12-22)**
   - Assesses access rights implementation
   - Validates rectification procedures
   - Checks erasure mechanisms

6. **General Principle for Transfers (Article 44)**
   - Evaluates international transfer basis
   - Checks adequacy decisions
   - Validates safeguards

### Assessment Capabilities

- **Comprehensive Scoring**: 0-100% compliance scores per area
- **Risk Classification**: Low, Medium, High, Critical risk levels
- **Automated Recommendations**: Context-aware improvement suggestions
- **Risk Indicators**: Identification of compliance gaps and violations

## Usage

### Basic Usage

```python
from evaluation_system.agents import GDPRComplianceAgent

# Initialize agent
agent = GDPRComplianceAgent(db_path="path/to/gdpr/knowledge/base")

if agent.initialize():
    scenario = """
    Company processes customer emails for marketing with pre-checked consent.
    Data is shared with US partners without adequate safeguards.
    """

    results = agent.assess_gdpr_compliance(scenario)
    print(f"Compliance Score: {results['compliance_score']}%")
    print(f"Risk Level: {results['risk_level']}")
```

### Specific Area Assessment

```python
# Assess specific GDPR areas
lawfulness_result = agent.assess_lawfulness_of_processing(scenario)
minimization_result = agent.assess_data_minimization(scenario)
rights_result = agent.assess_data_subject_rights(scenario)

# Focus on specific area
focused_assessment = agent.assess_gdpr_compliance(
    scenario,
    focus_area="right_to_explanation"
)
```

### Generate Compliance Report

```python
assessment = agent.assess_gdpr_compliance(scenario)
report = agent.generate_compliance_report(assessment)
print(report)
```

## API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `assess_gdpr_compliance(context, focus_area=None)` | Full GDPR compliance assessment |
| `assess_lawfulness_of_processing(context)` | Article 6 compliance |
| `assess_data_minimization(context)` | Article 5(1)(c) compliance |
| `assess_purpose_limitation(context)` | Article 5(1)(b) compliance |
| `assess_right_to_explanation(context)` | Recital 71 compliance |
| `assess_data_subject_rights(context)` | Articles 12-22 compliance |
| `assess_transfer_principles(context)` | Article 44 compliance |
| `generate_compliance_report(results)` | Generate formatted report |
| `get_gdpr_guidance(topic)` | Retrieve specific GDPR guidance |

### Assessment Result Structure

```python
{
    "overall_assessment": {
        "compliance_status": "Moderate compliance - several areas need attention",
        "key_strengths": ["lawfulness: Article 6 - Lawfulness of processing"],
        "critical_gaps": ["data_minimization: Article 5(1)(c) - Data minimisation"],
        "priority_actions": ["Implement proper consent mechanism", ...],
        "regulatory_risk": "medium"
    },
    "detailed_assessments": {
        "lawfulness": {
            "area": "lawfulness",
            "gdpr_reference": "Article 6 - Lawfulness of processing",
            "compliance_score": 75,
            "recommendations": [...],
            "risk_indicators": [...]
        },
        ...
    },
    "compliance_score": 68.5,
    "risk_level": "medium"
}
```

## Testing

```bash
python test_gdpr_agent.py
```

## Limitations

- **Knowledge Base Dependency**: Accuracy depends on quality of underlying knowledge base
- **Context Sensitivity**: Requires detailed scenario descriptions for accurate assessment
- **Legal Interpretation**: Results should be validated by legal professionals
- **Jurisdiction Specificity**: Focused on EU GDPR requirements
