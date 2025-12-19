# EU AI Act Compliance Agent

A specialized RAG agent for assessing EU AI Act compliance within the evaluation framework.

## Overview

The `EUAIActAgent` extends the base `RAGAgent` to provide comprehensive assessment capabilities for EU AI Act compliance, focusing on risk categorization, prohibited practices, and requirements for high-risk AI systems.

## Key Features

### Core EU AI Act Assessment Areas

1. **Risk Categorization (Articles 6-15)**
   - Assesses if an AI system is classified as prohibited, high-risk, limited-risk, or minimal-risk
   - Validates justification for the selected risk category

2. **Prohibited AI Practices (Article 5)**
   - Checks for practices like manipulative techniques, exploitation of vulnerabilities, and social scoring

3. **High-Risk AI System Requirements (Articles 16-51)**
   - Evaluates compliance with risk management, data governance, technical documentation, transparency, human oversight, and cybersecurity

4. **Transparency Obligations (Article 52)**
   - Assesses if AI interactions and synthetic content are properly disclosed to users

5. **CE Marking (Article 48)**
   - Checks for conformity assessment and proper CE marking for high-risk systems

### Assessment Capabilities

- **Comprehensive Scoring**: 0-100% compliance scores per area
- **Risk Classification**: Low, Medium, High, Critical risk levels
- **Automated Recommendations**: Context-aware improvement suggestions
- **Risk Indicators**: Identification of compliance gaps and potential violations

## Usage

### Basic Usage

```python
from evaluation_system.agents import EUAIActAgent

# Initialize agent
agent = EUAIActAgent(db_path="path/to/eu-ai-act/knowledge/base")

if agent.initialize():
    scenario = """
    An AI system used for credit scoring, making decisions on loan applications.
    The model is a complex neural network, and its training data includes demographic information.
    """

    results = agent.assess_eu_ai_act_compliance(scenario)
    print(f"Compliance Score: {results['compliance_score']}%")
    print(f"Risk Level: {results['risk_level']}")
```

### Specific Area Assessment

```python
# Assess specific EU AI Act areas
risk_cat_result = agent.assess_risk_categorization(scenario)
high_risk_req_result = agent.assess_high_risk_requirements(scenario)
transparency_result = agent.assess_transparency_obligations(scenario)

# Focus on a specific area
focused_assessment = agent.assess_eu_ai_act_compliance(
    scenario,
    focus_area="high_risk_requirements"
)
```

### Generate Compliance Report

```python
assessment = agent.assess_eu_ai_act_compliance(scenario)
report = agent.generate_compliance_report(assessment)
print(report)
```

## API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `assess_eu_ai_act_compliance(context, focus_area=None)` | Full EU AI Act compliance assessment |
| `assess_risk_categorization(context)` | Articles 6-15 risk categorization |
| `assess_prohibited_practices(context)` | Article 5 prohibited practices |
| `assess_high_risk_requirements(context)` | Articles 16-51 high-risk requirements |
| `assess_transparency_obligations(context)` | Article 52 transparency |
| `assess_ce_marking_requirements(context)` | Article 48 CE marking |
| `generate_compliance_report(results)` | Generate formatted report |
| `get_ai_act_guidance(topic)` | Retrieve specific EU AI Act guidance |

## Testing

```bash
python test_eu_ai_act_agent.py
```

## Limitations

- **Knowledge Base Dependency**: Accuracy depends on quality and completeness of the underlying knowledge base
- **Legal Interpretation**: Results are not a substitute for professional legal advice
- **Dynamic Regulation**: The AI Act and its interpretations will evolve; the knowledge base requires continuous updates
