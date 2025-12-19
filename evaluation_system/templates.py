"""
Template management for the RAG vs Agentic AI Evaluation Framework.

This module handles Jinja2 template loading and rendering for evaluation prompts
and other templated content.
"""

from datetime import datetime
from jinja2 import Template

from .config import EvalConfig


def load_system_prompt(template_str: str, 
                      config: EvalConfig,
                      timestamp: str = None) -> str:
    """
    Load and render the academic evaluation prompt with rubrics.
    
    Args:
        template_str: The Jinja2 template string
        config: Evaluation configuration containing rubrics
        timestamp: Optional timestamp string (defaults to current time)
    
    Returns:
        Rendered prompt string ready for LLM consumption
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    return Template(template_str).render(
        rubrics=config.rubrics,
        dimensions=config.dimensions,
        timestamp=timestamp
    )


def render_template(template_str: str, **kwargs) -> str:
    """
    Generic template rendering function.
    
    Args:
        template_str: The Jinja2 template string
        **kwargs: Template variables
    
    Returns:
        Rendered template string
    """
    return Template(template_str).render(**kwargs)


# Enhanced academic evaluation prompt with snippet support
SNIPPET_EVALUATION_PROMPT = """
You are an expert academic evaluator specializing in Retrieval-Augmented Generation (RAG) systems assessment.

Your task is to evaluate an AI system's response using rigorous academic standards, with special attention to source snippet grounding and citation quality.

## EVALUATION FRAMEWORK

### Core Academic Dimensions (Score 1-5):
{% for dimension in dimensions %}{{ loop.index }}. **{{ dimension.value.replace('_', ' ').title() }}**: {{ rubrics[dimension].description }}
{% endfor %}

### Snippet-Specific Evaluation Criteria:
- **Grounding Score**: How well the answer is supported by provided snippets
- **Citation Alignment**: Accuracy of references to specific snippet content (if present)
- **Source Attribution**: Proper acknowledgment of information sources (if provided)
- **Content Synthesis**: Integration of multiple snippet sources

### Academic Rubric (1-5 scale):
- **5 (Excellent)**: Exceeds academic standards (90-100%)
- **4 (Good)**: Meets high academic standards (80-89%)  
- **3 (Satisfactory)**: Acceptable academic quality (70-79%)
- **2 (Needs Improvement)**: Below academic standards (60-69%)
- **1 (Inadequate)**: Significantly deficient (<60%)

### Special Considerations for Citation Quality:
- **Systems WITHOUT explicit citations**: Focus on content accuracy and implicit grounding in provided snippets
- **Systems WITH explicit citations**: Evaluate both citation presence and accuracy
- **Mixed approaches**: Assess based on system capabilities and context

## INPUT DATA

**User Query**: {{ query }}

**AI Response**: {{ answer }}

{% if snippets and snippets|length > 0 %}
**Source Snippets** ({{ snippets|length }} total):
{% for snippet in snippets %}
**Snippet {{ snippet.snippet_id or loop.index }}**:
{{ snippet.content }}
{% if snippet.source_document %}
*Source: {{ snippet.source_document }}*
{% endif %}

{% endfor %}

### SNIPPET ANALYSIS REQUIRED:
For RAG systems, you must evaluate:
1. How well the answer references and uses the provided snippets
2. Whether claims in the answer are supported by snippet content  
3. Quality of information synthesis across multiple snippets
4. **Citation Quality Assessment**:
   - If the system provides explicit citations/references: Evaluate accuracy and completeness
   - If the system does NOT provide explicit citations: Focus on implicit grounding and content accuracy
   - Rate based on the system's design capabilities, not the absence of features it wasn't designed to provide

**IMPORTANT**: Do not penalize systems for missing explicit citations if they demonstrate strong implicit grounding in the source material.

{% else %}
**Note**: No source snippets provided. Evaluate based on general knowledge and content quality.
{% endif %}

## EVALUATION INSTRUCTIONS

1. **Read Carefully**: Analyze the query, response{% if snippets %}, and all provided snippets{% endif %}
2. **Score Each Dimension**: Assign scores 1-5 based on the academic rubric
3. **Provide Evidence**: Include specific text excerpts supporting each score
4. **Calculate Confidence**: Rate your evaluation confidence (0.0-1.0)
5. **Academic Justification**: Explain reasoning with reference to academic standards

{% if snippets and snippets|length > 0 %}
6. **Snippet Grounding**: Assess how well the answer leverages provided source material
7. **Citation Evaluation**: 
   - For systems WITH explicit citations: Check accuracy of references to snippet content
   - For systems WITHOUT explicit citations: Evaluate implicit grounding and content derivation from snippets
   - Do not penalize systems for architectural design choices (e.g., no explicit citation mechanism)
{% endif %}

## OUTPUT FORMAT (JSON)

You must respond with a valid JSON object in the following format:

{
    "scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <score 1-5>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "justifications": {
        {% for dimension in dimensions %}"{{ dimension.value }}": "<evidence-based justification>"{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "confidence_scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <confidence 0.0-1.0>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "overall_assessment": "<comprehensive evaluation summary>",
    {% if snippets and snippets|length > 0 %}
    "snippet_analysis": {
        "grounding_quality": "<how well answer uses snippets>",
        "citation_approach": "<explicit citations present: true/false>",
        "citation_accuracy": "<if explicit citations: accuracy assessment; if no citations: content grounding assessment>",
        "synthesis_quality": "<quality of multi-snippet integration>"
    },
    {% endif %}
    "academic_recommendation": "<actionable feedback for improvement>"
}

Focus on academic rigor, evidence-based evaluation, and transparent methodology. All scores must be supported by specific textual evidence."""


# Standard evaluation prompt (backwards compatibility)
STANDARD_EVALUATION_PROMPT = """
You are an expert academic evaluator for AI system responses.

Your task is to evaluate the following AI response using rigorous academic standards.

## EVALUATION DIMENSIONS (Score 1-5):
{% for dimension in dimensions %}{{ loop.index }}. **{{ dimension.value.replace('_', ' ').title() }}**: {{ rubrics[dimension].description }}
{% endfor %}

## INPUT DATA

**User Query**: {{ query }}

**AI Response**: {{ answer }}

## EVALUATION RUBRIC (1-5 scale):
- **5 (Excellent)**: Exceeds academic standards (90-100%)
- **4 (Good)**: Meets high academic standards (80-89%)
- **3 (Satisfactory)**: Acceptable academic quality (70-79%)
- **2 (Needs Improvement)**: Below academic standards (60-69%)
- **1 (Inadequate)**: Significantly deficient (<60%)

## OUTPUT FORMAT (JSON)

You must respond with a valid JSON object in the following format:

{
    "scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <score 1-5>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "justifications": {
        {% for dimension in dimensions %}"{{ dimension.value }}": "<evidence-based justification>"{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "confidence_scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <confidence 0.0-1.0>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "overall_assessment": "<comprehensive evaluation summary>"
}

Focus on academic rigor, evidence-based evaluation, and transparent methodology."""

# Enhanced evaluation prompt with proper separation of concerns
ENHANCED_SNIPPET_EVALUATION_PROMPT = """
You are an expert academic evaluator specializing in AI system assessment with expertise in both content quality and information retrieval.

Your task is to evaluate an AI system's response using rigorous academic standards with proper separation between:
1. **Content Quality**: Factual accuracy, relevance, completeness based on domain knowledge
2. **Source Utilization**: How well the system uses provided source materials
3. **Citation Mechanics**: Attribution and referencing practices

## EVALUATION FRAMEWORK

### Primary Academic Dimensions (Score 1-5):
{% for dimension in dimensions %}{{ loop.index }}. **{{ dimension.value.replace('_', ' ').title() }}**: {{ rubrics[dimension].description }}
{% endfor %}

### CRITICAL EVALUATION PRINCIPLES:

**Factual Accuracy Assessment:**
- Evaluate correctness based on domain knowledge and established facts
- A response can be factually accurate even if not explicitly supported by snippets
- Focus on whether the information is objectively correct, not just snippet-supported

**Relevance Assessment:**
- Evaluate how well the response addresses the user's question
- Consider directness, completeness of addressing query intent
- Independent of source material quality

**Source Grounding vs. Content Quality:**
- **Source Grounding**: Separate assessment of how well snippets are utilized
- **Content Quality**: Independent assessment of response accuracy and relevance
- Do NOT penalize accurate responses for poor snippet support if the content is correct

### Academic Rubric (1-5 scale):
- **5 (Excellent)**: Exceeds academic standards (90-100%)
- **4 (Good)**: Meets high academic standards (80-89%)  
- **3 (Satisfactory)**: Acceptable academic quality (70-79%)
- **2 (Needs Improvement)**: Below academic standards (60-69%)
- **1 (Inadequate)**: Significantly deficient (<60%)

## INPUT DATA

**User Query**: {{ query }}

**AI Response**: {{ answer }}

{% if snippets and snippets|length > 0 %}
**Source Snippets** ({{ snippets|length }} total):
{% for snippet in snippets %}
**Snippet {{ snippet.snippet_id or loop.index }}**:
{{ snippet.content }}
{% if snippet.source_document %}
*Source: {{ snippet.source_document }}*
{% endif %}

{% endfor %}

### EVALUATION APPROACH FOR SNIPPET-BASED SYSTEMS:

**Step 1: Content Quality Assessment (Independent)**
- Evaluate factual accuracy based on domain knowledge
- Assess relevance to user query independent of snippets
- Judge completeness and clarity on their own merits

**Step 2: Source Utilization Assessment (Separate)**
- How effectively does the response use provided snippets?
- Are claims supported where snippet evidence exists?
- Quality of information synthesis across sources

**Step 3: Citation Quality Assessment (System-Aware)**
- For systems WITH explicit citations: Evaluate accuracy and completeness
- For systems WITHOUT explicit citations: Focus on implicit content grounding
- DO NOT penalize systems for architectural design choices

**IMPORTANT PRINCIPLE:** 
A factually correct, relevant response should score highly on content dimensions even if snippet support is weak. Conversely, a response well-grounded in snippets but factually incorrect should score poorly on factual accuracy.

{% else %}
**Note**: No source snippets provided. Evaluate based on content quality and domain knowledge.
{% endif %}

## EVALUATION INSTRUCTIONS

1. **Analyze Content First**: Evaluate factual accuracy, relevance, completeness based on the response quality itself
2. **Assess Source Use Separately**: If snippets provided, evaluate how well they're utilized (separate from content accuracy)
3. **System-Aware Evaluation**: Consider the system's intended capabilities and design
4. **Evidence-Based Scoring**: Support all scores with specific justifications
5. **Academic Rigor**: Apply consistent, transparent methodology

{% if snippets and snippets|length > 0 %}
6. **Dual Assessment Approach**:
   - **Content Dimensions**: Score based on response quality and domain knowledge
   - **Source Dimensions**: Score based on snippet utilization and grounding
   - **DO NOT conflate these two aspects**
{% endif %}

## OUTPUT FORMAT (JSON)

You must respond with a valid JSON object in the following format:

{
    "scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <score 1-5>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "justifications": {
        {% for dimension in dimensions %}"{{ dimension.value }}": "<evidence-based justification explaining why this score was assigned, referencing specific content>"{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "confidence_scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <confidence 0.0-1.0>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "overall_assessment": "<comprehensive evaluation summary addressing both content quality and source utilization>",
    {% if snippets and snippets|length > 0 %}
    "snippet_analysis": {
        "grounding_quality": "<assessment of how well the response uses snippets - separate from content accuracy>",
        "citation_approach": "<explicit citations present: true/false>",
        "content_vs_snippet_alignment": "<how well the content aligns with snippet information, noting any discrepancies>",
        "synthesis_quality": "<quality of multi-snippet integration if applicable>"
    },
    {% endif %}
    "evaluation_methodology": "<explanation of how content quality was separated from source grounding in this assessment>"
}

**REMEMBER**: Factual accuracy should be evaluated based on whether the information is objectively correct, NOT whether it's supported by snippets. A response can be perfectly accurate even with poor snippet support."""
