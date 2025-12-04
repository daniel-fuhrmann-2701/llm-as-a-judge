#!/usr/bin/env python3
"""
Enhanced evaluation script that properly includes snippets for citation and provenance scoring.

This script loads Excel data with Question, Answer, and Snippet columns,
then evaluates the answers with proper snippet context for accurate 
citation quality and provenance assessment.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import with error handling and fallbacks
try:
    # Try to import the actual evaluation system first
    from evaluation_system.config import EvalConfig, logger, get_default_config
    from evaluation_system.excel_processor import load_excel_with_snippets, validate_excel_structure
    from evaluation_system.evaluation import evaluate_answers_with_snippets
    from evaluation_system.report import generate_summary_report
    from evaluation_system.utils import ensure_directory
    EVALUATION_SYSTEM_AVAILABLE = True
    logger.info("Using full evaluation_system with academic rubrics")
except ImportError as e:
    print(f"Warning: evaluation_system modules not available: {e}")
    print("Using fallback implementations...")
    EVALUATION_SYSTEM_AVAILABLE = False
    
    # Fallback imports
    import pandas as pd
    import logging
    
    # Create basic logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  Using fallback evaluation - install evaluation_system for full academic analysis")
    
    # Fallback functions
    def ensure_directory(path):
        """Fallback ensure directory function"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_default_config():
        """Fallback configuration"""
        return EvalConfig()
    
    def validate_excel_structure(excel_path):
        """Fallback validation function"""
        try:
            df = pd.read_excel(excel_path)
            return {
                "valid": True,
                "total_rows": len(df),
                "valid_rows": len(df.dropna()),
                "columns": list(df.columns)
            }
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    def load_excel_with_snippets(excel_file, question_col="Question", answer_col="Answer", snippet_col="Database_Used"):
        """Enhanced fallback Excel loading function"""
        df = pd.read_excel(excel_file)
        
        logger.info(f"📊 Excel columns detected: {list(df.columns)}")
        logger.info(f"📊 First few rows preview:")
        logger.info(df.head(2).to_string())
        
        # Create evaluation input objects with metadata
        evaluation_inputs = []
        for idx, row in df.iterrows():
            # Extract metadata if available
            metadata = {}
            metadata_mapping = {
                'question_id': 'Question_ID',
                'search_method': 'Search_Method',
                'status': 'Status',
                'fallback_used': 'Fallback_Used', 
                'web_fallback_status': 'Web_Fallback_Status',
                'error': 'Error'
            }
            
            for meta_key, meta_col in metadata_mapping.items():
                if meta_col in df.columns:
                    metadata[meta_key] = str(row.get(meta_col, ""))
                else:
                    logger.warning(f"⚠️  Column '{meta_col}' not found in Excel file")
            
            # Debug: log metadata for first few rows
            if idx < 3:
                logger.info(f"📋 Row {idx+1} metadata: {metadata}")
            
            # Create snippet from database_used or snippet column
            snippet_content = str(row.get(snippet_col, ""))
            snippets = []
            if pd.notna(row.get(snippet_col)) and snippet_content.strip():
                snippets = [type('Snippet', (), {
                    'content': snippet_content,
                    'source': 'database_retrieval',
                    'metadata': metadata
                })()]
            
            # Create evaluation input object
            item = type('EvaluationInput', (), {
                'query': str(row.get(question_col, "")),
                'answer': str(row.get(answer_col, "")),
                'source_snippets': snippets,
                'metadata': metadata,
                'row_index': idx
            })()
            evaluation_inputs.append(item)
        
        return evaluation_inputs
    
    def evaluate_answers_with_snippets(evaluation_inputs, config):
        """Enhanced fallback evaluation function with metadata analysis"""
        evaluations = []
        
        # Analyze patterns in the data
        fallback_used_count = sum(1 for item in evaluation_inputs 
                                 if item.metadata.get('fallback_used', '').upper() == 'TRUE')
        success_count = sum(1 for item in evaluation_inputs 
                           if item.metadata.get('status', '') == 'success')
        
        logger.info(f"Data Analysis:")
        logger.info(f"  - Total items: {len(evaluation_inputs)}")
        logger.info(f"  - Success rate: {success_count}/{len(evaluation_inputs)} ({success_count/len(evaluation_inputs)*100:.1f}%)")
        logger.info(f"  - Fallback usage: {fallback_used_count}/{len(evaluation_inputs)} ({fallback_used_count/len(evaluation_inputs)*100:.1f}%)")
        
        for i, item in enumerate(evaluation_inputs):
            # Calculate base scores
            base_score = 0.75
            
            # Adjust score based on metadata
            status = item.metadata.get('status', '')
            fallback_used = item.metadata.get('fallback_used', '').upper() == 'TRUE'
            web_fallback_status = item.metadata.get('web_fallback_status', '')
            has_error = item.metadata.get('error', '') not in ['None', '', 'N/A']
            
            # Score adjustments
            if status == 'success':
                base_score += 0.15
            if fallback_used and web_fallback_status == 'success':
                base_score += 0.1  # Bonus for successful fallback
            elif fallback_used and web_fallback_status != 'success':
                base_score -= 0.1  # Penalty for failed fallback
            if has_error:
                base_score -= 0.2
            
            # Ensure score is in valid range
            base_score = max(0.0, min(1.0, base_score))
            
            # Analyze answer quality and classify system type
            answer_length = len(item.answer)
            has_sources = 'Sources:' in item.answer or '#' in item.answer
            has_structured_content = any(marker in item.answer for marker in ['1.', '2.', '3.', '-', '*'])
            
            # Classify system type based on characteristics
            rag_indicators = 0
            agentic_indicators = 0
            
            # RAG system indicators
            if has_sources:
                rag_indicators += 2
            if 'Based on my search' in item.answer or 'According to' in item.answer:
                rag_indicators += 1
            if 'database' in item.metadata.get('search_method', '').lower():
                rag_indicators += 1
            if len(item.source_snippets) > 0:
                rag_indicators += 1
            
            # Agentic system indicators  
            if item.metadata.get('fallback_used', '').upper() == 'TRUE':
                agentic_indicators += 2  # Fallback suggests autonomous reasoning
            if 'Web Search' in item.answer:
                agentic_indicators += 1
            if answer_length > 300:  # Longer, more complex responses
                agentic_indicators += 1
            if any(phrase in item.answer.lower() for phrase in ['i recommend', 'you should', 'the best approach', 'steps to take']):
                agentic_indicators += 1
            
            # Determine system type
            if rag_indicators > agentic_indicators:
                system_type_str = "rag_system"
            elif agentic_indicators > rag_indicators:
                system_type_str = "agentic_system" 
            else:
                system_type_str = "hybrid_system"
            
            # Quality adjustments
            quality_score = base_score
            if answer_length > 200:  # Detailed answer
                quality_score += 0.05
            if has_sources:  # Has citations
                quality_score += 0.1
            if has_structured_content:  # Well-structured
                quality_score += 0.05
            if "I don't know" in item.answer:  # Honest uncertainty
                quality_score -= 0.3
            
            quality_score = max(0.0, min(1.0, quality_score))
            
            # Calculate snippet grounding score
            snippet_grounding = 0.0
            if item.source_snippets:
                snippet_grounding = 0.8 if has_sources else 0.6
            
            # Create evaluation result
            eval_result = type('EvaluationResult', (), {
                'weighted_total': quality_score,
                'raw_total': base_score,
                'scores': {
                    'accuracy': quality_score + 0.1 if status == 'success' else quality_score - 0.1,
                    'relevance': quality_score,
                    'completeness': quality_score + 0.05 if has_structured_content else quality_score - 0.05,
                    'citation_quality': 0.9 if has_sources else 0.3,
                    'clarity': 0.8 if has_structured_content else 0.6
                },
                'justifications': {
                    'accuracy': f"Status: {status}, Fallback: {fallback_used}, Error: {has_error}",
                    'relevance': f"Answer addresses the question with {answer_length} characters",
                    'completeness': f"Structured content: {has_structured_content}, Sources: {has_sources}",
                    'citation_quality': f"Sources cited: {has_sources}",
                    'clarity': f"Well-structured: {has_structured_content}"
                },
                'overall_assessment': f"Score: {quality_score:.3f} | Status: {status} | Type: {system_type_str} | Sources: {has_sources}",
                'evaluation_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'evaluator': 'enhanced_fallback',
                    'original_metadata': item.metadata,
                    'answer_length': answer_length,
                    'has_sources': has_sources,
                    'has_structured_content': has_structured_content,
                    'rag_indicators': rag_indicators,
                    'agentic_indicators': agentic_indicators
                },
                'system_type': type('SystemType', (), {'value': system_type_str})(),
                'source_snippets': item.source_snippets,
                'snippet_grounding_score': snippet_grounding,
                'fallback_analysis': {
                    'used_fallback': fallback_used,
                    'fallback_success': web_fallback_status == 'success' if fallback_used else None,
                    'primary_source': item.metadata.get('search_method', 'unknown'),
                    'system_classification': system_type_str,
                    'classification_confidence': abs(rag_indicators - agentic_indicators) / max(rag_indicators + agentic_indicators, 1)
                }
            })()
            evaluations.append(eval_result)
        
        return evaluations
    
    def generate_summary_report(evaluations):
        """Fallback report generation"""
        report = f"""# Fallback Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total items evaluated: {len(evaluations)}
- Average score: {sum(eval.weighted_total for eval in evaluations) / len(evaluations):.3f}

**Note**: This is a fallback report. For detailed analysis, please install the full evaluation_system package.

## Items Evaluated
"""
        for i, eval in enumerate(evaluations, 1):
            report += f"\n### Item {i}\n"
            report += f"- Score: {eval.weighted_total:.3f}\n"
            report += f"- Assessment: {eval.overall_assessment}\n"
        
        return report
    
    class EvalConfig:
        """Fallback configuration class"""
        def __init__(self):
            self.model_name = "fallback-model"
            self.temperature = 0.1
            self.max_retries = 3
            self.dimensions = [type('Dimension', (), {'value': dim})() for dim in ['accuracy', 'relevance', 'completeness']]


def main():
    """Main evaluation function with snippet-enhanced context."""
    
    # =============================================================================
    # CONFIGURATION - EASILY CHANGE THESE FOR DIFFERENT FILES
    # =============================================================================
    
    # Try to import configuration, fall back to defaults if not available
    try:
        from snippets_evaluation_config import CURRENT_SETTINGS, EXCEL_CONFIGURATIONS, EVALUATION_OPTIONS, EVALUATION_PROFILES
        
        current_file_key = CURRENT_SETTINGS["file"]
        evaluation_profile_key = CURRENT_SETTINGS.get("evaluation_profile", "academic_research")
        
        if current_file_key not in EXCEL_CONFIGURATIONS:
            logger.error(f"Unknown file key: {current_file_key}")
            logger.info(f"Available files: {list(EXCEL_CONFIGURATIONS.keys())}")
            return
            
        if evaluation_profile_key not in EVALUATION_PROFILES:
            logger.error(f"Unknown evaluation profile: {evaluation_profile_key}")
            logger.info(f"Available profiles: {list(EVALUATION_PROFILES.keys())}")
            return
            
        config_data = EXCEL_CONFIGURATIONS[current_file_key]
        profile_info = EVALUATION_PROFILES[evaluation_profile_key]
        
        excel_file = Path(__file__).parent / config_data["file_path"]
        column_mapping = {
            "question_col": config_data["question_col"],
            "answer_col": config_data["answer_col"], 
            "snippet_col": config_data["snippet_col"]
        }
        description = config_data["description"]
        
        logger.info(f"Target: {description}")
        logger.info(f"Evaluation Profile: {profile_info['name']}")
        logger.info(f"Profile Description: {profile_info['description']}")
        logger.info(f"Emphasis: {', '.join(profile_info['emphasis'])}")
        
    except ImportError:
        # Fallback configuration if config file is not available
        logger.warning("⚠️  Configuration file not found, using fallback settings")
        current_file_key = "newHQ"
        evaluation_profile_key = "agentic_focused"
        excel_file = Path(__file__).parent / "Q&A/Agentic q&a/newHQ.xlsx"
        column_mapping = {
            "question_col": "Question",
            "answer_col": "Answer", 
            "snippet_col": "Database_Used"
        }
        description = "IT Governance Q&A evaluation (fallback configuration)"
        profile_info = {"name": "Academic Research (Fallback)", "description": "Fallback academic evaluation"}
    
    output_dir = f"{current_file_key}_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"=== RAG Evaluation with Snippet Context - {current_file_key.upper()} ===")
    logger.info(f"Script location: {Path(__file__).parent}")
    logger.info(f"Excel file: {excel_file}")
    logger.info(f"Excel file exists: {excel_file.exists()}")
    logger.info(f"Column mapping: {column_mapping}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if Excel file exists, if not, provide helpful information
    if not excel_file.exists():
        logger.error(f"Excel file not found: {excel_file}")
        logger.info("Available files in directory:")
        for file in Path(__file__).parent.glob("*.xlsx"):
            logger.info(f"  - {file.name}")
        
        # Check Q&A subdirectory
        qa_dir = Path(__file__).parent / "Q&A" / "Agentic q&a"
        if qa_dir.exists():
            logger.info("Available files in Q&A/Agentic q&a/:")
            for file in qa_dir.glob("*.xlsx"):
                logger.info(f"  - {file.name}")
        
        logger.info(f"Please ensure the file exists or update CURRENT_FILE to one of the available options")
        return
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Validate Excel structure first
    logger.info("Validating Excel file structure...")
    try:
        excel_path = Path(excel_file).resolve()
        logger.info(f"Validating Excel file at: {excel_path}")
        
        # First, let's check what columns are actually in the file
        import pandas as pd
        df = pd.read_excel(excel_path)
        actual_columns = list(df.columns)
        logger.info(f"Actual columns in file: {actual_columns}")
        logger.info(f"Expected columns: {column_mapping}")
        
        # Check if expected columns exist
        missing_columns = []
        for col_type, col_name in column_mapping.items():
            if col_name not in actual_columns:
                missing_columns.append(f"{col_type}: '{col_name}'")
        
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            logger.info("You may need to update the column mapping for this file")
            # Let's try to auto-detect similar columns
            logger.info("Auto-detecting similar columns:")
            for col_type, expected_col in column_mapping.items():
                similar_cols = [col for col in actual_columns if expected_col.lower() in col.lower() or col.lower() in expected_col.lower()]
                if similar_cols:
                    logger.info(f"  {col_type}: Found similar columns: {similar_cols}")
        
        validation_result = validate_excel_structure(str(excel_path))
        logger.info(f"Excel validation: {validation_result}")
        
        if not validation_result.get("valid", False):
            logger.warning("Excel file validation had issues, but proceeding anyway")
            if "errors" in validation_result:
                for error in validation_result["errors"]:
                    logger.warning(f"  - {error}")
        else:
            logger.info(f"Excel file is valid: {validation_result.get('valid_rows', 0)}/{validation_result.get('total_rows', 0)} valid rows")
        
    except Exception as e:
        logger.error(f"Failed to validate Excel file: {e}")
        logger.info("Proceeding anyway - will attempt to load with available columns")
    
    # Load evaluation data with snippets using dynamic column mapping
    logger.info("Loading evaluation data with snippets...")
    try:
        evaluation_inputs = load_excel_with_snippets(
            excel_file,
            question_col=column_mapping["question_col"],
            answer_col=column_mapping["answer_col"], 
            snippet_col=column_mapping["snippet_col"]
        )
        
        logger.info(f"Loaded {len(evaluation_inputs)} evaluation items")
        
        # Log snippet statistics
        total_snippets = sum(len(item.source_snippets) for item in evaluation_inputs)
        items_with_snippets = sum(1 for item in evaluation_inputs if item.source_snippets)
        
        logger.info(f"Snippet statistics:")
        logger.info(f"  - Total snippets: {total_snippets}")
        logger.info(f"  - Items with snippets: {items_with_snippets}/{len(evaluation_inputs)}")
        
        # Show example of loaded data
        if evaluation_inputs:
            example = evaluation_inputs[0]
            logger.info(f"Example loaded item:")
            logger.info(f"  - Query: {example.query[:100]}...")
            logger.info(f"  - Answer: {example.answer[:100]}...")
            logger.info(f"  - Snippets: {len(example.source_snippets)}")
            if example.source_snippets:
                logger.info(f"  - First snippet: {example.source_snippets[0].content[:100]}...")
        
    except Exception as e:
        logger.error(f"Failed to load Excel data: {e}")
        return
    
    # Create evaluation configuration
    logger.info("Setting up evaluation configuration...")
    if EVALUATION_SYSTEM_AVAILABLE:
        try:
            # Try to use the advanced configuration system with profile selection
            from evaluation_system.advanced_config import AdvancedEvalConfig, ConfigurationProfile, get_config_for_profile
            
            # Map our profile keys to advanced config profiles
            profile_mapping = {
                "rag_focused": ConfigurationProfile.RAG_FOCUSED,
                "agentic_focused": ConfigurationProfile.AGENTIC_FOCUSED, 
                "academic_research": ConfigurationProfile.ACADEMIC_RESEARCH,
                "comparative_study": ConfigurationProfile.COMPARATIVE_STUDY,
                "production_evaluation": ConfigurationProfile.PRODUCTION_EVALUATION
            }
            
            if evaluation_profile_key in profile_mapping:
                selected_profile = profile_mapping[evaluation_profile_key]
                config = get_config_for_profile(selected_profile)
                logger.info(f"Using advanced evaluation configuration: {selected_profile.value}")
                
                # Log profile-specific settings
                if evaluation_profile_key == "rag_focused":
                    logger.info("RAG-focused evaluation emphasizes:")
                    logger.info("  - Citation Quality & Source Grounding")
                    logger.info("  - Factual Accuracy (30% weight)")
                    logger.info("  - Retrieval Effectiveness")
                elif evaluation_profile_key == "agentic_focused":
                    logger.info("Agentic-focused evaluation emphasizes:")
                    logger.info("  - Completeness & Problem-solving (25% weight)")
                    logger.info("  - Relevance & Multi-step reasoning (25% weight)")
                    logger.info("  - Autonomous capability assessment")
                
            else:
                # Fallback to default academic config
                config = get_default_config()
                logger.info("✅ Using default academic evaluation configuration")
                
        except ImportError:
            # Fallback if advanced config not available
            config = get_default_config()
            logger.info("⚠️  Advanced config not available, using basic academic configuration")
    else:
        # Use fallback configuration
        config = EvalConfig()
        logger.info("⚠️  Using fallback configuration")
    
    # Log configuration details
    logger.info(f"Evaluation configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Temperature: {config.temperature}")
    if hasattr(config, 'dimensions'):
        logger.info(f"  - Dimensions: {[dim.value for dim in config.dimensions]}")
    if hasattr(config, 'rubrics'):
        logger.info(f"  - Academic rubrics: {len(config.rubrics)} dimensions with weighted scoring")
    logger.info(f"  - Max retries: {config.max_retries}")
    if hasattr(config, 'confidence_threshold'):
        logger.info(f"  - Confidence threshold: {config.confidence_threshold}")
    
    # Log evaluation weights if available
    if hasattr(config, 'rubrics') and config.rubrics:
        logger.info(f"Evaluation weights (Profile: {evaluation_profile_key}):")
        for dim, rubric in config.rubrics.items():
            logger.info(f"  - {dim.value}: {rubric.weight:.3f}")
    
    # Load metadata from config if needed
    try:
        from snippets_evaluation_config import EVALUATION_WEIGHTS, ANALYSIS_CATEGORIES
        logger.info("Additional analysis options loaded from snippets_evaluation_config")
        if EVALUATION_WEIGHTS:
            logger.info(f"Custom weights: {EVALUATION_WEIGHTS}")
    except ImportError:
        logger.info("📊 Using default evaluation weights from config.py")
    
    # Perform evaluation with snippet support
    logger.info("Starting evaluation with snippet context...")
    try:
        evaluations = evaluate_answers_with_snippets(evaluation_inputs, config)
        
        logger.info(f"Evaluation completed: {len(evaluations)} items evaluated")
        
        # Log evaluation summary
        if evaluations:
            scores = [eval.weighted_total for eval in evaluations]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            logger.info(f"Score statistics:")
            logger.info(f"  - Average weighted score: {avg_score:.3f}")
            logger.info(f"  - Score range: {min_score:.3f} - {max_score:.3f}")
            
            # Log individual dimension averages if available
            if hasattr(evaluations[0], 'scores') and evaluations[0].scores:
                dimension_averages = {}
                for dimension in evaluations[0].scores.keys():
                    if hasattr(dimension, 'value'):
                        dim_scores = [eval.scores.get(dimension, 0) for eval in evaluations if hasattr(eval, 'scores')]
                        if dim_scores:
                            dimension_averages[dimension.value] = sum(dim_scores) / len(dim_scores)
                    else:
                        # Handle string dimension keys (fallback case)
                        dim_scores = [eval.scores.get(dimension, 0) for eval in evaluations if hasattr(eval, 'scores')]
                        if dim_scores:
                            dimension_averages[str(dimension)] = sum(dim_scores) / len(dim_scores)
                
                if dimension_averages:
                    logger.info(f"Dimension averages:")
                    for dim, avg in dimension_averages.items():
                        logger.info(f"  - {dim}: {avg:.3f}")
            
            # Log snippet grounding scores
            grounding_scores = [getattr(eval, 'snippet_grounding_score', 0) for eval in evaluations 
                              if hasattr(eval, 'snippet_grounding_score') and getattr(eval, 'snippet_grounding_score', None) is not None]
            if grounding_scores:
                avg_grounding = sum(grounding_scores) / len(grounding_scores)
                logger.info(f"  - Average snippet grounding: {avg_grounding:.3f}")
            
            # Log confidence scores if available
            confidence_scores = []
            for eval in evaluations:
                if hasattr(eval, 'confidence_scores') and eval.confidence_scores:
                    eval_confidences = [conf for conf in eval.confidence_scores.values() if isinstance(conf, (int, float))]
                    if eval_confidences:
                        confidence_scores.extend(eval_confidences)
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                logger.info(f"  - Average evaluation confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    # Generate comprehensive report
    logger.info("Generating evaluation report...")
    try:
        report_path = Path(output_dir) / "evaluation_report.md"
        report_content = generate_summary_report(evaluations)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Save detailed results as JSON
        json_path = Path(output_dir) / "detailed_results.json"
        detailed_results = []
        
        for i, evaluation in enumerate(evaluations):
            # Handle both proper evaluation_system and fallback result formats
            system_type_value = "unknown"
            if hasattr(evaluation, 'system_type'):
                if hasattr(evaluation.system_type, 'value'):
                    system_type_value = evaluation.system_type.value
                elif evaluation.system_type is not None:
                    system_type_value = str(evaluation.system_type)
            
            # If system_type is still unknown, try to classify based on evaluation metadata
            if system_type_value in ["unknown", "None", None]:
                # Get the original input for classification
                if i < len(evaluation_inputs):
                    input_item = evaluation_inputs[i]
                    answer = input_item.answer
                    
                    # Simple classification based on answer content
                    rag_indicators = 0
                    agentic_indicators = 0
                    
                    # RAG indicators
                    if 'Sources:' in answer or '#' in answer:
                        rag_indicators += 2
                    if 'Based on my search' in answer or 'database' in answer.lower():
                        rag_indicators += 2
                    if hasattr(input_item, 'source_snippets') and len(input_item.source_snippets) > 0:
                        rag_indicators += 1
                    
                    # Agentic indicators
                    if 'Web Search' in answer or 'web search' in answer.lower():
                        agentic_indicators += 2
                    if len(answer) > 500:  # Longer responses suggest more reasoning
                        agentic_indicators += 1
                    if any(phrase in answer.lower() for phrase in ['i recommend', 'you should', 'best approach', 'steps to take']):
                        agentic_indicators += 1
                    
                    # Classify
                    if rag_indicators > agentic_indicators:
                        system_type_value = "rag_system"
                    elif agentic_indicators > rag_indicators:
                        system_type_value = "agentic_system"
                    else:
                        system_type_value = "hybrid_system"
                        
                    # Debug log for first few items
                    if i < 3:
                        logger.info(f"Classified item {i+1}: {system_type_value} (RAG:{rag_indicators}, Agentic:{agentic_indicators})")
            
            # Extract scores - handle both enum and string keys
            scores_dict = {}
            if hasattr(evaluation, 'scores') and evaluation.scores:
                for dim, score in evaluation.scores.items():
                    if hasattr(dim, 'value'):
                        scores_dict[dim.value] = score
                    else:
                        scores_dict[str(dim)] = score
            
            # Extract justifications - handle both enum and string keys
            justifications_dict = {}
            if hasattr(evaluation, 'justifications') and evaluation.justifications:
                for dim, justification in evaluation.justifications.items():
                    if hasattr(dim, 'value'):
                        justifications_dict[dim.value] = justification
                    else:
                        justifications_dict[str(dim)] = str(justification)
            
            # Extract confidence scores if available
            confidence_dict = {}
            if hasattr(evaluation, 'confidence_scores') and evaluation.confidence_scores:
                for dim, confidence in evaluation.confidence_scores.items():
                    if hasattr(dim, 'value'):
                        confidence_dict[dim.value] = confidence
                    else:
                        confidence_dict[str(dim)] = confidence
            
            result = {
                "item_number": i + 1,
                "query": evaluation_inputs[i].query if i < len(evaluation_inputs) else "",
                "answer": evaluation_inputs[i].answer if i < len(evaluation_inputs) else "",
                "system_type": system_type_value,
                "scores": scores_dict,
                "weighted_total": evaluation.weighted_total,
                "raw_total": getattr(evaluation, 'raw_total', evaluation.weighted_total),
                "snippet_grounding_score": getattr(evaluation, 'snippet_grounding_score', None),
                "snippet_count": len(evaluation.source_snippets) if hasattr(evaluation, 'source_snippets') and evaluation.source_snippets else 0,
                "citation_alignment": getattr(evaluation, 'citation_snippet_alignment', {}),
                "justifications": justifications_dict,
                "confidence_scores": confidence_dict,
                "overall_assessment": getattr(evaluation, 'overall_assessment', "No assessment available"),
                "evaluation_metadata": getattr(evaluation, 'evaluation_metadata', {}),
                "timestamp": getattr(evaluation, 'evaluation_metadata', {}).get("timestamp", datetime.now().isoformat()),
                # Add metadata from Excel if available
                "excel_metadata": getattr(evaluation_inputs[i], 'metadata', {}) if i < len(evaluation_inputs) else {}
            }
            detailed_results.append(result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_summary": {
                    "total_items": len(evaluations),
                    "average_score": sum(eval.weighted_total for eval in evaluations) / len(evaluations) if evaluations else 0,
                    "average_grounding_score": sum(getattr(eval, 'snippet_grounding_score', 0) for eval in evaluations) / len(evaluations) if evaluations else 0,
                    "evaluation_timestamp": datetime.now().isoformat()
                },
                "detailed_results": detailed_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {json_path}")
        
        # Save summary CSV
        csv_path = Path(output_dir) / "evaluation_summary.csv"
        import pandas as pd
        
        csv_data = []
        for i, evaluation in enumerate(evaluations):
            # Handle system type extraction with better classification
            system_type_value = "unknown"
            if hasattr(evaluation, 'system_type'):
                if hasattr(evaluation.system_type, 'value'):
                    system_type_value = evaluation.system_type.value
                elif evaluation.system_type is not None:
                    system_type_value = str(evaluation.system_type)
                    
            # If system_type is still unknown, classify based on content
            if system_type_value in ["unknown", "None", None]:
                if i < len(evaluation_inputs):
                    input_item = evaluation_inputs[i]
                    answer = input_item.answer
                    
                    # Classification logic
                    rag_indicators = 0
                    agentic_indicators = 0
                    
                    # RAG indicators
                    if 'Sources:' in answer or '#' in answer:
                        rag_indicators += 2
                    if 'Based on my search' in answer or 'database' in answer.lower():
                        rag_indicators += 2
                    if hasattr(input_item, 'source_snippets') and len(input_item.source_snippets) > 0:
                        rag_indicators += 1
                    
                    # Agentic indicators
                    if 'Web Search' in answer or 'web search' in answer.lower():
                        agentic_indicators += 2
                    if len(answer) > 500:
                        agentic_indicators += 1
                    if any(phrase in answer.lower() for phrase in ['i recommend', 'you should', 'best approach', 'steps to take']):
                        agentic_indicators += 1
                    
                    # Classify
                    if rag_indicators > agentic_indicators:
                        system_type_value = "rag_system"
                    elif agentic_indicators > rag_indicators:
                        system_type_value = "agentic_system"
                    else:
                        system_type_value = "hybrid_system"
                        
            # Debug: log the first few system types to understand the issue
            if i < 3:
                logger.info(f"Debug item {i+1}: system_type object = {getattr(evaluation, 'system_type', 'missing')}")
                logger.info(f"Debug item {i+1}: system_type_value = {system_type_value}")
            
            # Base row data
            row = {
                "Item": i + 1,
                "Query": (evaluation_inputs[i].query if i < len(evaluation_inputs) else "")[:100] + "...",
                "System_Type": system_type_value,
                "Weighted_Total": f"{evaluation.weighted_total:.3f}",
                "Snippet_Grounding": f"{getattr(evaluation, 'snippet_grounding_score', 0):.3f}",
                "Snippet_Count": len(evaluation.source_snippets) if hasattr(evaluation, 'source_snippets') and evaluation.source_snippets else 0,
            }
            
            # Add dimension scores - handle both enum and string keys
            if hasattr(evaluation, 'scores') and evaluation.scores:
                for dim, score in evaluation.scores.items():
                    if hasattr(dim, 'value'):
                        col_name = dim.value.replace('_', ' ').title()
                        row[col_name] = f"{score:.2f}"
                    else:
                        col_name = str(dim).replace('_', ' ').title()
                        row[col_name] = f"{score:.2f}"
            
            # Add Excel metadata if available - with better error handling
            metadata_added = False
            if i < len(evaluation_inputs) and hasattr(evaluation_inputs[i], 'metadata'):
                metadata = evaluation_inputs[i].metadata
                if metadata:  # Check if metadata dict is not empty
                    row.update({
                        "Status": metadata.get('status', ''),
                        "Fallback_Used": metadata.get('fallback_used', ''),
                        "Search_Method": metadata.get('search_method', ''),
                        "Has_Error": 'Yes' if metadata.get('error', '') not in ['None', '', 'N/A'] else 'No'
                    })
                    metadata_added = True
                    
            # If no metadata was added, add empty columns to maintain structure
            if not metadata_added:
                row.update({
                    "Status": "",
                    "Fallback_Used": "",
                    "Search_Method": "",
                    "Has_Error": "No"
                })
                
            # Debug: log metadata for first few items
            if i < 3:
                logger.info(f"Debug item {i+1}: metadata = {getattr(evaluation_inputs[i], 'metadata', 'missing') if i < len(evaluation_inputs) else 'no input'}")
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to: {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    logger.info("=== Evaluation Complete ===")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("Check the markdown report for detailed analysis including snippet grounding scores.")


if __name__ == "__main__":
    main()
