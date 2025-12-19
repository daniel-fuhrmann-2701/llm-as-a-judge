"""
Modular LLM client for the RAG vs Agentic AI Evaluation Framework.

This module provides a unified interface for multiple LLM providers:
- Azure OpenAI
- Google Gemini

Features:
- Abstract base class for consistent interface
- Provider-specific implementations
- Factory function for dynamic client selection
- Retry logic and error handling
- Response validation
"""

import os
import json
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

import certifi
import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, ClientSecretCredential, get_bearer_token_provider
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .advanced_config import AdvancedEvalConfig, LLMProvider
from .enums import EvaluationDimension, SystemType
from .models import AnswerEvaluation, EvaluationRubric, SourceSnippet
from .templates import load_system_prompt, render_template, SNIPPET_EVALUATION_PROMPT, STANDARD_EVALUATION_PROMPT, ENHANCED_SNIPPET_EVALUATION_PROMPT

@dataclass
class LLMMessage:
    """Unified message format for all LLM providers."""
    role: str  # "system", "user", "assistant"
    content: str


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: AdvancedEvalConfig):
        """Initialize the client with configuration."""
        self.config = config
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """Get logger instance."""
        import logging
        return logging.getLogger(__name__)
    
    @abstractmethod
    def get_llm_response(self, messages: List[LLMMessage]) -> str:
        """
        Get response from LLM provider.
        
        Args:
            messages: List of LLMMessage objects
            
        Returns:
            Raw response content as string
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the LLM provider."""
        pass
    
    def get_llm_response_with_retries(self, messages: List[LLMMessage]) -> str:
        """Get LLM response with retry logic."""
        current_config = self.config.get_current_provider_config()
        
        @retry(
            stop=stop_after_attempt(current_config.max_retries),
            wait=wait_exponential(multiplier=current_config.retry_delay, min=1, max=60)
        )
        def _call_with_retry():
            return self.get_llm_response(messages)
        
        return _call_with_retry()


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI implementation of the LLM client."""
    
    def __init__(self, config: AdvancedEvalConfig):
        """Initialize Azure OpenAI client."""
        super().__init__(config)
        self.client = self._create_client()
        
    def _create_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client with authentication."""
        # Load environment variables
        load_dotenv()
        
        # Set NO_PROXY for local development
        os.environ["NO_PROXY"] = "localhost,127.0.1,::1"
        
        azure_config = self.config.azure_config
        
        # Use config values or fall back to environment variables
        azure_endpoint = azure_config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "https://begobaiatest.openai.azure.com/")
        deployment_name = azure_config.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_version = azure_config.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        # Service principal credentials
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        if not all([tenant_id, client_id, client_secret]):
            raise ValueError("Missing required Azure environment variables: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        
        try:
            self.logger.info(f"Using Azure OpenAI with endpoint: {azure_endpoint}")
            self.logger.info(f"Deployment: {deployment_name}")
            self.logger.info(f"API Version: {api_version}")
            
            # Initialize Azure credential with service principal
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            
            # Create token provider for Azure OpenAI
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
            
            self.logger.info("Azure OpenAI client created successfully")
            self.deployment_name = deployment_name
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create Azure OpenAI client: {e}")
            raise
    
    def get_llm_response(self, messages: List[LLMMessage]) -> str:
        """Get response from Azure OpenAI."""
        azure_config = self.config.azure_config
        
        # Convert LLMMessage to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=openai_messages,
                temperature=azure_config.temperature,
                top_p=azure_config.top_p,
                max_tokens=azure_config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            self.logger.debug(f"Azure OpenAI Response: {content[:200]}...")
            return content
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI API call failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test Azure OpenAI connection."""
        try:
            # Use simple test without JSON response format for connection test
            test_messages = [
                {"role": "user", "content": "Test connection. Reply with: OK"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=test_messages,
                max_tokens=10,
                temperature=0.0
            )
            
            self.logger.info(f"Azure OpenAI connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Azure OpenAI connection test failed: {e}")
            return False


class GeminiClient(BaseLLMClient):
    """Google Gemini implementation of the LLM client."""
    
    def __init__(self, config: AdvancedEvalConfig):
        """Initialize Gemini client."""
        super().__init__(config)
        # Configure the client using your working approach
        self.client = self._create_client()
        
    def _create_client(self):
        """Configure google-genai client based on your working implementation."""
        try:
            import certifi
            from google import genai
            from google.genai.types import HttpOptions
            
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-central2")
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if project:             
                # Set required environment variables for Vertex AI
                os.environ["GOOGLE_CLOUD_PROJECT"] = project
                os.environ["GOOGLE_CLOUD_LOCATION"] = location
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
                
                # Set certificate bundle for SSL verification
                os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
                os.environ["SSL_CERT_FILE"] = certifi.where()
                
                # Create client with HttpOptions
                client = genai.Client(http_options=HttpOptions(api_version="v1"))
                self.logger.info("Google Gemini Vertex AI client created successfully.")
                return client
                
            elif api_key:
                # Configure for Google AI API
                self.logger.info("Configuring Gemini for Google AI API with API key")
                import google.generativeai as genai_std
                genai_std.configure(api_key=api_key)
                # For API key, we'll use the standard google-generativeai client
                self.logger.info("Google Gemini API client configured successfully.")
                return None  # Will use standard genai_std methods
            else:
                raise ValueError(
                    "Gemini provider requires authentication. Please configure either:\n"
                    "1. Vertex AI: Run 'gcloud auth application-default login' and set GOOGLE_CLOUD_PROJECT.\n"
                    "2. Google AI API: Set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                )

        except ImportError as e:
            if "google" in str(e):
                raise ImportError("Google GenAI library not installed. Install with: pip install google-genai")
            elif "certifi" in str(e):
                raise ImportError("Certifi library not installed. Install with: pip install certifi")
            else:
                raise e
        except Exception as e:
            self.logger.error(f"Failed to create Gemini client: {e}")
            raise
    
    def get_llm_response(self, messages: List[LLMMessage]) -> str:
        """Get response from Google Gemini using your working implementation."""
        try:
            from google.genai import types
            
            gemini_config = self.config.gemini_config
            
            # Convert LLMMessage to Gemini format using your approach
            combined_content = ""
            for msg in messages:
                if msg.role == "system":
                    combined_content += f"System: {msg.content}\n\n"
                elif msg.role == "user":
                    combined_content += f"User: {msg.content}\n\n"
                elif msg.role == "assistant":
                    combined_content += f"Assistant: {msg.content}\n\n"
            
            # Create contents in your format
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=combined_content)]
                )
            ]
            
            # Configure safety settings using your format
            safety_settings = []
            if gemini_config.disable_safety_filters:
                safety_settings = [
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                ]
            
            # Generate content config using your format
            generate_content_config = types.GenerateContentConfig(
                temperature=gemini_config.temperature,
                top_p=gemini_config.top_p,
                max_output_tokens=gemini_config.max_output_tokens,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=getattr(gemini_config, 'thinking_budget', 0),
                ),
            )
            
            # Get response from Gemini using your streaming approach
            response_chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=gemini_config.model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_chunks.append(chunk.text)
            
            full_response = "".join(response_chunks)
            self.logger.debug(f"Gemini Response: {full_response[:200]}...")
            
            # Extract JSON from markdown code blocks if present
            cleaned_response = self._extract_json_from_response(full_response)
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from Gemini response, handling markdown code blocks."""
        import re
        import json
        
        # First, try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, response, re.IGNORECASE)
        
        if matches:
            # Use the first JSON block found
            json_content = matches[0].strip()
            self.logger.debug(f"Extracted JSON from code block: {json_content[:100]}...")
            
            # Validate it's proper JSON
            try:
                json.loads(json_content)
                return json_content
            except json.JSONDecodeError:
                self.logger.warning("Extracted content is not valid JSON, falling back to original response")
        
        # If no code blocks or invalid JSON, try to find JSON-like content
        # Look for content that starts with { and ends with }
        json_like_pattern = r'\{[\s\S]*\}'
        json_matches = re.findall(json_like_pattern, response)
        
        if json_matches:
            for json_candidate in json_matches:
                try:
                    json.loads(json_candidate)
                    self.logger.debug(f"Found valid JSON content: {json_candidate[:100]}...")
                    return json_candidate
                except json.JSONDecodeError:
                    continue
        
        # If all else fails, return the original response
        self.logger.warning(f"Could not extract valid JSON from response: {response[:200]}...")
        return response
    
    def test_connection(self) -> bool:
        """Test Gemini connection."""
        try:
            # Simple test using the same approach as get_llm_response
            test_messages = [LLMMessage(role="user", content="Test connection. Reply with one word: OK")]
            response = self.get_llm_response(test_messages)
            self.logger.info("Gemini connection test successful")
            return "OK" in response.upper()
        except Exception as e:
            self.logger.error(f"Gemini connection test failed: {e}")
            return False
            self.logger.info("Gemini connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Gemini connection test failed: {e}")
            return False


def get_llm_client(config: Optional[AdvancedEvalConfig] = None) -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client based on configuration.
    
    Args:
        config: Optional configuration. If None, creates default config.
        
    Returns:
        Appropriate LLM client instance
    """
    if config is None:
        from .advanced_config import create_default_config
        config = create_default_config()
    
    if config.llm_provider == LLMProvider.AZURE:
        return AzureOpenAIClient(config)
    elif config.llm_provider == LLMProvider.GEMINI:
        return GeminiClient(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


# Legacy compatibility functions


# Legacy compatibility functions
def create_openai_client() -> openai.OpenAI:
    """
    Legacy function for backward compatibility.
    Creates Azure OpenAI client using the new modular system.
    """
    from .advanced_config import create_default_config
    config = create_default_config()
    client = get_llm_client(config)
    
    if isinstance(client, AzureOpenAIClient):
        return client.client
    else:
        raise ValueError("Legacy function only supports Azure OpenAI")


def get_model_name(config) -> str:
    """
    Legacy function for backward compatibility.
    Get the appropriate model name based on the client type.
    """
    # Handle both old EvalConfig and new AdvancedEvalConfig
    if hasattr(config, 'azure_config'):
        # New AdvancedEvalConfig
        azure_config = config.azure_config
        deployment_name = azure_config.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    else:
        # Legacy config
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    return deployment_name


def call_llm_evaluator(
    query: str, 
    answer: str, 
    config,
    system_type: Optional[SystemType] = None,
    snippets: Optional[list[SourceSnippet]] = None
) -> Optional[AnswerEvaluation]:
    """
    Legacy function for backward compatibility.
    Enhanced LLM evaluation with academic rigor and snippet support.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.now().isoformat()
    
    # Convert legacy config to new config if needed
    if not hasattr(config, 'llm_provider'):
        from .advanced_config import create_default_config
        new_config = create_default_config()
        # Copy relevant attributes if they exist
        if hasattr(config, 'temperature'):
            new_config.azure_config.temperature = config.temperature
        if hasattr(config, 'max_retries'):
            new_config.azure_config.max_retries = config.max_retries
        if hasattr(config, 'dimensions'):
            new_config.evaluation.dimensions = config.dimensions
        if hasattr(config, 'rubrics'):
            new_config.rubrics = config.rubrics
        if hasattr(config, 'confidence_threshold'):
            new_config.evaluation.confidence_threshold = config.confidence_threshold
        config = new_config
    
    # Choose appropriate template based on whether snippets are provided
    if snippets and len(snippets) > 0:
        # Use enhanced template for better evaluation
        prompt_template = ENHANCED_SNIPPET_EVALUATION_PROMPT
        logger.debug(f"Using snippet-enhanced evaluation template with {len(snippets)} snippets")
    else:
        prompt_template = STANDARD_EVALUATION_PROMPT
        logger.debug("Using standard evaluation template")
    
    # Render the prompt with query, answer, snippets, and configuration
    rendered_prompt = render_template(
        prompt_template,
        query=query,
        answer=answer,
        snippets=snippets or [],
        dimensions=config.evaluation.dimensions,
        rubrics=config.rubrics
    )
    
    # Create LLM messages
    messages = [LLMMessage(role="system", content=rendered_prompt)]
    
    # Get LLM client
    client = get_llm_client(config)
    
    for attempt in range(1, config.get_current_provider_config().max_retries + 1):
        try:
            content = client.get_llm_response(messages)
            logger.debug(f"LLM Response (attempt {attempt}): {content[:200]}...")
            
            # DEBUG: Log the full response for troubleshooting
            logger.info(f"=== FULL LLM RESPONSE (Attempt {attempt}) ===")
            logger.info(content)
            logger.info("=== END RESPONSE ===")
            
            data = json.loads(content)
            
            # Enhanced validation
            if not validate_evaluation_response(data, config.evaluation.dimensions):
                raise ValueError("Invalid evaluation response format")
            
            # Convert string keys back to enum dimensions
            scores = {EvaluationDimension(k): v for k, v in data["scores"].items()}
            justifications = {EvaluationDimension(k): v for k, v in data["justifications"].items()}
            confidence_scores = {EvaluationDimension(k): v for k, v in data["confidence_scores"].items()}
            
            # Calculate weighted score
            weighted_total = calculate_weighted_score(scores, config.rubrics)
            raw_total = sum(scores.values())
            
            # Check confidence threshold
            avg_confidence = statistics.mean(confidence_scores.values())
            if avg_confidence < config.evaluation.confidence_threshold:
                logger.warning(f"Low confidence evaluation: {avg_confidence:.2f}")
            
            # Build evaluation metadata
            model_name = get_model_name(config)
            evaluation_metadata = {
                "model_used": model_name,
                "tokens_used": 0,  # Token counting not implemented for all providers yet
                "timestamp": timestamp,
                "system_type": system_type.value if system_type else None,
                "has_snippets": bool(snippets and len(snippets) > 0),
                "provider": config.llm_provider.value
            }
            
            # Add snippet analysis if present
            if data.get("snippet_analysis"):
                evaluation_metadata["snippet_analysis"] = data["snippet_analysis"]
            
            return AnswerEvaluation(
                answer=answer,
                system_type=system_type,
                scores=scores,
                justifications=justifications,
                confidence_scores=confidence_scores,
                weighted_total=weighted_total,
                raw_total=raw_total,
                evaluation_metadata=evaluation_metadata,
                overall_assessment=data.get("overall_assessment", ""),
                comparative_notes=data.get("academic_recommendation", ""),
                source_snippets=snippets,
                snippet_grounding_score=None,  # Will be calculated separately
                citation_snippet_alignment=None  # Will be calculated separately
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt} failed - JSON decode error: {e}")
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed - {type(e).__name__}: {e}")
            
        if attempt < config.get_current_provider_config().max_retries:
            logger.info(f"Retrying evaluation (attempt {attempt + 1}/{config.get_current_provider_config().max_retries})")
    
    logger.error("LLM evaluation failed after all retries")
    return None


def test_llm_connection(config=None) -> bool:
    """
    Legacy function for backward compatibility.
    Test connection to the LLM API with a simple query.
    """
    try:
        if config is None:
            from .advanced_config import create_default_config
            config = create_default_config()
        elif not hasattr(config, 'llm_provider'):
            # Convert legacy config
            from .advanced_config import create_default_config
            config = create_default_config()
        
        client = get_llm_client(config)
        return client.test_connection()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"LLM connection test failed: {e}")
        return False


# Import utility functions
from .utils import calculate_weighted_score, validate_evaluation_response


