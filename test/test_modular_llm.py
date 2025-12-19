#!/usr/bin/env python3
"""
Test script for the new modular LLM system.

This script demonstrates how to use the modular LLM provider system
to switch between Azure OpenAI and Google Gemini.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def test_azure_provider():
    """Test Azure OpenAI provider."""
    print("=== Testing Azure OpenAI Provider ===")
    
    # Load environment variables
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    try:
        from evaluation_system.advanced_config import create_default_config
        from evaluation_system.enums import LLMProvider
        from evaluation_system.llm_client import get_llm_client, LLMMessage
        
        # Create configuration with Azure provider
        config = create_default_config()
        config.llm_provider = LLMProvider.AZURE
        
        # Get Azure client
        client = get_llm_client(config)
        print(f"Created client: {type(client).__name__}")
        
        # Test connection
        success = client.test_connection()
        print(f"Connection test: {'✓ PASSED' if success else '✗ FAILED'}")
        
        if success:
            # Test a simple evaluation
            test_messages = [
                LLMMessage(
                    role="system", 
                    content="You are an evaluator. Rate the following answer on a scale of 1-5 for accuracy. Respond with JSON: {\"score\": 4, \"reason\": \"explanation\"}"
                ),
                LLMMessage(
                    role="user", 
                    content="Question: What is 2+2? Answer: The result is 4."
                )
            ]
            
            response = client.get_llm_response(test_messages)
            print(f"Test response: {response[:100]}...")
            
    except Exception as e:
        print(f"Azure test failed: {e}")
        import traceback
        traceback.print_exc()


def test_gemini_provider():
    """Test Google Gemini provider (requires google-genai package)."""
    print("\n=== Testing Google Gemini Provider ===")
    
    try:
        from evaluation_system.advanced_config import create_default_config
        from evaluation_system.enums import LLMProvider
        from evaluation_system.llm_client import get_llm_client, LLMMessage
        
        # Create configuration with Gemini provider
        config = create_default_config()
        config.llm_provider = LLMProvider.GEMINI
        
        # Get Gemini client
        client = get_llm_client(config)
        print(f"Created client: {type(client).__name__}")
        
        # Test connection
        success = client.test_connection()
        print(f"Connection test: {'✓ PASSED' if success else '✗ FAILED'}")
        
        if success:
            # Test a simple evaluation
            test_messages = [
                LLMMessage(
                    role="system", 
                    content="You are an evaluator. Rate the following answer on a scale of 1-5 for accuracy. Respond with JSON: {\"score\": 4, \"reason\": \"explanation\"}"
                ),
                LLMMessage(
                    role="user", 
                    content="Question: What is 2+2? Answer: The result is 4."
                )
            ]
            
            response = client.get_llm_response(test_messages)
            print(f"Test response: {response[:100]}...")
            
    except ImportError as e:
        print(f"Gemini test skipped - missing dependency: {e}")
        print("To test Gemini, install: pip install google-genai")
    except Exception as e:
        print(f"Gemini test failed: {e}")
        if "authentication" in str(e).lower() or "credentials" in str(e).lower() or "permission" in str(e).lower():
            print("💡 This may be due to authentication. Ensure you're logged in with: gcloud auth application-default login")
        else:
            import traceback
            traceback.print_exc()


def test_provider_switching():
    """Test switching between providers using environment variables."""
    print("\n=== Testing Provider Switching ===")
    
    try:
        from evaluation_system.advanced_config import create_default_config
        from evaluation_system.enums import LLMProvider
        
        # Test environment variable configuration
        original_provider = os.getenv("LLM_PROVIDER")
        
        # Test Azure configuration
        os.environ["LLM_PROVIDER"] = "AZURE"
        config_azure = create_default_config()
        print(f"Environment Azure config: {config_azure.llm_provider}")
        
        # Test Gemini configuration
        os.environ["LLM_PROVIDER"] = "GEMINI"
        config_gemini = create_default_config()
        print(f"Environment Gemini config: {config_gemini.llm_provider}")
        
        # Restore original
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
            
    except Exception as e:
        print(f"Provider switching test failed: {e}")


def display_config_info():
    """Display configuration information."""
    print("\n=== Configuration Information ===")
    
    try:
        from evaluation_system.advanced_config import create_default_config
        
        config = create_default_config()
        print(f"Default provider: {config.llm_provider}")
        print(f"Config version: {config.config_version}")
        
        # Azure config
        azure_config = config.azure_config
        print(f"\nAzure Config:")
        print(f"  Temperature: {azure_config.temperature}")
        print(f"  Max tokens: {azure_config.max_tokens}")
        print(f"  Max retries: {azure_config.max_retries}")
        print(f"  Endpoint: {azure_config.endpoint or 'from environment'}")
        
        # Gemini config
        gemini_config = config.gemini_config
        print(f"\nGemini Config:")
        print(f"  Model: {gemini_config.model_name}")
        print(f"  Temperature: {gemini_config.temperature}")
        print(f"  Max tokens: {gemini_config.max_output_tokens}")
        print(f"  Max retries: {gemini_config.max_retries}")
        
    except Exception as e:
        print(f"Config display failed: {e}")


def main():
    """Main test function."""
    print("Testing Modular LLM Provider System")
    print("=" * 50)
    
    # Display configuration
    display_config_info()
    
    # Test provider switching
    test_provider_switching()
    
    # Test Azure provider (default)
    test_azure_provider()
    
    # Test Gemini provider
    test_gemini_provider()
    
    print("\n" + "=" * 50)
    print("Modular LLM testing complete!")
    print("\nTo switch providers:")
    print("1. Set environment variable: LLM_PROVIDER=AZURE or LLM_PROVIDER=GEMINI")
    print("2. Or programmatically: config.llm_provider = LLMProvider.GEMINI")


if __name__ == "__main__":
    main()
