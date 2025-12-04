"""
Setup script to fix import issues and run tests.
"""
import sys
import os
from pathlib import Path

def setup_environment():
    """Set up the Python environment for        elif choice == "5":
            print("\n🔧 Running Azure OpenAI diagnostics...")
            run_azure_diagnostics()
            
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice. Please enter 1-6.")r imports."""
    # Get the agentic_system directory
    current_dir = Path(__file__).parent.absolute()
    
    # Add current directory to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add parent directory to Python path for package imports
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    print(f"✅ Environment set up")
    print(f"📁 Current directory: {current_dir}")
    print(f"🐍 Python path updated with: {current_dir}")
    
    return current_dir

def test_imports():
    """Test if all imports work correctly."""
    print("\n🔍 Testing imports...")
    
    try:
        from agents.rag_agent import RAGAgent
        print("✅ RAGAgent imported successfully")
    except ImportError as e:
        print(f"❌ RAGAgent import failed: {e}")
        return False
    
    try:
        from agents.topic_identification_agent import TopicIdentificationAgent
        print("✅ TopicIdentificationAgent imported successfully")
    except ImportError as e:
        print(f"❌ TopicIdentificationAgent import failed: {e}")
        return False
    
    try:
        from core.base import Task
        print("✅ Task imported successfully")
    except ImportError as e:
        print(f"❌ Task import failed: {e}")
        return False
    
    try:
        from enums import AgentType
        print("✅ AgentType imported successfully")
    except ImportError as e:
        print(f"❌ AgentType import failed: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

def run_system_analysis():
    """Run system analysis to understand the agentic system structure."""
    try:
        import analyze_system
        print("✅ System analysis script imported successfully")
        analyze_system.main()
    except ImportError as e:
        print(f"❌ Failed to import system analysis script: {e}")
    except Exception as e:
        print(f"❌ System analysis failed: {e}")
        import traceback
        traceback.print_exc()

def run_excel_evaluation():
    """Run evaluation using questions from Excel file."""
    try:
        import test_with_excel_questions
        print("✅ Excel evaluation script imported successfully")
        
        # Run the async main function properly
        import asyncio
        asyncio.run(test_with_excel_questions.main())
        
    except ImportError as e:
        print(f"❌ Failed to import Excel evaluation script: {e}")
    except Exception as e:
        print(f"❌ Excel evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def run_azure_diagnostics():
    """Run Azure OpenAI diagnostics."""
    try:
        import check_azure_config
        print("✅ Azure diagnostics script imported successfully")
        check_azure_config.main()
    except ImportError as e:
        print(f"❌ Failed to import diagnostics script: {e}")
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()

def run_simple_evaluation():
    """Run simple RAG evaluation."""
    try:
        import simple_evaluation
        print("✅ Simple evaluation script imported successfully")
        
        # Run the async main function properly
        import asyncio
        asyncio.run(simple_evaluation.main())
        
    except ImportError as e:
        print(f"❌ Failed to import simple evaluation script: {e}")
    except Exception as e:
        print(f"❌ Simple evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def show_menu():
    """Show the main menu for testing options."""
    print("\n🎯 TESTING OPTIONS")
    print("=" * 30)
    print("1. Run system analysis (understand structure)")
    print("2. Run Excel evaluation (test with questions)")
    print("3. Run original integration test")
    print("4. Run simple RAG evaluation (recommended)")
    print("5. Check Azure OpenAI configuration")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    return choice

def main():
    """Main setup and test function."""
    print("🚀 AGENTIC SYSTEM TESTING SUITE")
    print("=" * 50)
    
    # Setup environment
    current_dir = setup_environment()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your environment.")
        return
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            print("\n🔍 Running system analysis...")
            run_system_analysis()
            
        elif choice == "2":
            print("\n📊 Running Excel evaluation...")
            run_excel_evaluation()
            
        elif choice == "3":
            print("\n🎯 Running original integration test...")
            try:
                import test_simple_integration
                print("✅ Test script imported successfully")
                test_simple_integration.main()
            except ImportError as e:
                print(f"❌ Failed to import test script: {e}")
            except Exception as e:
                print(f"❌ Test execution failed: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == "4":
            print("\n� Running simple RAG evaluation...")
            run_simple_evaluation()
            
        elif choice == "5":
            print("\n�👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
