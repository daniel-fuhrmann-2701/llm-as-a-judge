"""
Robust evaluation script that handles Azure         return None
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return NoneI initialization properly.
"""
import sys
import os
import pandas as pd
from pathlib import Path
import traceback
from datetime import datetime
import asyncio
import uuid

def setup_environment():
    """Set up the Python environment for proper imports."""
    current_dir = Path(__file__).parent.absolute()
    
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    print(f"Environment set up")
    print(f"Current directory: {current_dir}")
    return current_dir

def load_excel_questions(excel_path):
    """Load questions from the Excel file."""
    try:
        print(f"Loading questions from: {excel_path}")
        df = pd.read_excel(excel_path)
        
        print(f"Excel file loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nQuestions to test:")
        
        # Find question column
        question_col = None
        for col in df.columns:
            if 'question' in col.lower():
                question_col = col
                break
        
        if question_col:
            for i, q in enumerate(df[question_col].dropna(), 1):
                print(f"  {i}. {q}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return None

async def test_autonomous_agent_fallback(question):
    """Test with Autonomous Agent as fallback for web search."""
    try:
        from agents.autonomous_agent import AutonomousAgent
        from core.base import Task, Priority
        
        print(f"\n🌐 Testing Autonomous Agent fallback with: {question}")
        
        # Initialize Autonomous Agent
        autonomous_agent = AutonomousAgent(agent_id=f"fallback_auto_{uuid.uuid4().hex[:8]}")
        
        # Initialize and check status
        await autonomous_agent.initialize()
        print(f"Autonomous Agent initialized successfully")
        
        # Create a task for web search
        task = Task(
            id=f"fallback_{uuid.uuid4().hex[:8]}",
            name="Web Search Fallback",
            description=f"Search for information about: {question}",
            input_data={'query': question},
            priority=Priority.MEDIUM
        )
        
        # Process the task
        print(f"Processing web search...")
        response = await autonomous_agent.process(task)
        
        print(f"Response success: {response.success}")
        
        if response.success:
            result_data = response.data.get('result', {})
            web_search_results = result_data.get('web_search', [])
            synthesis = result_data.get('synthesis', 'No synthesis available')
            
            print(f"Found {len(web_search_results) if web_search_results else 0} web search results")
            if synthesis and synthesis != "No content to synthesize.":
                print(f"Synthesis: {synthesis[:200]}...")
            
            # Prepare the answer from web search results
            if web_search_results and len(web_search_results) > 0:
                # Create a summary from search results
                answer_parts = [f"Based on web search results for '{question}':"]
                for i, result in enumerate(web_search_results[:3], 1):
                    if hasattr(result, 'title') and hasattr(result, 'snippet'):
                        answer_parts.append(f"{i}. {result.title}: {result.snippet}")
                
                # Only add synthesis if it's actually useful content
                if (synthesis and 
                    synthesis not in ["No content to synthesize.", "Content synthesis failed.", "No synthesis available"] and
                    "Content synthesis error" not in synthesis):
                    answer_parts.append(f"\nSynthesis: {synthesis}")
                
                web_answer = "\n".join(answer_parts)
            else:
                web_answer = "Web search did not return relevant results."
            
            return {
                'question': question,
                'answer': web_answer,
                'database_used': 'Web Search (Autonomous Agent)',
                'search_method': 'autonomous_agent_web_search',
                'status': 'success',
                'source': 'web_fallback'
            }
        else:
            print(f"Autonomous Agent processing failed: {response.error_message}")
            return {
                'question': question,
                'answer': None,
                'status': f'failed: {response.error_message}',
                'error': response.error_message,
                'source': 'web_fallback'
            }
        
        # Cleanup
        await autonomous_agent.shutdown()
            
    except Exception as e:
        print(f"Error in Autonomous Agent fallback: {e}")
        traceback.print_exc()
        return {
            'question': question,
            'answer': None,
            'status': f'error: {str(e)}',
            'error': str(e),
            'source': 'web_fallback'
        }

async def test_rag_agent_only(question):
    """Test with RAG Agent directly, bypassing complex topic identification."""
    try:
        from agents.rag_agent import RAGAgent
        from core.base import Task, Priority
        
        print(f"\nTesting RAG Agent directly with: {question}")
        
        # Initialize RAG Agent
        rag_agent = RAGAgent(agent_id=f"direct_rag_{uuid.uuid4().hex[:8]}")
        
        # Initialize and check status
        await rag_agent.initialize()
        print(f"RAG Agent initialized successfully")
        
        # Check knowledge base status
        kb_stats = rag_agent.get_knowledge_base_stats()
        print(f"Knowledge base status: {kb_stats}")
        
        # Create a simple task
        task = Task(
            id=f"test_{uuid.uuid4().hex[:8]}",
            name="Direct RAG Test",
            description=question,
            input_data={'query': question},
            priority=Priority.MEDIUM
        )
        
        # Process the task
        print(f"Processing query...")
        response = await rag_agent.process(task)
        
        print(f"Response success: {response.success}")
        print(f"Response data keys: {list(response.data.keys()) if response.data else 'None'}")
        
        if response.success:
            answer = response.data.get('answer', response.data.get('response', 'No answer found'))
            database_used = response.data.get('database_used', 'Unknown')
            search_method = response.data.get('search_method', 'Unknown')
            
            print(f"Answer: {answer[:200]}...")
            print(f"Database: {database_used}")
            print(f"Method: {search_method}")
            
            return {
                'question': question,
                'answer': answer,
                'database_used': database_used,
                'search_method': search_method,
                'status': 'success'
            }
        else:
            print(f"RAG processing failed: {response.error_message}")
            return {
                'question': question,
                'answer': None,
                'status': f'failed: {response.error_message}',
                'error': response.error_message
            }
            
    except Exception as e:
        print(f"Error in RAG test: {e}")
        traceback.print_exc()
        return {
            'question': question,
            'answer': None,
            'status': f'error: {str(e)}',
            'error': str(e)
        }

async def run_simple_evaluation():
    """Run a simple evaluation focusing on RAG functionality."""
    print("SIMPLE RAG EVALUATION")
    print("=" * 60)
    
    # Setup environment
    current_dir = setup_environment()
    
    # Load Excel file
    excel_path = "\\\\dnsbego.de\\dfsbego\\home04\\FuhrmannD\\Documents\\01_Trainee\\Master\\Thesis\\code\\it_governance_questions.xlsx"
    df = load_excel_questions(excel_path)
    
    if df is None:
        print("Cannot proceed without Excel data")
        return
    
    # Get questions
    question_col = None
    for col in df.columns:
        if 'question' in col.lower():
            question_col = col
            break
    
    if not question_col:
        print("No question column found")
        return
    
    questions = df[question_col].dropna().tolist()
    print(f"\nTesting {len(questions)} questions...")
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"{'='*60}")
        
        # First try RAG agent
        result = await test_rag_agent_only(question)
        
        # Check if we got "I don't know" or similar responses
        answer = result.get('answer', '')
        if (answer and 
            (answer.strip().lower() in ['ich weiß es nicht', "i don't know", 'i don\'t know'] or
             'I don\'t know' in answer or 
             'ich weiß es nicht' in answer.lower())):
            
            print(f"\n🔄 RAG returned 'I don't know' - trying web search fallback...")
            
            # Try autonomous agent as fallback
            web_result = await test_autonomous_agent_fallback(question)
            
            if web_result['status'] == 'success' and web_result.get('answer'):
                # Combine the results
                combined_answer = f"Internal knowledge base: {answer}\n\n--- Web Search Fallback ---\n{web_result['answer']}"
                result.update({
                    'answer': combined_answer,
                    'database_used': f"{result.get('database_used', 'Unknown')} + Web Search",
                    'search_method': f"{result.get('search_method', 'Unknown')} + autonomous_agent_fallback",
                    'fallback_used': True,
                    'web_fallback_status': 'success'
                })
            else:
                # Web search also failed
                result.update({
                    'fallback_used': True,
                    'web_fallback_status': 'failed',
                    'web_fallback_error': web_result.get('error', 'Unknown error')
                })
        else:
            result['fallback_used'] = False
        
        results.append(result)
        
        # Small delay between questions
        await asyncio.sleep(2)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text results
    results_file = current_dir / f"simple_evaluation_results_{timestamp}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("SIMPLE RAG EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total questions: {len(results)}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"QUESTION {i}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Answer: {result.get('answer', 'N/A')}\n")
            f.write(f"Database: {result.get('database_used', 'N/A')}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Fallback Used: {result.get('fallback_used', False)}\n")
            if result.get('fallback_used'):
                f.write(f"Web Fallback Status: {result.get('web_fallback_status', 'N/A')}\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            f.write("\n")
    
    # Save Excel results
    excel_results = []
    for i, result in enumerate(results, 1):
        excel_results.append({
            'Question_ID': i,
            'Question': result['question'],
            'Answer': result.get('answer', 'N/A'),
            'Database_Used': result.get('database_used', 'N/A'),
            'Search_Method': result.get('search_method', 'N/A'),
            'Status': result['status'],
            'Fallback_Used': result.get('fallback_used', False),
            'Web_Fallback_Status': result.get('web_fallback_status', 'N/A'),
            'Error': result.get('error', 'None')
        })
    
    results_df = pd.DataFrame(excel_results)
    excel_file = current_dir / f"simple_evaluation_results_{timestamp}.xlsx"
    results_df.to_excel(excel_file, index=False)
    
    print(f"\nEVALUATION COMPLETED")
    print("=" * 40)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)}")
    print(f"Results saved to: {results_file.name}")
    print(f"Excel saved to: {excel_file.name}")

async def main():
    """Main function for simple evaluation."""
    await run_simple_evaluation()

if __name__ == "__main__":
    asyncio.run(main())