from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query
import warnings
import time

# Cache LLM instance to avoid recreation
_cached_llm = None
_cached_agent_executor = None

def _get_agent_executor():
    """Get or create cached agent executor."""
    global _cached_llm, _cached_agent_executor
    
    if _cached_agent_executor is None:
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Use Google Gemini for cloud LLM
        _cached_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            timeout=30
        )
        
        tools = [indian_constitution_pdf_query, indian_laws_pdf_query]
        prompt_template = get_prompt_template()
        
        agent = create_react_agent(
            _cached_llm,
            tools,
            prompt_template
        )
        
        _cached_agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,  # Reduced from 10 to 3 for speed
            max_execution_time=45  # 45 second timeout
        )
    
    return _cached_agent_executor


def agent(query: str):
    """
    Create and run an agent with Google Gemini LLM
    
    Args:
        query (str): The user's query
    """
    start_time = time.time()
    agent_executor = _get_agent_executor()

    try:
        result = agent_executor.invoke({"input": query})
        elapsed = time.time() - start_time
        print(f"Query completed in {elapsed:.2f} seconds")
        return result["output"]
    except TimeoutError:
        return "Sorry, the query took too long to process. Please try a simpler question or rephrase it."
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error after {elapsed:.2f}s: {str(e)}"
        print(error_msg)
        
        # Provide helpful fallback
        if "timeout" in str(e).lower():
            return "The query timed out. Please try asking a more specific question."
        elif "rate" in str(e).lower() or "quota" in str(e).lower():
            return "API rate limit reached. Please wait a moment and try again."
        else:
            return f"Sorry, I encountered an error while processing your query. Please try rephrasing it."