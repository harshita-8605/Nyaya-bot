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
            max_iterations=6,  # allow a few more reasoning steps
            max_execution_time=55  # modestly higher timeout
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
        return result.get("output", "I could not produce a final answer.")
    except TimeoutError:
        return "Sorry, the query took too long to process. Please try a simpler question or rephrase it."
    except Exception as e:
        elapsed = time.time() - start_time
        msg = str(e)
        print(f"Agent primary failure after {elapsed:.2f}s: {msg}")

        # Fallback: direct retrieval + synthesis if iteration/time limit or parsing issues
        trigger_phrases = ["iteration limit", "time limit", "parsing", "Stopped"]
        if any(tp.lower() in msg.lower() for tp in trigger_phrases):
            try:
                from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query
                const_passages = indian_constitution_pdf_query(query)
                law_passages = indian_laws_pdf_query(query)
                context = f"Constitution References:\n{const_passages}\n\nLaw References:\n{law_passages}"[:6000]
                synthesis_llm = _cached_llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
                prompt = (
                    "You are a legal assistant for Indian law. Using ONLY the provided excerpts, "
                    "answer the user's question clearly. If information is insufficient, say so.\n\n" \
                    f"Question: {query}\n\nExcerpts:\n{context}\n\nAnswer:" )
                answer = synthesis_llm.invoke(prompt)
                return getattr(answer, "content", str(answer))
            except Exception as inner:
                return (
                    "The agent stopped early and the fallback also failed. "
                    "Please try rephrasing your question or narrow its scope."
                )

        if "timeout" in msg.lower():
            return "The query timed out. Please try asking a more specific question."
        if any(w in msg.lower() for w in ["rate", "quota"]):
            return "API rate limit reached. Please wait a moment and try again."
        return "Sorry, I encountered an error while processing your query. Please try rephrasing it."