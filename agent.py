from langchain.agents import create_react_agent, AgentExecutor
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query
import warnings
import time
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Cache LLM instance to avoid recreation
_cached_llm = None
_cached_agent_executor = None
_latency_lock = threading.Lock()
_latency_samples = {
    "retrieval_ms": [],
    "time_to_first_token_ms": [],
    "model_ms": [],
    "total_ms": [],
}


def _record_latency(metric: str, value_ms: float) -> None:
    with _latency_lock:
        samples = _latency_samples[metric]
        samples.append(round(value_ms, 2))
        del samples[:-1000]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def get_latency_metrics() -> dict[str, dict[str, float | int | None]]:
    """Return current-process request counts and p50/p95 latency."""
    with _latency_lock:
        return {
            metric: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
            }
            for metric, values in _latency_samples.items()
        }


def _get_llm():
    """Create the configured cloud or local chat model."""
    global _cached_llm
    if _cached_llm is None:
        warnings.filterwarnings("ignore", category=FutureWarning)
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

        if provider == "groq":
            from langchain_groq import ChatGroq

            _cached_llm = ChatGroq(
                model=model or "llama-3.1-8b-instant",
                temperature=temperature,
                timeout=timeout,
            )
        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            _cached_llm = ChatOllama(
                model=model or "llama3.1:8b",
                base_url=os.getenv(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                ),
                temperature=temperature,
                timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            _cached_llm = ChatGoogleGenerativeAI(
                model=model or "gemini-3.6-flash",
                temperature=temperature,
                timeout=timeout,
            )
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. "
                "Use gemini, groq, or ollama."
            )
    return _cached_llm

def _get_agent_executor():
    """Get or create cached agent executor."""
    global _cached_llm, _cached_agent_executor
    
    if _cached_agent_executor is None:
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Use Google Gemini for cloud LLM
        _cached_llm = _get_llm()
        
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
            max_iterations=10,  # increased for deeper reasoning
            max_execution_time=70  # allow proportionally more time
        )
    
    return _cached_agent_executor


def _retrieve_context(query: str) -> str:
    """Retrieve from both local indexes concurrently for a single synthesis call."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        constitution_future = executor.submit(
            indian_constitution_pdf_query.invoke, query
        )
        laws_future = executor.submit(indian_laws_pdf_query.invoke, query)
        constitution = constitution_future.result()
        laws = laws_future.result()

    return (
        f"Constitution references:\n{constitution}\n\n"
        f"BNS law references:\n{laws}"
    )[:7000]


def stream_agent(query: str):
    """Yield the answer incrementally after one parallel retrieval pass."""
    request_start = time.perf_counter()
    try:
        retrieval_start = time.perf_counter()
        context = _retrieve_context(query)
        _record_latency(
            "retrieval_ms", (time.perf_counter() - retrieval_start) * 1000
        )
        prompt = (
            "You are a careful legal assistant for Indian law. Answer the user's "
            "question using only the supplied excerpts. Be concise but helpful. "
            "If the excerpts do not contain enough information, say so clearly. "
            "Do not invent sections, citations, or legal advice.\n\n"
            f"Question: {query}\n\nExcerpts:\n{context}\n\nAnswer:"
        )
        model_start = time.perf_counter()
        first_token_recorded = False
        for chunk in _get_llm().stream(prompt):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                if not first_token_recorded:
                    _record_latency(
                        "time_to_first_token_ms",
                        (time.perf_counter() - request_start) * 1000,
                    )
                    first_token_recorded = True
                yield content
        _record_latency("model_ms", (time.perf_counter() - model_start) * 1000)
        _record_latency("total_ms", (time.perf_counter() - request_start) * 1000)
    except Exception as error:
        error_text = str(error).lower()
        _record_latency("total_ms", (time.perf_counter() - request_start) * 1000)
        print(f"Streaming query failed: {type(error).__name__}: {error}")
        if any(term in error_text for term in ("quota", "resource_exhausted", "rate limit")):
            yield (
                "The configured language model is temporarily unavailable because "
                "its API quota or rate limit was exhausted. Please try again later."
            )
            return
        yield "Sorry, I encountered an error while processing your query. Please try rephrasing it."


def agent(query: str):
    """
    Create and run an agent with Google Gemini LLM
    
    Args:
        query (str): The user's query
    """
    start_time = time.time()
    agent_executor = _get_agent_executor()

    def _fallback_synthesis(q: str):
        try:
            from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query
            const_passages = indian_constitution_pdf_query(q)
            law_passages = indian_laws_pdf_query(q)
            context = f"Constitution References:\n{const_passages}\n\nLaw References:\n{law_passages}"[:6000]
            synthesis_llm = _cached_llm or ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
            prompt = (
                "You are a legal assistant for Indian law. Using ONLY the provided excerpts, "
                "answer the user's question clearly. If information is insufficient, say so.\n\n" \
                f"Question: {q}\n\nExcerpts:\n{context}\n\nAnswer:" )
            answer = synthesis_llm.invoke(prompt)
            return getattr(answer, "content", str(answer))
        except Exception:
            return (
                "The agent stopped early and the fallback also failed. "
                "Please try rephrasing your question or narrow its scope."
            )

    try:
        result = agent_executor.invoke({"input": query})
        elapsed = time.time() - start_time
        print(f"Query completed in {elapsed:.2f} seconds")
        output = result.get("output", "")
        lowered = output.lower()
        if any(phrase in lowered for phrase in ["iteration limit", "time limit"]):
            print("Early stop detected in output; triggering fallback synthesis.")
            return _fallback_synthesis(query)
        if not output:
            return _fallback_synthesis(query)
        return output
    except TimeoutError:
        return "Sorry, the query took too long to process. Please try a simpler question or rephrase it."
    except Exception as e:
        elapsed = time.time() - start_time
        msg = str(e)
        print(f"Agent primary failure after {elapsed:.2f}s: {msg}")

        # Fallback: direct retrieval + synthesis if iteration/time limit or parsing issues
        trigger_phrases = ["iteration limit", "time limit", "parsing", "Stopped"]
        if any(tp.lower() in msg.lower() for tp in trigger_phrases):
            return _fallback_synthesis(query)

        if "timeout" in msg.lower():
            return "The query timed out. Please try asking a more specific question."
        if any(w in msg.lower() for w in ["rate", "quota"]):
            return "API rate limit reached. Please wait a moment and try again."
        return "Sorry, I encountered an error while processing your query. Please try rephrasing it."