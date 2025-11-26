from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import indian_constitution_pdf_query, indian_laws_pdf_query
import warnings


def agent(query: str):
    """
    Create and run an agent with Google Gemini LLM
    
    Args:
        query (str): The user's query
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Use Google Gemini for cloud LLM
    LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("Using Google Gemini model: gemini-2.5-flash")

    tools = [indian_constitution_pdf_query, indian_laws_pdf_query]
    prompt_template = get_prompt_template()

    agent = create_react_agent(
        LLM,
        tools,
        prompt_template
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        handle_parsing_errors=True,
        max_iterations=10
    )

    try:
        result = agent_executor.invoke({"input": query})
        return result["output"]
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return f"Sorry, I encountered an error while processing your query: {str(e)}"