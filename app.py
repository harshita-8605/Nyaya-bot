from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import dotenv_values
import streamlit as st
from agent import agent

# Load environment variables
try:
    ENVs = dotenv_values(".env")  # for dev env
    GOOGLE_API_KEY = ENVs.get("GOOGLE_API_KEY", "")
except:
    ENVs = st.secrets  # for streamlit deployment
    GOOGLE_API_KEY = ENVs.get("GOOGLE_API_KEY", "")

# Set environment variables
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if ENVs.get("HUGGINGFACE_API_KEY"):
    os.environ["HUGGINGFACE_API_KEY"] = ENVs["HUGGINGFACE_API_KEY"]

# Configure Streamlit
st.set_page_config(
    page_title="Nyaya-BOT👩‍⚖️",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Nyaya-BOT⚖️")

# Sidebar for settings
with st.sidebar:
    st.header("Configuration⚙️")
    
    # Display current configuration
    st.subheader("Current Config:")
    st.write(f"**Provider:** Google Gemini (Cloud)")
    st.write(f"**Model:** gemini-2.5-flash")

# Main content
initial_msg = """
#### Welcome!!! I am your legal assistant chatbot👩‍⚖️
#### You can ask me any queries about the laws or constitution of India

"""
st.markdown(initial_msg)

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = []

store = st.session_state.store

# Display chat history
for message in store:
    if message.type == "ai":
        avatar = "👩‍⚖️"
    else:
        avatar = "🗨️"
    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("What is your query?"):
    # Display user message
    st.chat_message("user", avatar="🗨️").markdown(prompt)
    
    # Show thinking message
    thinking_placeholder = st.chat_message("assistant", avatar="⚖️")
    thinking_placeholder.markdown("Thinking...")
    
    # Add user message to store
    store.append(HumanMessage(content=prompt))
    
    try:
        # Check if Google API key is available
        if not GOOGLE_API_KEY:
            response_content = "Sorry, no API key found for Google Gemini. Please set GOOGLE_API_KEY in your .env file."
        else:
            response_content = agent(prompt)
        
        response = AIMessage(content=response_content)
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        if "API" in str(e).upper():
            error_msg += "\n\nThis might be due to API limits or network issues."
        response = AIMessage(content=error_msg)
    
    # Add response to store
    store.append(response)
    
    # Update the thinking message with actual response
    thinking_placeholder.markdown(response.content)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        💡 Powered by Google Gemini for fast and reliable legal assistance
    </div>
    """, 
    unsafe_allow_html=True
)