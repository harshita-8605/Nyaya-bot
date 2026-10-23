from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import dotenv_values
import streamlit as st
from agent import get_latency_metrics, stream_agent
from tools.pdf_query_tools import warmup_retrieval

# ============================================================================
# Load environment variables
# ============================================================================
def _merged_env():
    # Read .env first
    local_env = {}
    try:
        local_env = dotenv_values(".env") or {}
    except Exception:
        local_env = {}
    # Overlay Streamlit secrets for any missing or empty values
    secrets_env = {}
    try:
        secrets_env = dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:
        secrets_env = {}

    def pick(key: str, default: str = ""):
        v = local_env.get(key)
        if v is None or str(v).strip() == "":
            v = secrets_env.get(key, default)
        return v

    return {
        "GOOGLE_API_KEY": pick("GOOGLE_API_KEY"),
        "HUGGINGFACE_API_KEY": pick("HUGGINGFACE_API_KEY"),
    }

ENVs = _merged_env()
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

# ============================================================================
# Session State Initialization
# ============================================================================
if "store" not in st.session_state:
    st.session_state.store = []

# ============================================================================
# Authentication Gate - Streamlit Native OAuth
# ============================================================================
try:
    auth_configured = bool(st.secrets.get("auth"))
except Exception:
    auth_configured = False

if not auth_configured:
    logged_in = True
else:
    try:
        logged_in = st.user.is_logged_in
    except Exception as error:
        st.error(f"AUTH ERROR: {type(error).__name__}: {error}")
        st.write("Streamlit version:", st.__version__)
        st.stop()

if auth_configured and not logged_in:
    st.title("🔐 Nyaya-BOT Authentication")
    st.markdown("Sign in with Google to access your legal assistant")

    if st.button("🔑 Sign in with Google", use_container_width=True, type="primary"):
        st.login("google")

    st.markdown("---")
    st.caption("By signing in, you agree to our Terms of Service and Privacy Policy")
    st.stop()

# Preload retrieval resources in this Streamlit process so the first query
# does not pay the embedding-model startup cost.
try:
    warmup_retrieval()
except Exception as error:
    print(f"Retrieval warmup warning: {type(error).__name__}: {error}")

# ============================================================================
# Main Chatbot UI (Only shown after successful login)
# ============================================================================

st.title("Nyaya-BOT⚖️")

# Sidebar for settings
with st.sidebar:
    st.header("Configuration⚙️")
    
    # User info
    st.subheader("👤 User Session")
    user_email = getattr(st.user, "email", "Local development session")
    st.write(f"**Logged in as:** {user_email}")
    st.write("🟢 **Status:** Authenticated")
    
    if auth_configured and st.button("🚪 Logout", use_container_width=True):
        st.logout()
    
    st.markdown("---")
    
    # Display current configuration
    st.subheader("Current Config:")
    st.write(f"**Provider:** Google Gemini (Cloud)")
    st.write(f"**Model:** gemini-3.6-flash")

# Main content
initial_msg = """
#### Welcome!!! I am your legal assistant chatbot👩‍⚖️
#### You can ask me any queries about the laws or constitution of India

"""
st.markdown(initial_msg)

# Use store from session state (already initialized above)
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
    
    # Stream the answer into the assistant message as Gemini produces it.
    thinking_placeholder = st.chat_message("assistant", avatar="⚖️")
    
    with thinking_placeholder:
        store.append(HumanMessage(content=prompt))
        if not GOOGLE_API_KEY:
            response_content = "Sorry, no API key found for Google Gemini. Please set GOOGLE_API_KEY in your .env file."
            st.markdown(response_content)
        else:
            response_content = st.write_stream(stream_agent(prompt))

    store.append(AIMessage(content=response_content))
    print(f"latency_metrics={get_latency_metrics()}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        💡 Powered by Google Gemini for fast and reliable legal assistance
    </div>
    """,
    unsafe_allow_html=True,
)