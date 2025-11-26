from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import dotenv_values
import streamlit as st
from agent import agent

# ============================================================================
# Simple Authentication Configuration
# ============================================================================
# You can add more users here or connect to a database later
VALID_USERS = {
    "admin": "admin123",
    "user": "password123",
    "test": "test123"
}
# ============================================================================

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

# ============================================================================
# Authentication Functions
# ============================================================================

def simple_login(username: str, password: str) -> tuple[bool, str]:
    """Simple authentication check."""
    if username in VALID_USERS and VALID_USERS[username] == password:
        st.session_state.auth_token = "authenticated"
        st.session_state.username = username
        return True, "Login successful!"
    else:
        return False, "Invalid username or password"


def logout():
    """Clear authentication and reset session."""
    st.session_state.auth_token = None
    st.session_state.username = None
    st.session_state.store = []
    st.rerun()


# Initialize auth session state
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "username" not in st.session_state:
    st.session_state.username = None

# ============================================================================
# Login Gate: Show login page if not authenticated
# ============================================================================

if st.session_state.auth_token is None:
    st.title("🔐 Nyaya-BOT Login")
    st.markdown("Please login to access the legal assistant chatbot")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit_button = st.form_submit_button("Login", use_container_width=True)
        
        if submit_button:
            if not username or not password:
                st.error("⚠️ Please enter both username and password")
            else:
                success, message = simple_login(username, password)
                
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    st.markdown("---")
    st.info("**Default credentials:**\n- Username: `admin` / Password: `admin123`\n- Username: `user` / Password: `password123`")
    st.stop()  # Stop execution here if not authenticated

# ============================================================================
# Main Chatbot UI (Only shown after successful login)
# ============================================================================

st.title("Nyaya-BOT⚖️")

# Sidebar for settings
with st.sidebar:
    st.header("Configuration⚙️")
    
    # User info
    st.subheader("👤 User Session")
    st.write(f"**Logged in as:** {st.session_state.username}")
    st.write("🟢 **Status:** Authenticated")
    
    if st.button("🚪 Logout", use_container_width=True):
        logout()
    
    st.markdown("---")
    
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