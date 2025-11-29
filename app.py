from langchain_core.messages import HumanMessage, AIMessage
import os
import json
import base64
import time
import hashlib
import requests
from datetime import datetime, timedelta
from dotenv import dotenv_values
import streamlit as st
import streamlit.components.v1 as components
import jwt
from agent import agent

# ============================================================================
# User Storage Configuration
# ============================================================================
USERS_FILE = "users.json"

# ==========================================================================
# GitHub-backed persistence (optional via secrets)
# ==========================================================================
GITHUB_TOKEN = None
GITHUB_REPO = None  # format: owner/repo
GITHUB_BRANCH = "main"

# ==========================================================================
# Auth configuration
# ==========================================================================
JWT_SECRET = None
JWT_ALGO = "HS256"
PASSWORD_SALT = ""

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def _gh_api_url(path: str) -> str:
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"


def _github_get_file(path: str) -> tuple[dict, str] | tuple[None, None]:
    try:
        resp = requests.get(_gh_api_url(path), params={"ref": GITHUB_BRANCH}, headers=_gh_headers(), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            content_b64 = data.get("content", "")
            content = base64.b64decode(content_b64).decode("utf-8")
            return json.loads(content) if content else {}, data.get("sha")
        return None, None
    except Exception:
        return None, None


def _github_put_file(path: str, content_obj: dict, message: str) -> bool:
    try:
        # Get current sha if exists
        _, sha = _github_get_file(path)
        content_b64 = base64.b64encode(json.dumps(content_obj, indent=2).encode("utf-8")).decode("utf-8")
        payload = {
            "message": message,
            "content": content_b64,
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha
        resp = requests.put(_gh_api_url(path), headers=_gh_headers(), json=payload, timeout=20)
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def _hash_pw(pw: str) -> str:
    h = hashlib.sha256()
    h.update((PASSWORD_SALT + pw).encode("utf-8"))
    return h.hexdigest()


def _verify_pw(stored_value: str, provided_pw: str) -> bool:
    # Backward-compatible: accept either plaintext or salted hash
    if stored_value == provided_pw:
        return True
    return stored_value == _hash_pw(provided_pw)


def _create_jwt(username: str, exp_minutes: int = 7 * 24 * 60) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=exp_minutes)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def _verify_jwt(token: str) -> str | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload.get("sub")
    except Exception:
        return None


def load_users():
    """Load users from GitHub if configured, else local file."""
    if GITHUB_TOKEN and GITHUB_REPO:
        data, _ = _github_get_file(USERS_FILE)
        if isinstance(data, dict):
            return data
        # Fall through to local if GitHub failed
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        default_users = {
            "admin": _hash_pw("admin123"),
            "user": _hash_pw("password123"),
            "test": _hash_pw("test123"),
        }
        save_users(default_users)
        return default_users
    except json.JSONDecodeError:
        return {}

def save_users(users_dict):
    """Save users to GitHub if configured, else local file."""
    try:
        if GITHUB_TOKEN and GITHUB_REPO:
            ok = _github_put_file(USERS_FILE, users_dict, message="chore(auth): update users.json via app")
            if not ok:
                st.warning("Could not save users to GitHub. Falling back to local file.")
            else:
                return True
        with open(USERS_FILE, 'w') as f:
            json.dump(users_dict, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving users: {str(e)}")
        return False
# ============================================================================

# Load environment variables
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
        "GITHUB_TOKEN": pick("GITHUB_TOKEN"),
        "GITHUB_REPO": pick("GITHUB_REPO", "harshita-8605/Nyaya-bot"),
        "GITHUB_BRANCH": pick("GITHUB_BRANCH", "main"),
        "JWT_SECRET": pick("JWT_SECRET"),
        "PASSWORD_SALT": pick("PASSWORD_SALT", "nyaya-salt"),
    }

ENVs = _merged_env()
GOOGLE_API_KEY = ENVs.get("GOOGLE_API_KEY", "")

# Inject optional secrets
GITHUB_TOKEN = ENVs.get("GITHUB_TOKEN")
GITHUB_REPO = ENVs.get("GITHUB_REPO") or "harshita-8605/Nyaya-bot"
GITHUB_BRANCH = ENVs.get("GITHUB_BRANCH") or "main"
JWT_SECRET = ENVs.get("JWT_SECRET") or ENVs.get("GOOGLE_API_KEY") or "dev-insecure-secret"
PASSWORD_SALT = ENVs.get("PASSWORD_SALT") or "nyaya-salt"

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
# Session State Initialization (persists across page refreshes)
# ============================================================================
# NOTE: Streamlit session_state persists in the same browser tab across refreshes
# Users will stay logged in unless they explicitly logout or close the tab

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "show_register" not in st.session_state:
    st.session_state.show_register = False
if "store" not in st.session_state:
    st.session_state.store = []
# ============================================================================

# Attempt to restore session from localStorage via query param token
components.html(
        """
        <script>
        try {
            const t = localStorage.getItem('nyaya_jwt');
            const url = new URL(window.location.href);
            if (t && !url.searchParams.get('token')) {
                url.searchParams.set('token', t);
                window.location.replace(url.toString());
            }
        } catch (e) {}
        </script>
        """,
        height=0,
)

try:
        qp = st.experimental_get_query_params()
        tkn = (qp.get("token") or [None])[0]
        if tkn and st.session_state.auth_token is None:
                uname = _verify_jwt(tkn)
                if uname:
                        st.session_state.auth_token = tkn
                        st.session_state.username = uname
                        # Clear token from URL
                        st.experimental_set_query_params()
except Exception:
        pass

# ============================================================================
# Authentication Functions
# ============================================================================

def register_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user."""
    # Validate inputs
    if not username or not password:
        return False, "Username and password cannot be empty"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Check for special characters in username
    if not username.isalnum():
        return False, "Username can only contain letters and numbers"
    
    # Load existing users
    users = load_users()
    
    # Check if user already exists
    if username in users:
        return False, "User already registered! Please login instead."
    
    # Add new user
    users[username] = _hash_pw(password)
    
    # Save to file
    if save_users(users):
        return True, "Registration successful! Please login with your credentials."
    else:
        return False, "Error saving user data. Please try again."


def login_user(username: str, password: str) -> tuple[bool, str]:
    """Authenticate existing user."""
    users = load_users()
    
    if username not in users:
        return False, "User not found! Please register first."
    if _verify_pw(users[username], password):
        token = _create_jwt(username)
        st.session_state.auth_token = token
        st.session_state.username = username
        # Store token in browser
        components.html(
            f"<script>localStorage.setItem('nyaya_jwt', '{token}');</script>",
            height=0,
        )
        return True, "Login successful!"
    else:
        return False, "Invalid password!"


def logout():
    """Clear authentication and reset session."""
    st.session_state.auth_token = None
    st.session_state.username = None
    st.session_state.store = []
    # Clear token from browser
    components.html("<script>localStorage.removeItem('nyaya_jwt');</script>", height=0)
    st.rerun()


# ============================================================================
# Login/Registration Gate
# ============================================================================

if st.session_state.auth_token is None:
    st.title("🔐 Nyaya-BOT Authentication")
    
    # Toggle between Login and Register
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Login", use_container_width=True, type="primary" if not st.session_state.show_register else "secondary"):
            st.session_state.show_register = False
            st.rerun()
    with col2:
        if st.button("📝 Register", use_container_width=True, type="primary" if st.session_state.show_register else "secondary"):
            st.session_state.show_register = True
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.show_register:
        # ==================== REGISTRATION FORM ====================
        st.subheader("📝 Create New Account")
        st.markdown("Fill in the details below to register")
        
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", placeholder="Enter a username (letters/numbers only)")
            new_password = st.text_input("Choose Password", type="password", placeholder="Enter a password (min 6 characters)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
            register_button = st.form_submit_button("Register", use_container_width=True)
            
            if register_button:
                if not new_username or not new_password or not confirm_password:
                    st.error("⚠️ Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("❌ Passwords do not match!")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(f"✅ {message}")
                        st.info("👉 Click on 'Login' tab above to sign in")
                    else:
                        st.error(f"❌ {message}")
        
        st.markdown("---")
        st.caption("Already have an account? Click 'Login' above")
    
    else:
        # ==================== LOGIN FORM ====================
        st.subheader("🔑 Login to Your Account")
        st.markdown("Enter your credentials to access the chatbot")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Login", use_container_width=True)
            
            if login_button:
                if not username or not password:
                    st.error("⚠️ Please enter both username and password")
                else:
                    success, message = login_user(username, password)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        
        st.markdown("---")
        st.caption("Don't have an account? Click 'Register' above")
    
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
    
    # Show detailed thinking message with progress
    thinking_placeholder = st.chat_message("assistant", avatar="⚖️")
    
    with thinking_placeholder:
        with st.spinner("🔍 Analyzing your query..."):
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
    
    # Update with final response
    thinking_placeholder.markdown(response.content)

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