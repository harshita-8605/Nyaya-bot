import streamlit as st
import requests
import json

# ============================================================================
# API Configuration
# ============================================================================
# IMPORTANT: Change this based on your environment:
# - For Docker: use "http://backend:5000" (where "backend" is the service name)
# - For local testing: use "http://localhost:5000"
API_URL = "http://localhost:5000"  # Change to "http://backend:5000" for Docker
# ============================================================================

# Configure Streamlit page
st.set_page_config(
    page_title="Nyaya-BOT Login",
    page_icon="🔐",
    layout="centered",
)

# Initialize session state for auth token
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Authenticate user with Flask backend.
    
    Args:
        username: User's username
        password: User's password
    
    Returns:
        (success: bool, message: str) - True if login successful, error message otherwise
    """
    try:
        # Send POST request to login endpoint
        response = requests.post(
            f"{API_URL}/login",
            json={"username": username, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        # Check if login was successful
        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            
            if token:
                # Store token in session state
                st.session_state.auth_token = token
                return True, "Login successful!"
            else:
                return False, "Invalid response from server (no token received)"
        else:
            # Login failed - extract error message
            try:
                error_msg = response.json().get("message", "Login failed")
            except:
                error_msg = f"Login failed with status code {response.status_code}"
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to backend at {API_URL}. Is the Flask server running?"
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def call_protected_endpoint() -> tuple[bool, str]:
    """
    Call the protected Flask endpoint using the stored JWT token.
    
    Returns:
        (success: bool, response_text: str) - Server response or error message
    """
    try:
        # Get token from session state
        token = st.session_state.auth_token
        
        if not token:
            return False, "No authentication token found. Please login again."
        
        # Send GET request with Authorization header
        response = requests.get(
            f"{API_URL}/protected",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        
        if response.status_code == 200:
            # Successfully accessed protected route
            try:
                data = response.json()
                return True, json.dumps(data, indent=2)
            except:
                return True, response.text
        else:
            # Access denied or token expired
            try:
                error_msg = response.json().get("message", "Access denied")
            except:
                error_msg = f"Access denied (status code {response.status_code})"
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to backend at {API_URL}"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"


def logout():
    """Clear authentication token and reset session."""
    st.session_state.auth_token = None
    st.rerun()


# ============================================================================
# Main UI Logic
# ============================================================================

# Check if user is authenticated
if st.session_state.auth_token is None:
    # ========================================
    # LOGIN PAGE (Not authenticated)
    # ========================================
    
    st.title("🔐 Nyaya-BOT Login")
    st.markdown("Please login to access the application")
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit_button = st.form_submit_button("Login", use_container_width=True)
        
        if submit_button:
            # Validate inputs
            if not username or not password:
                st.error("⚠️ Please enter both username and password")
            else:
                # Show loading spinner while authenticating
                with st.spinner("Authenticating..."):
                    success, message = login(username, password)
                
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()  # Refresh to show authenticated UI
                else:
                    st.error(f"❌ {message}")
    
    # Help section
    st.markdown("---")
    st.info(f"**Backend URL:** `{API_URL}`\n\nMake sure your Flask backend is running!")

else:
    # ========================================
    # WELCOME PAGE (Authenticated)
    # ========================================
    
    st.title("✅ Welcome to Nyaya-BOT!")
    st.success("You are successfully logged in")
    
    # Logout button in sidebar
    with st.sidebar:
        st.header("User Session")
        st.write("🟢 **Status:** Authenticated")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    st.markdown("---")
    
    # Section to test protected endpoint
    st.subheader("🔒 Test Protected Endpoint")
    st.markdown("Click the button below to call the protected Flask endpoint using your JWT token.")
    
    if st.button("Call Protected Route", use_container_width=True):
        with st.spinner("Calling protected endpoint..."):
            success, response = call_protected_endpoint()
        
        if success:
            st.success("✅ Protected endpoint accessed successfully!")
            st.code(response, language="json")
        else:
            st.error(f"❌ {response}")
            
            # If token is invalid/expired, suggest logout
            if "denied" in response.lower() or "expired" in response.lower():
                st.warning("Your session may have expired. Please logout and login again.")
    
    st.markdown("---")
    
    # Display token info (first and last 10 chars only for security)
    with st.expander("🔑 View Token Info"):
        token = st.session_state.auth_token
        if len(token) > 20:
            masked_token = f"{token[:10]}...{token[-10:]}"
        else:
            masked_token = token[:5] + "..." + token[-5:]
        
        st.code(f"Token: {masked_token}", language="text")
        st.caption("Full token is stored securely in session state")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 12px;'>
            Nyaya-BOT with JWT Authentication
        </div>
        """,
        unsafe_allow_html=True
    )
