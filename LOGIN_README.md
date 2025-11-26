# Nyaya-BOT Login Flow

A minimal JWT-authenticated login flow for Streamlit that integrates with a Flask backend.

## 📁 Files

- `login_app.py` - Streamlit frontend with JWT authentication

## 🔧 Configuration

### API URL Setup

The `API_URL` variable at the top of `login_app.py` must be configured based on your environment:

```python
# For local development (Flask running on your machine):
API_URL = "http://localhost:5000"

# For Docker (Flask service named "backend" in docker-compose):
API_URL = "http://backend:5000"
```

## 🚀 Running the Application

### Option 1: Local Development

1. **Start the Flask backend** (in a separate terminal):
   ```bash
   python flask_app.py
   # Or however your Flask backend starts
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run login_app.py
   ```

3. **Access the app** at `http://localhost:8501`

### Option 2: Docker

1. **Update API_URL** in `login_app.py`:
   ```python
   API_URL = "http://backend:5000"  # Use service name from docker-compose
   ```

2. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the app** at `http://localhost:8501` (or whatever port you mapped)

## 🔐 How It Works

### Authentication Flow

1. **Login Page** (unauthenticated state):
   - User enters username and password
   - App sends `POST /login` to Flask backend
   - On success, JWT token is stored in `st.session_state["auth_token"]`
   - On failure, error message is displayed

2. **Welcome Page** (authenticated state):
   - Displays welcome message
   - Shows logout button in sidebar
   - Provides button to test protected endpoint
   - When clicked, calls `GET /protected` with JWT token in header

3. **Logout**:
   - Clears `st.session_state["auth_token"]`
   - Refreshes UI to show login page

### Backend API Requirements

Your Flask backend must implement:

#### 1. Login Endpoint
```
POST http://localhost:5000/login
Content-Type: application/json

Request Body:
{
  "username": "your_username",
  "password": "your_password"
}

Success Response (200):
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Error Response (401):
{
  "message": "Invalid credentials"
}
```

#### 2. Protected Endpoint
```
GET http://localhost:5000/protected
Authorization: Bearer <your_jwt_token>

Success Response (200):
{
  "message": "Access granted",
  "user": "username"
}

Error Response (401):
{
  "message": "Token expired" or "Invalid token"
}
```

## 🧪 Testing

### Manual Testing

1. **Test login with valid credentials**:
   - Enter valid username/password
   - Should redirect to welcome page
   - Token should be stored in session

2. **Test login with invalid credentials**:
   - Enter wrong username/password
   - Should show error message
   - Should stay on login page

3. **Test protected endpoint**:
   - After login, click "Call Protected Route"
   - Should display server response
   - Should show success message

4. **Test logout**:
   - Click logout button
   - Should clear session and return to login page

### Using curl for Backend Testing

Test your Flask backend independently:

```bash
# Test login
curl -X POST http://localhost:5000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'

# Test protected route (replace TOKEN with actual token)
curl -X GET http://localhost:5000/protected \
  -H "Authorization: Bearer TOKEN"
```

## 🔒 Security Notes

1. **Token Storage**: JWT tokens are stored in `st.session_state` which is:
   - Session-specific (cleared when browser tab is closed)
   - Not persistent across page reloads (by default)
   - Memory-only (not written to disk)

2. **Production Recommendations**:
   - Use HTTPS in production
   - Implement token expiration on backend
   - Add refresh token mechanism for long sessions
   - Use secure password hashing (bcrypt, argon2)
   - Add rate limiting on login endpoint

3. **Environment Variables**:
   - Store sensitive configs in `.env` file
   - Never commit `.env` to version control
   - Use different tokens for dev/staging/prod

## 🐛 Troubleshooting

### "Cannot connect to backend"
- Check if Flask server is running
- Verify `API_URL` is correct for your environment
- For Docker, ensure services are in the same network

### "Invalid response from server (no token received)"
- Backend response must be JSON with `"token"` key
- Check Flask backend logs for errors

### "Access denied" on protected endpoint
- Token may be expired
- Token format must be: `Authorization: Bearer <token>`
- Check backend JWT secret key is configured

### Token displayed as "undefined"
- Check backend returns token in correct format
- Verify response parsing in `login()` function

## 📦 Dependencies

Required Python packages (already in `requirements.txt`):
- `streamlit` - Frontend framework
- `requests` - HTTP client for API calls

## 🔄 Integration with Existing App

To integrate this login flow with your existing `app.py`:

1. **Add authentication check** at the top of `app.py`:
   ```python
   if "auth_token" not in st.session_state or st.session_state.auth_token is None:
       st.warning("Please login first")
       st.stop()
   ```

2. **Add logout button** to sidebar in `app.py`:
   ```python
   with st.sidebar:
       if st.button("Logout"):
           st.session_state.auth_token = None
           st.rerun()
   ```

3. **Use token for API calls** (if needed):
   ```python
   headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
   response = requests.get(f"{API_URL}/your-endpoint", headers=headers)
   ```

## 📝 Notes

- This is a minimal implementation focused on core JWT authentication
- No styling libraries are used (follows requirement for simplicity)
- All code is in a single file (`login_app.py`)
- Session state is used for token storage (follows Streamlit best practices)
- Error handling is comprehensive with user-friendly messages
