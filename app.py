import streamlit as st
import requests
from typing import Optional, Dict
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update this if your FastAPI server is running elsewhere

# Page configuration
st.set_page_config(
    page_title="RAG Chat Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.messages = []
    st.session_state.auth_token = ""

# Utility functions
def get_basic_auth_header(username: str, password: str) -> str:
    """Create Basic Auth header from username and password"""
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded_credentials}"

def make_authenticated_request(endpoint: str, method: str = "get", data: Optional[Dict] = None):
    """Make an authenticated request to the API"""
    headers = {
        "Authorization": st.session_state.auth_token,
        "Content-Type": "application/json"
    }
    
    try:
        if method.lower() == "get":
            response = requests.get(f"{API_BASE_URL}{endpoint}", headers=headers)
        elif method.lower() == "post":
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                headers=headers,
                json=data or {}
            )
        else:
            return None, "Unsupported HTTP method"
            
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

# Authentication
def login():
    """Login form"""
    st.title("🔑 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return
                
            # Create auth token
            auth_token = get_basic_auth_header(username, password)
            
            # Test login
            try:
                response = requests.get(
                    f"{API_BASE_URL}/login",
                    headers={"Authorization": auth_token}
                )
                response.raise_for_status()
                user_data = response.json()
                
                # Update session state
                st.session_state.authenticated = True
                # Use username from response if available, otherwise fallback to the entered one
                st.session_state.username = user_data.get("username", username)
                st.session_state.role = user_data.get("role")
                st.session_state.auth_token = auth_token
                st.session_state.messages = []
                st.rerun()
                
            except requests.exceptions.RequestException as e:
                st.error(f"Login failed: {str(e)}")

# Main chat interface
def chat_interface():
    """Main chat interface"""
    st.title("🤖 RAG Chat Assistant")
    
    # Sidebar with user info
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.write(f"**Role:** {st.session_state.role}")
        
        if st.button("🔄 Refresh"):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.messages = []
            st.session_state.auth_token = ""
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call the API
                    response, error = make_authenticated_request(
                        "/query",
                        method="post",
                        data={"query": prompt}
                    )
                    
                    if error:
                        raise Exception(error)
                    
                    # Display AI response
                    st.markdown(response.get("response", "No response received"))
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("response", "No response received")
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Main app
def main():
    """Main app function"""
    if not st.session_state.authenticated:
        login()
    else:
        chat_interface()

if __name__ == "__main__":
    main()