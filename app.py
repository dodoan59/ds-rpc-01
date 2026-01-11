import streamlit as st
import requests
from typing import Optional, Dict
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIG API ---
API_BASE_URL = "http://localhost:8000"

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="RAG RBAC System",
    page_icon="❇️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INIT SESSION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.description = ""
    st.session_state.messages = []
    st.session_state.auth_header = ""

# --- HELPER FUNCTIONS ---
def get_basic_auth_header(username: str, password: str) -> str:
    """Create Basic Auth header"""
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded_credentials}"

def make_authenticated_request(endpoint: str, method: str = "get", data: Optional[Dict] = None):
    """Send authenticated request with Auth Header"""
    headers = {
        "Authorization": st.session_state.auth_header,
        "Content-Type": "application/json"
    }
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method.lower() == "post":
            response = requests.post(url, headers=headers, json=data)
        else:
            response = requests.get(url, headers=headers)
            
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            st.session_state.authenticated = False # Auto logout if session expired
            return None, "Session expired or invalid credentials."
        else:
            return None, f"Server error ({response.status_code}): {response.text}"
            
    except Exception as e:
        return None, f"Failed to connect to server: {str(e)}"

# --- LOGIN PAGE ---
def login_page():
    st.title("🔐 Login RAG RBAC System")
    st.markdown("System to search employee information & internal regulations (Role-based access control).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Login Information")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", type="primary"):
            if username and password:
                # Create auth header to check login
                auth_header = get_basic_auth_header(username, password)
                
                # Call API check login (or endpoint /login)
                headers = {"Authorization": auth_header}
                try:
                    res = requests.get(f"{API_BASE_URL}/login", headers=headers)
                    if res.status_code == 200:
                        data = res.json()
                        # Save info to session
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.auth_header = auth_header
                        st.session_state.role = data.get("role", "Unknown")
                        st.session_state.description = data.get("description", "")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter full information.")

    # Show User Demo table for testing
    with col2:
        st.info("💡 **User Demo (RBAC Demo)**")
        st.markdown("""
        | Role | Username | Password |
        |---|---|---|
        | **HR** | `Natasha` | `hrpass123` |
        | **Finance** | `Sam` | `financepass` |
        | **Engineer** | `Tony` | `password123` |
        | **Marketing** | `Bruce` | `password123` |
        | **Employee** | `John` | `johnpass123` |
        | **C-level** | `Alan` | `ceo123` |
        """)

# --- CHAT INTERFACE ---
def chat_interface():
    # Sidebar info
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.success(f"Role: **{st.session_state.role.upper()}**")
        if st.session_state.description:
            st.caption(st.session_state.description)
        
        st.markdown("---")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("### 🛠️ Debug Info")
        st.markdown(f"- Auth Status: {'✅' if st.session_state.authenticated else '❌'}")
        
    st.header("💬 AI Assistant (RBAC Enabled)")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Enter your question... (Example: What is my salary this month?)"):
        # Show user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process AI response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response_data, error = make_authenticated_request(
                    "/query", 
                    method="post", 
                    data={"query": prompt}
                )
                
                if error:
                    st.error(error)
                    final_response = f"⚠️ {error}"
                else:
                    # Get response (Support both 'answer' and 'response' key for backward compatibility)
                    final_response = response_data.get("answer") or response_data.get("response") or "No response received."
                    
                    st.markdown(final_response)
                    
                    # If debug info from server, show it below
                    if "user_role" in response_data:
                        st.caption(f"🔒 Answered with role: `{response_data['user_role']}`")

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

# --- MAIN FUNCTION ---
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_interface()

if __name__ == "__main__":
    main()