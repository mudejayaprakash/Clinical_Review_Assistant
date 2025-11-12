"""
User authentication system for Clinical Review Assistant
"""
import streamlit as st
import hashlib
import json
import os
from datetime import datetime


class UserAuth:
    """Manages user authentication with registration and login"""
    
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self._ensure_users_file()
    
    def _ensure_users_file(self):
        """Create users file if it doesn't exist"""
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, user_id: str, password: str) -> tuple[bool, str]:
        """
        Register new user
        
        Args:
            user_id: User ID
            password: Password
            
        Returns:
            Tuple of (success, message)
        """
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if user_id in users:
                return False, "User ID already exists"
            
            users[user_id] = {
                "password": self.hash_password(password),
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True, "Registration successful"
            
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def authenticate(self, user_id: str, password: str) -> bool:
        """
        Authenticate user
        
        Args:
            user_id: User ID
            password: Password
            
        Returns:
            True if authenticated, False otherwise
        """
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if user_id not in users:
                return False
            
            return users[user_id]["password"] == self.hash_password(password)
            
        except Exception:
            return False
    
    def login_page(self):
        """Render login/registration page"""
        st.title("🏥 Clinical Review Assistant")
        st.markdown("### AI-Powered Medical Record Analysis")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            user_id = st.text_input("User ID", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Login", type="primary", use_container_width=True):
                    if not user_id or not password:
                        st.error("Please enter both User ID and Password")
                    elif self.authenticate(user_id, password):
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_id
                        # Save session to file for persistence
                        from app import save_session
                        save_session(user_id)
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
        
        with tab2:
            st.subheader("Register New Account")
            new_user_id = st.text_input("User ID", key="reg_user", help="Choose a unique user ID")
            new_password = st.text_input("Password", type="password", key="reg_pass", help="Minimum 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Register", use_container_width=True):
                    if not new_user_id or not new_password:
                        st.error("Please fill all fields")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = self.register_user(new_user_id, new_password)
                        if success:
                            st.success(f"✅ {message}! Please login.")
                        else:
                            st.error(f"❌ {message}")
        
        # Footer
        st.markdown("---")
        st.caption("⚕️ Clinical Review Assistant - Secure, HIPAA-Compliant AI Agent")


# Global authentication instance
auth = UserAuth()
