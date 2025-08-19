import streamlit as st
from passlib.hash import pbkdf2_sha256
from jose import jwt
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Secret key for JWT - in production, use a secure secret key
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# In-memory user storage (replace with a database in production)
users_db = {}

class User:
    def __init__(self, username: str, email: str, hashed_password: str):
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.saved_searches = []
        self.annotations = {}
        self.preferences = {
            "theme": "light",
            "notifications": True,
            "default_category": "Computer Science"
        }

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pbkdf2_sha256.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pbkdf2_sha256.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user."""
    if username not in users_db:
        return None
    user = users_db[username]
    if not verify_password(password, user.hashed_password):
        return None
    return user

def register_user(username: str, email: str, password: str) -> Optional[User]:
    """Register a new user."""
    if username in users_db:
        return None
    hashed_password = get_password_hash(password)
    user = User(username=username, email=email, hashed_password=hashed_password)
    users_db[username] = user
    return user

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get the current user from the session."""
    if 'user' not in st.session_state:
        return None
    try:
        token = st.session_state['user']
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            return None
        user = users_db[username]
        return {
            "username": user.username,
            "email": user.email,
            "preferences": user.preferences,
            "saved_searches": user.saved_searches,
            "token": token
        }
    except jwt.JWTError:
        return None

def login_required(func):
    """Decorator to protect routes that require authentication."""
    def wrapper(*args, **kwargs):
        if 'user' not in st.session_state or not get_current_user():
            st.warning("Please log in to access this page.")
            if st.button("Go to Login"):
                st.session_state['page'] = 'login'
                st.rerun()
            return None
        return func(*args, **kwargs)
    return wrapper

def get_user_preferences(username: str) -> dict:
    """Get user preferences."""
    if username in users_db:
        return users_db[username].preferences
    return {}

def save_user_preferences(username: str, preferences: dict) -> bool:
    """Save user preferences."""
    if username in users_db:
        users_db[username].preferences.update(preferences)
        return True
    return False

def save_search(username: str, search_query: str, filters: dict) -> bool:
    """Save a user's search."""
    if username not in users_db:
        return False
    users_db[username].saved_searches.append({
        "query": search_query,
        "filters": filters,
        "timestamp": datetime.utcnow().isoformat()
    })
    return True

def get_saved_searches(username: str) -> list:
    """Get a user's saved searches."""
    if username not in users_db:
        return []
    return users_db[username].saved_searches

def save_annotation(username: str, paper_id: str, annotation: dict) -> bool:
    """Save a user's annotation on a paper."""
    if username not in users_db:
        return False
    if paper_id not in users_db[username].annotations:
        users_db[username].annotations[paper_id] = []
    users_db[username].annotations[paper_id].append({
        **annotation,
        "timestamp": datetime.utcnow().isoformat()
    })
    return True

def get_annotations(username: str, paper_id: str = None) -> dict:
    """Get a user's annotations."""
    if username not in users_db:
        return {}
    if paper_id:
        return users_db[username].annotations.get(paper_id, [])
    return users_db[username].annotations
