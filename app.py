import streamlit as st
import os
import base64
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import fitz  # PyMuPDF
import requests
from io import BytesIO
import functools

# Import utility modules
# Import from utils modules directly to avoid circular imports
from utils.auth import authenticate_user, register_user, get_current_user, login_required, get_user_preferences, save_user_preferences
from utils.data_processing import load_arxiv_data, search_papers, get_paper_embeddings
from utils.nlp_utils import get_paper_summary, get_concept_explanation, visualize_concept_relationships
from utils.pdf_utils import PDFProcessor
from utils.trends import ResearchTrendAnalyzer, plot_citation_network
from utils.cache_utils import memory_cache, disk_cache, clear_all_caches, get_cache_info

# Caching decorator for Streamlit components
@st.cache_resource(ttl=86400)  # Cache for 24 hours
def get_pdf_processor():
    """Get a cached instance of PDFProcessor."""
    return PDFProcessor()

# For backward compatibility
def get_paper_embeddings(*args, **kwargs):
    """Get paper embeddings (stub for backward compatibility)."""
    return []

def generate_abstract(*args, **kwargs):
    """Generate abstract (stub for backward compatibility)."""
    return ""

# Initialize components with caching
pdf_processor = get_pdf_processor()

# Constants
PAGE_CONFIG = {
    "page_title": "arXiv Research Assistant",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    "Search": "/",
    "New Note": "n",
    "Save Paper": "s",
    "Toggle Dark Mode": "d",
    "Next Page": "→",
    "Previous Page": "←"
}

# Reference Manager Integrations
REFERENCE_MANAGERS = {
    "Zotero": {
        "enabled": True,
        "api_key": "",
        "library_id": "",
        "library_type": "user"
    },
    "Mendeley": {
        "enabled": False,
        "client_id": "",
        "client_secret": ""
    },
    "EndNote": {
        "enabled": False,
        "export_format": "BibTeX"
    }
}

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {
            'categories': [],
            'authors': [],
            'date_range': (None, None),
            'sort_by': 'relevance'
        }
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'saved_papers' not in st.session_state:
        st.session_state.saved_papers = {}
    if 'trend_analyzer' not in st.session_state:
        st.session_state.trend_analyzer = None
    if 'reference_manager' not in st.session_state:
        st.session_state.reference_manager = {
            'active': 'Zotero',
            'connected': False,
            'collections': []
        }
    if 'keyboard_shortcuts' not in st.session_state:
        st.session_state.keyboard_shortcuts = KEYBOARD_SHORTCUTS
    if 'last_keypress' not in st.session_state:
        st.session_state.last_keypress = None
    if 'research_goals' not in st.session_state:
        st.session_state.research_goals = []
    if 'active_research_goal' not in st.session_state:
        st.session_state.active_research_goal = None

def setup_reference_manager():
    """Set up the reference manager integration."""
    st.sidebar.title("🔗 Reference Manager")
    
    # Select reference manager
    manager = st.sidebar.selectbox(
        "Select Reference Manager",
        [name for name, config in REFERENCE_MANAGERS.items() if config['enabled']],
        index=0
    )
    
    if manager == "Zotero":
        st.sidebar.subheader("Zotero Settings")
        api_key = st.sidebar.text_input("API Key", type="password")
        library_id = st.sidebar.text_input("Library ID")
        library_type = st.sidebar.selectbox(
            "Library Type", 
            ["user", "group"],
            index=0
        )
        
        if api_key and library_id:
            if st.sidebar.button("Connect to Zotero", key="connect_zotero_btn"):
                try:
                    # In a real app, you would verify the connection
                    st.session_state.reference_manager = {
                        'active': 'Zotero',
                        'connected': True,
                        'api_key': api_key,
                        'library_id': library_id,
                        'library_type': library_type,
                        'collections': []
                    }
                    st.sidebar.success("Connected to Zotero!")
                except Exception as e:
                    st.sidebar.error(f"Failed to connect: {str(e)}")
    
    # Show connection status
    if st.session_state.reference_manager.get('connected'):
        st.sidebar.success(f"✅ Connected to {st.session_state.reference_manager['active']}")
        
        # Sync collections
        if st.sidebar.button("🔄 Sync Collections", key="sync_collections_btn"):
            with st.spinner("Syncing collections..."):
                # In a real app, fetch collections from the reference manager
                time.sleep(1)  # Simulate API call
                st.session_state.reference_manager['collections'] = [
                    "My Collection",
                    "Research Papers",
                    "To Read"
                ]
                st.sidebar.success("Collections synced!")
        
        # Show collections
        if st.session_state.reference_manager['collections']:
            st.sidebar.subheader("Collections")
            for collection in st.session_state.reference_manager['collections']:
                st.sidebar.checkbox(collection, value=True)
    else:
        st.sidebar.warning("Not connected to any reference manager")

def save_to_reference_manager(paper: Dict):
    """Save a paper to the connected reference manager."""
    if not st.session_state.reference_manager.get('connected'):
        st.warning("Please connect to a reference manager first")
        return False
    
    try:
        # In a real app, this would call the reference manager's API
        if 'saved_papers' not in st.session_state:
            st.session_state.saved_papers = {}
        
        paper_id = paper.get('id')
        if paper_id not in st.session_state.saved_papers:
            st.session_state.saved_papers[paper_id] = {
                'paper': paper,
                'saved_at': datetime.now().isoformat(),
                'collections': ["My Collection"]  # Default collection
            }
            st.success(f"✅ Saved to {st.session_state.reference_manager['active']}")
            return True
        else:
            st.warning("This paper is already saved")
            return False
    except Exception as e:
        st.error(f"Failed to save paper: {str(e)}")
        return False

def render_pdf_viewer():
    """Render the PDF viewer component with enhanced caching and error handling."""
    st.header("PDF Viewer")
    
    if 'selected_paper' not in st.session_state or not st.session_state.selected_paper:
        st.warning("No paper selected. Please select a paper from the list.")
        return
    
    paper = st.session_state.selected_paper
    pdf_url = paper.get('pdf_url', '') if paper else ''
    
    if not pdf_url:
        st.error("No PDF URL available for this paper.")
        return
    
    # Generate a unique cache key for this PDF
    cache_key = f"pdf_viewer_{hashlib.md5(pdf_url.encode()).hexdigest()}"
    
    # Check if we have a cached version of the first page
    if cache_key not in st.session_state:
        try:
            # Try to render the first page as a preview
            with st.spinner("Loading PDF preview..."):
                img_data = pdf_processor.render_page_as_image(pdf_url, page_num=0)
                st.session_state[cache_key] = img_data
                
        except Exception as e:
            st.error(f"Error loading PDF preview: {str(e)}")
            return
    
    # Display the first page as a preview
    st.image(st.session_state[cache_key], use_column_width=True)
    
    # Add download button
    st.download_button(
        label="Download PDF",
        data=requests.get(pdf_url).content,
        file_name=pdf_url.split("/")[-1] or "paper.pdf",
        mime="application/pdf"
    )
    
    # Add option to view full PDF in a new tab
    st.markdown(f"""
    <a href="{pdf_url}" target="_blank" class="button">
        <button class="btn">
            Open Full PDF in New Tab
        </button>
    </a>
    <style>
    .button { text-decoration: none; }
    .btn {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .btn:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with navigation and filters."""
    with st.sidebar:
        st.title("🔍 Navigation")
        
        # Main navigation
        page = st.radio(
            "Go to",
            ["📚 Papers", "📈 Research Trends", "🔗 Citation Network", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Search filters
        st.subheader("Filters")
        
        # Category filter
        categories = sorted(list({
            cat for paper in st.session_state.papers 
            for cat in paper.get('categories', [])
        }))
        
        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=st.session_state.search_filters['categories']
        )
        
        # Date range
        st.write("Publication Date")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", value=st.session_state.search_filters['date_range'][0])
        with col2:
            end_date = st.date_input("To", value=st.session_state.search_filters['date_range'][1])
        
        # Apply filters
        if st.button("Apply Filters"):
            st.session_state.search_filters = {
                'categories': selected_categories,
                'date_range': (start_date, end_date),
                'sort_by': st.session_state.search_filters['sort_by']
            }
            st.rerun()
        
        # Reset filters
        if st.button("Reset Filters"):
            st.session_state.search_filters = {
                'categories': [],
                'date_range': (None, None),
                'sort_by': 'relevance'
            }
            st.rerun()
        
        st.markdown("---")
        
        # Reference manager
        setup_reference_manager()
        
        st.markdown("---")
        
        # User info and logout
        username = getattr(st.session_state.user, 'username', 'User')
        st.write(f"Logged in as: **{username}**")
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

def render_paper_list():
    """Render the main paper listing view."""
    st.title("📚 Research Papers")
    
    # Search bar
    search_query = st.text_input(
        "Search papers...",
        placeholder="Search by title, author, or keywords...",
        help="Use AND, OR, NOT for boolean search. Example: 'machine learning AND vision NOT nlp'"
    )
    
    # Sort options
    col1, col2 = st.columns([3, 1])
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Relevance", "Newest", "Most Cited"],
            index=0
        )
    
    # Apply search and filters
    if search_query or st.session_state.search_filters['categories'] or st.session_state.search_filters['date_range']:
        with st.spinner("Searching papers..."):
            # Prepare filters
            filters = {}
            if st.session_state.search_filters['categories']:
                filters['categories'] = st.session_state.search_filters['categories']
            
            # Parse date range if available
            date_range = st.session_state.search_filters.get('date_range', [])
            start_date = date_range[0] if date_range and len(date_range) > 0 else None
            end_date = date_range[1] if date_range and len(date_range) > 1 else None
            
            # Call search function with proper parameters
            filtered_papers, _ = search_papers(
                query=search_query,
                papers=st.session_state.papers,
                filters=filters,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_by.lower().replace(" ", "_")
            )
    else:
        filtered_papers = st.session_state.papers
    
    # Display papers
    if not filtered_papers:
        st.info("No papers found. Try adjusting your search or filters.")
        return
    
    # Pagination
    papers_per_page = 10
    total_pages = (len(filtered_papers) + papers_per_page - 1) // papers_per_page
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    # Display pagination controls
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Previous") and st.session_state.current_page > 0:
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
        with col3:
            if st.button("Next →") and st.session_state.current_page < total_pages - 1:
                st.session_state.current_page += 1
                st.rerun()
    
    # Display current page of papers
    start_idx = st.session_state.current_page * papers_per_page
    end_idx = min((st.session_state.current_page + 1) * papers_per_page, len(filtered_papers))
    
    for i, paper in enumerate(filtered_papers[start_idx:end_idx]):
        with st.expander(f"{i + start_idx + 1}. {paper.get('title', 'Untitled')}"):
            render_paper_card(paper)

def render_paper_card(paper: Dict):
    """Render a single paper card with actions."""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Paper title with link
        st.markdown(f"### [{paper.get('title', 'Untitled')}]({paper.get('pdf_url', '#')})")
        
        # Authors
        authors = ", ".join(paper.get('authors', ['Unknown']))
        st.caption(f"👥 {authors}")
        
        # Published date
        published = paper.get('published', '')
        if published:
            try:
                pub_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
                st.caption(f"📅 {pub_date.strftime('%B %d, %Y')}")
            except:
                st.caption(f"📅 {published}")
        
        # Abstract (collapsed by default)
        with st.expander("Abstract"):
            st.write(paper.get('summary', 'No abstract available.'))
        
        # Categories
        if 'categories' in paper and paper['categories']:
            categories = ", ".join(paper['categories'])
            st.caption(f"🏷️ {categories}")
        
        # Citations
        if 'citation_count' in paper:
            st.caption(f"📊 {paper['citation_count']} citations")
    
    with col2:
        # Save to reference manager
        if st.button("💾", key=f"save_{paper['id']}"):
            save_to_reference_manager(paper)
        
        # View PDF
        if st.button("📄", key=f"view_{paper['id']}"):
            st.session_state.selected_paper = paper
            st.rerun()
        
        # More actions
        with st.popover("⋮"):
            if st.button("📝 Summary", key=f"summary_{paper['id']}"):
                with st.spinner("Generating summary..."):
                    summary = get_paper_summary(paper)
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': f"Summary of '{paper['title']}':\n\n{summary}"
                    })
                    st.rerun()
            
            if st.button("❓ Explain", key=f"explain_{paper['id']}"):
                with st.spinner("Analyzing concepts..."):
                    explanation = get_concept_explanation(paper)
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': f"Key concepts in '{paper['title']}':\n\n{explanation}"
                    })
                    st.rerun()

def render_settings():
    """Render the settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("Appearance")
    dark_mode = st.toggle("Dark Mode", value=st.session_state.get('dark_mode', False))
    if dark_mode != st.session_state.get('dark_mode'):
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.subheader("Keyboard Shortcuts")
    for action, shortcut in st.session_state.keyboard_shortcuts.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            new_shortcut = st.text_input(
                f"{action}:",
                value=shortcut,
                key=f"shortcut_{action}"
            )
            if new_shortcut != shortcut:
                st.session_state.keyboard_shortcuts[action] = new_shortcut
                st.success(f"Shortcut for '{action}' updated to '{new_shortcut}'")
    
    st.subheader("Data Management")
    if st.button("Clear Cache"):
        # Clear cached data
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    st.warning("Changes to settings will be saved automatically.")

def show_auth_ui():
    """Show authentication UI (login/register forms)."""
    st.title("🔐 Welcome to arXiv Research Assistant")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form_1"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        # Create and store JWT token
                        from datetime import timedelta
                        from utils.auth import create_access_token, SECRET_KEY, ALGORITHM
                        access_token = create_access_token(
                            data={"sub": user.username},
                            expires_delta=timedelta(minutes=30)
                        )
                        st.session_state.user = access_token
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form_1"):
            st.subheader("Create an Account")
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Create a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            
            if st.form_submit_button("Register"):
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    user = register_user(new_username, new_email, new_password)
                    if user:
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed: Username already exists")

def apply_custom_styles():
    """Apply custom CSS styles to the app."""
    dark_mode = st.session_state.get('dark_mode', False)
    
    custom_css = """
    <style>
        /* Main container */
        .main {
            max-width: 1200px;
            padding: 2rem;
        }
        
        /* Paper cards */
        .paper-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .paper-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 4px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        /* Dark mode */
        .dark-mode .paper-card {
            background-color: #2d3748;
            border-color: #4a5568;
            color: #e2e8f0;
        }
        
        .dark-mode .stTextInput > div > div > input,
        .dark-mode .stSelectbox > div > div > div,
        .dark-mode .stTextArea > div > div > textarea {
            background-color: #2d3748;
            color: #e2e8f0;
            border-color: #4a5568;
        }
        
        .dark-mode .stMarkdown {
            color: #e2e8f0;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main {
                padding: 1rem;
            }
            
            .paper-card {
                padding: 1rem;
            }
        }
    </style>
    """
    
    # Apply dark mode class to body if dark mode is enabled
    if dark_mode:
        custom_css += """
        <script>
            document.body.classList.add('dark-mode');
        </script>
        """
    
    st.markdown(custom_css, unsafe_allow_html=True)

def render_main_content():
    """Render the main content area based on the selected page."""
    # Get the current page from the URL or default to "papers"
    page = st.query_params.get('page', 'papers')
    if isinstance(page, list):
        page = page[0] if page else 'papers'
    
    if page == 'papers':
        render_paper_list()
    elif page == 'trends':
        render_research_trends()
    elif page == 'citations':
        render_citation_network()
    elif page == 'settings':
        render_settings()
    else:
        st.warning("Page not found")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Set page config
    st.set_page_config(
        page_title="arXiv Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Apply custom styles
    apply_custom_styles()
    
    # Check authentication
    if 'user' not in st.session_state:
        show_auth_ui()
    else:
        # Main layout
        render_sidebar()
        render_main_content()

def main():
    # Initialize session state
    init_session_state()
    
    # Set page config
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply custom CSS
    apply_custom_styles()
    
    # Setup keyboard shortcuts
    setup_keyboard_shortcuts()
    
    # Check authentication
    if 'user' not in st.session_state:
        show_auth_ui()
        return
        
    # Main layout
    render_sidebar()
    render_main_content()

def setup_keyboard_shortcuts():
    """Handle keyboard shortcuts for the application."""
    # This is a placeholder for actual keyboard shortcut handling
    # In a real app, you would use a library like streamlit-shortcuts
    pass

def render_research_trends():
    """Render the research trends analysis view."""
    st.title("📈 Research Trends")
    
    if not st.session_state.papers:
        st.warning("No papers loaded. Please load papers first.")
        return
    
    # Initialize trend analyzer if not already done
    if st.session_state.trend_analyzer is None:
        with st.spinner("Analyzing research trends..."):
            st.session_state.trend_analyzer = ResearchTrendAnalyzer(st.session_state.papers)
    
    trend_analyzer = st.session_state.trend_analyzer
    
    # Tabs for different trend visualizations
    tab1, tab2, tab3 = st.tabs(["Publication Trends", "Topic Analysis", "Author Network"])
    
    with tab1:
        st.subheader("Publication Trends Over Time")
        fig = trend_analyzer.plot_yearly_trends()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Papers", len(st.session_state.papers))
            with col2:
                years = trend_analyzer.df['year'].unique()
                st.metric("Years Covered", f"{min(years)} - {max(years)}" if len(years) > 0 else "N/A")
            with col3:
                avg_papers = len(st.session_state.papers) / len(years) if len(years) > 0 else 0
                st.metric("Avg. Papers/Year", f"{avg_papers:.1f}")
            
            # Category distribution
            st.subheader("Category Distribution")
            cat_fig = trend_analyzer.plot_category_distribution()
            if cat_fig:
                st.plotly_chart(cat_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Topic Modeling")
        
        num_topics = st.slider("Number of Topics", 3, 10, 5, 1,
                             help="Adjust the number of topics to discover in the papers.")
        
        if st.button("Analyze Topics"):
            with st.spinner("Analyzing topics (this may take a minute)..."):
                topics_df, _ = trend_analyzer.topic_modeling(num_topics=num_topics)
                
                if topics_df is not None and not topics_df.empty:
                    # Display topics in a table
                    st.dataframe(
                        topics_df[['topic_id', 'top_words']],
                        column_config={
                            'topic_id': "Topic ID",
                            'top_words': "Top Keywords"
                        },
                        use_container_width=True
                    )
                    
                    # Show topic evolution
                    st.subheader("Topic Evolution Over Time")
                    evol_fig = trend_analyzer.plot_topic_evolution(num_topics=num_topics)
                    if evol_fig:
                        st.plotly_chart(evol_fig, use_container_width=True)
                else:
                    st.warning("Could not extract topics from the papers.")
        else:
            st.info("Click 'Analyze Topics' to discover topics in the papers.")
    
    with tab3:
        st.subheader("Author Collaboration Network")
        
        min_collaborations = st.slider(
            "Minimum Collaborations",
            min_value=1,
            max_value=10,
            value=2,
            help="Show authors with at least this many collaborations"
        )
        
        if st.button("Generate Network"):
            with st.spinner("Generating author network..."):
                network_html = trend_analyzer.get_author_network(
                    min_collaborations=min_collaborations
                )
                
                if network_html:
                    st.components.v1.html(network_html, height=600, scrolling=True)
                else:
                    st.warning("Could not generate author network.")
        else:
            st.info("Click 'Generate Network' to visualize author collaborations.")

def render_citation_network():
    """Render the citation network visualization."""
    st.title("🔗 Citation Network")
    
    if not st.session_state.papers:
        st.warning("No papers loaded. Please load papers first.")
        return
    
    st.markdown("""
    This visualization shows how papers in your collection cite each other.
    - **Nodes** represent papers
    - **Edges** represent citations between papers
    - **Node size** indicates number of citations
    - **Colors** represent different paper categories
    """)
    
    # Add controls for the visualization
    col1, col2 = st.columns(2)
    with col1:
        max_papers = st.slider(
            "Maximum Papers to Show",
            min_value=10,
            max_value=min(100, len(st.session_state.papers)),
            value=min(50, len(st.session_state.papers)),
            step=5
        )
    
    with col2:
        min_citations = st.slider(
            "Minimum Citations",
            min_value=0,
            max_value=20,
            value=1,
            help="Only show papers with at least this many citations"
        )
    
    if st.button("Generate Citation Network"):
        with st.spinner("Generating citation network (this may take a minute)..."):
            # Get top papers by citation count
            sorted_papers = sorted(
                st.session_state.papers,
                key=lambda x: len(x.get('citations', [])),
                reverse=True
            )
            
            # Filter papers
            filtered_papers = [
                p for p in sorted_papers[:max_papers]
                if len(p.get('citations', [])) >= min_citations
            ]
            
            if not filtered_papers:
                st.warning("No papers match the selected criteria.")
                return
            
            # Generate the network visualization
            network_html = plot_citation_network(filtered_papers)
            
            if network_html:
                # Display the network
                st.components.v1.html(network_html, height=800, scrolling=True)
                
                # Add download button for the network data
                st.download_button(
                    label="Download Network Data (GEXF)",
                    data=json.dumps({
                        'papers': filtered_papers,
                        'generated_at': datetime.now().isoformat()
                    }),
                    file_name="citation_network.json",
                    mime="application/json"
                )
            else:
                st.error("Failed to generate citation network. Please try again.")
    else:
        st.info("Click 'Generate Citation Network' to visualize paper citations.")

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            padding: 2rem;
        }
        .paper-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: box-shadow 0.3s ease;
        }
        .paper-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .search-filters {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {
            'categories': [],
            'authors': [],
            'date_range': (None, None),
            'sort_by': 'relevance'
        }
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

def render_auth_ui():
    """Render authentication UI (login/register)."""
    st.title("🔐 Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form_2"):
            st.subheader("Login to Your Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                user = authenticate_user(username, password)
                if user:
                    # Create and store JWT token
                    from datetime import timedelta
                    from utils.auth import create_access_token, SECRET_KEY, ALGORITHM
                    access_token = create_access_token(
                        data={"sub": user.username},
                        expires_delta=timedelta(minutes=30)
                    )
                    st.session_state.user = access_token
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form_2"):
            st.subheader("Create a New Account")
            new_username = st.text_input("Choose a username")
            email = st.text_input("Email address")
            new_password = st.text_input("Create a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            
            if st.form_submit_button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    user = register_user(new_username, email, new_password)
                    if user:
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Check authentication
    user = get_current_user()
    
    # If not authenticated, show login/register screen
    if not user:
        render_auth_ui()
        return
    
    # Main app layout
    st.title("📚 arXiv Research Assistant")
    st.write(f"Welcome back, {user.get('username', 'User')}!")
    
    # Navigation
    pages = {
        "📚 Papers": "papers",
        "🔍 Search": "search",
        "📂 Saved": "saved",
        "⚙️ Settings": "settings"
    }
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Logout button in sidebar
    if st.sidebar.button("🚪 Logout", key="logout_btn"):
        if 'user' in st.session_state:
            del st.session_state.user
        st.rerun()
    
    # Load papers if not already loaded
    if not st.session_state.papers and 'load_papers' not in st.session_state:
        st.session_state.load_papers = True
        with st.spinner("Loading papers..."):
            st.session_state.papers = load_arxiv_data("cs")  # Default to Computer Science
    
    # Render the selected page
    if selected_page == "📚 Papers" or 'viewing_pdf' in st.session_state:
        if 'viewing_pdf' in st.session_state and st.session_state.viewing_pdf:
            render_pdf_viewer()
        elif 'selected_paper' in st.session_state and st.session_state.selected_paper:
            render_paper_detail()
        else:
            render_paper_list()
    
    elif selected_page == "🔍 Search":
        st.title("🔍 Advanced Search")
        search_query = render_search_sidebar()
        
        if st.button("Search"):
            st.session_state.search_query = search_query
            st.session_state.current_page = 0
            st.rerun()
        
        if 'search_query' in st.session_state and st.session_state.search_query:
            render_paper_list()
    
    elif selected_page == "📂 Saved":
        st.title("📂 Saved Papers")
        st.info("Your saved papers will appear here.")
    
    elif selected_page == "⚙️ Settings":
        st.title("⚙️ Settings")
        
        # Theme settings
        st.subheader("Appearance")
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        # Data management
        st.subheader("Data Management")
        if st.button("Clear Cache"):
            st.session_state.papers = []
            st.session_state.load_papers = True
            st.success("Cache cleared. Reloading papers...")
            st.rerun()

def render_search_sidebar():
    """Render the search sidebar with filters."""
    with st.sidebar:
        st.title("🔍 Search Papers")
        
        # Search query
        search_query = st.text_input(
            "Search papers...",
            placeholder="e.g., 'machine learning' OR 'deep learning'"
        )
        
        # Advanced filters
        with st.expander("Advanced Filters", expanded=False):
            # Category filter
            categories = sorted(list({
                cat for paper in st.session_state.papers 
                for cat in paper.get('categories', [])
            }))
            
            selected_categories = st.multiselect(
                "Categories",
                options=categories,
                default=st.session_state.search_filters['categories']
            )
            
            # Author filter
            authors = sorted(list({
                author for paper in st.session_state.papers 
                for author in paper.get('authors', [])
            }))
            
            selected_authors = st.multiselect(
                "Authors",
                options=authors,
                default=st.session_state.search_filters['authors']
            )
            
            # Date range
            st.write("Publication Date")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", value=None)
            with col2:
                end_date = st.date_input("To", value=None)
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                ["Relevance", "Date (Newest First)", "Date (Oldest First)", "Citations"],
                index=0
            )
        
        # Apply filters
        if st.button("Apply Filters"):
            st.session_state.search_filters = {
                'categories': selected_categories,
                'authors': selected_authors,
                'date_range': (start_date, end_date),
                'sort_by': sort_by.lower().split()[0]  # Get first word in lowercase
            }
            st.rerun()
        
        # Reset filters
        if st.button("Reset Filters"):
            st.session_state.search_filters = {
                'categories': [],
                'authors': [],
                'date_range': (None, None),
                'sort_by': 'relevance'
            }
            st.rerun()
    
    return search_query

def render_paper_list():
    """Render the list of papers."""
    st.subheader("Research Papers")
    
    # Initialize papers if not in session state
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    
    # Show loading state if papers are being loaded
    if not st.session_state.papers:
        with st.spinner("Loading papers..."):
            st.session_state.papers = load_arxiv_data("cs.CL")  # Load default category
            if not st.session_state.papers:
                st.error("Failed to load papers. Please try again later.")
                return
    
    # Get search query and filters
    search_query = st.session_state.get('search_query', '').strip()
    filters = {
        'category': st.session_state.search_filters.get('categories', []),
        'author': st.session_state.search_filters.get('authors', [])
    }
    
    # Get date range and sort order
    date_range = st.session_state.search_filters.get('date_range', (None, None))
    start_date, end_date = date_range if date_range else (None, None)
    sort_by = st.session_state.search_filters.get('sort_by', 'date')
    
    try:
        # Search papers with the current filters
        results, total = search_papers(
            query=search_query,
            papers=st.session_state.papers,
            filters=filters if any(filters.values()) else None,
            sort_by=sort_by,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        st.error(f"Error searching papers: {str(e)}")
        return
    
    # Show search summary
    st.caption(f"Found {total} papers matching your criteria")
    
    # Pagination
    page_size = 5
    total_pages = (total + page_size - 1) // page_size
    current_page = st.session_state.get('current_page', 0)
    
    if total_pages > 1:
        col1, col2, _ = st.columns([1, 2, 5])
        with col1:
            if st.button("⬅️ Previous") and current_page > 0:
                st.session_state.current_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {current_page + 1} of {total_pages}")
        with col1:
            if st.button("Next ➡️") and current_page < total_pages - 1:
                st.session_state.current_page += 1
                st.rerun()
    
    # Display papers with pagination
    start_idx = current_page * page_size
    end_idx = min((current_page + 1) * page_size, len(results))
    
    # Use enumerate to get both index and paper for unique keys
    for i, paper in enumerate(results[start_idx:end_idx], start=start_idx):
        with st.container():
            st.markdown(f"### {paper.get('title', 'Untitled')}")
            
            # Create a unique key for this paper's UI elements using the loop index
            paper_key = f"{paper.get('id', '')}_{i}"
            
            # Paper metadata
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                st.caption(f"**Published:** {paper.get('published', 'N/A')}")
                st.caption(f"**Categories:** {', '.join(paper.get('categories', []))}")
            
            with col2:
                if st.button("View Details", key=f"view_{paper_key}"):
                    st.session_state.selected_paper = paper
                    st.rerun()
            
            # Paper summary
            st.write(paper.get('summary', 'No summary available')[:300] + "...")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📄 View PDF", key=f"pdf_{paper_key}"):
                    st.session_state.viewing_pdf = paper.get('pdf_url')
                    st.rerun()
            with col2:
                st.button("💾 Save", key=f"save_{paper_key}")
            with col3:
                st.button("📝 Cite", key=f"cite_{paper_key}")
            
            st.divider()

def render_paper_detail():
    """Render the detailed view of a selected paper."""
    paper = st.session_state.selected_paper
    
    if not paper:
        st.warning("No paper selected")
        if st.button("Back to List"):
            st.session_state.selected_paper = None
            st.rerun()
        return
    
    # Back button
    if st.button("← Back to Results"):
        st.session_state.selected_paper = None
        st.rerun()
    
    # Paper header
    st.title(paper.get('title', 'Untitled'))
    
    # Paper metadata
    st.caption(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
    st.caption(f"**Published:** {paper.get('published', 'N/A')}")
    st.caption(f"**Categories:** {', '.join(paper.get('categories', []))}")
    
    # Action buttons
    col1, col2, col3, _ = st.columns(4)
    with col1:
        if st.button("📄 View PDF"):
            st.session_state.viewing_pdf = paper.get('pdf_url')
            st.rerun()
    with col2:
        st.download_button(
            label="⬇️ Download PDF",
            data=paper.get('pdf_url', ''),
            file_name=f"{paper.get('id', 'paper')}.pdf",
            mime="application/pdf"
        )
    with col3:
        st.button("📝 Cite")
    
    st.divider()
    
    # Paper content
    st.subheader("Abstract")
    st.write(paper.get('summary', 'No abstract available'))
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Citations", "Annotations"])
    
    with tab1:
        # Generate and display summary
        with st.spinner("Generating summary..."):
            summary = get_paper_summary(paper)
            st.write(summary)
    
    with tab2:
        # Display citations (placeholder)
        st.info("Citation data will be displayed here")
    
    with tab3:
        # Annotations
        st.subheader("Your Annotations")
        
        # Add new annotation
        with st.form("add_annotation"):
            annotation_text = st.text_area("Add a note about this paper")
            col1, col2 = st.columns(2)
            with col1:
                annotation_page = st.number_input("Page", min_value=1, value=1)
            with col2:
                annotation_color = st.color_picker("Color", "#FFEB3B")
            
            if st.form_submit_button("Save Annotation"):
                if annotation_text:
                    # In a real app, this would save to a database
                    # Initialize annotations if not exists
                    if 'annotations' not in st.session_state:
                        st.session_state.annotations = []
                    
                    # Ensure annotations is a list
                    if not isinstance(st.session_state.annotations, list):
                        st.session_state.annotations = []
                    
                    # Create new annotation
                    annotation = {
                        'id': str(hash(f"{paper.get('id', '')}_{time.time()}")),
                        'paper_id': paper.get('id', ''),
                        'page': int(annotation_page),
                        'content': str(annotation_text),
                        'color': str(annotation_color),
                        'created_at': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # Add to annotations list
                    st.session_state.annotations = st.session_state.annotations + [annotation]
                    st.success("Annotation saved!")
                    st.rerun()
        
        # Display existing annotations
        if 'annotations' in st.session_state and st.session_state.annotations:
            # Ensure we have a list and handle potential dict format
            annotations = st.session_state.annotations
            if isinstance(annotations, dict):
                annotations = list(annotations.values())
            
            # Filter annotations for current paper
            paper_annotations = [
                a for a in annotations 
                if isinstance(a, dict) and a.get('paper_id') == paper.get('id')
            ]
            
            if paper_annotations:
                for ann in sorted(paper_annotations, key=lambda x: x.get('page', 0)):
                    with st.container():
                        st.markdown(
                            f"<div class='annotation' style='border-left-color: {ann.get('color', '#FFEB3B')}'>"
                            f"<small>Page {ann.get('page', 1)}</small>"
                            f"<p>{ann.get('content', '')}</p>"
                            f"<small style='color: #666'>{ann.get('created_at', '')}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Delete button
                        if st.button("Delete", key=f"del_{ann.get('id')}"):
                            # Safely remove annotation
                            annotations = st.session_state.annotations
                            if isinstance(annotations, dict):
                                annotations = {k: v for k, v in annotations.items() 
                                             if v.get('id') != ann.get('id')}
                            else:
                                annotations = [a for a in annotations 
                                            if isinstance(a, dict) and a.get('id') != ann.get('id')]
                            st.session_state.annotations = annotations
                            st.rerun()
            else:
                st.info("No annotations yet. Add one above!")
        else:
            st.info("No annotations yet. Add one above!")

def download_pdf(pdf_url: str, cache_dir: str = "./data/cache") -> Optional[str]:
    """
    Download a PDF from a URL and save it to the cache directory.
    
    Args:
        pdf_url: URL of the PDF to download
        cache_dir: Directory to cache downloaded PDFs
        
    Returns:
        Path to the downloaded PDF file or None if download failed
    """
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate a filename from the URL
        filename = hashlib.md5(pdf_url.encode('utf-8')).hexdigest() + ".pdf"
        filepath = os.path.join(cache_dir, filename)
        
        # If file already exists, return the cached path
        if os.path.exists(filepath):
            return filepath
            
        # Download the PDF
        response = requests.get(pdf_url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Save the PDF
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return filepath
        
    except Exception as e:
        st.error(f"Error downloading PDF: {str(e)}")
        return None

def render_pdf_viewer():
    """Render the PDF viewer component."""
    pdf_url = st.session_state.viewing_pdf
    if not pdf_url:
        st.warning("No PDF URL provided")
        if st.button("Back to Paper"):
            st.session_state.viewing_pdf = None
            st.rerun()
        return
    
    # Back button
    if st.button("← Back to Paper"):
        st.session_state.viewing_pdf = None
        st.rerun()
    
    st.title("PDF Viewer")
    
    # Download and load the PDF
    with st.spinner("Loading PDF..."):
        try:
            pdf_path = download_pdf(pdf_url)
            if not pdf_path or not os.path.exists(pdf_path):
                st.error("Failed to load PDF")
                return
            
            # Get PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Get current page from session state or default to first page
            current_page = st.session_state.get('current_pdf_page', 0)
            current_page = max(0, min(current_page, total_pages - 1))  # Ensure within bounds
            
            # Display page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⏮ Previous") and current_page > 0:
                    st.session_state.current_pdf_page = current_page - 1
                    st.rerun()
            with col2:
                st.write(f"Page {current_page + 1} of {total_pages}")
            with col3:
                if st.button("Next ⏭") and current_page < total_pages - 1:
                    st.session_state.current_pdf_page = current_page + 1
                    st.rerun()
            
            # Display the current page
            page = doc.load_page(current_page)
            zoom = 1.5  # Zoom factor for better readability
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to image
            img_data = pix.tobytes("png")
            st.image(img_data, use_column_width=True)
            
            # Add some space
            st.markdown("---")
            
            # Page navigation
            col1, col2 = st.columns([1, 4])
            with col1:
                page_num = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1
                )
            
            # Display current page
            with col2:
                st.caption(f"Page {page_num} of {total_pages}")
            
            # Navigation buttons
            col1, col2, col3, _ = st.columns(4)
            with col1:
                if st.button("⏮️ First") and page_num > 1:
                    page_num = 1
                    st.rerun()
            with col2:
                if st.button("⬅️ Previous") and page_num > 1:
                    page_num -= 1
                    st.rerun()
            with col3:
                if st.button("Next ➡️") and page_num < total_pages:
                    page_num += 1
                    st.rerun()
            
            # Render the PDF page
            st.markdown("---")
            
            # Display PDF page
            image = render_pdf_page(pdf_path, page_num - 1)  # 0-based index
            st.image(
                image,
                use_column_width=True,
                caption=f"Page {page_num} of {total_pages}"
            )
            
            # Search in PDF
            with st.expander("🔍 Search in PDF"):
                search_query = st.text_input("Search in this PDF")
                if search_query:
                    matches = search_in_pdf(pdf_path, search_query)
                    if matches:
                        st.write(f"Found {len(matches)} matches:")
                        for i, (page, text) in enumerate(matches[:5]):  # Show first 5 matches
                            with st.expander(f"Page {page + 1}"):
                                st.write(text[:500] + ("..." if len(text) > 500 else ""))
                                if st.button(f"Go to page {page + 1}", key=f"go_to_{i}"):
                                    page_num = page + 1
                                    st.rerun()
                    else:
                        st.info("No matches found")
            
            # Annotation tools
            with st.expander("📝 Annotate"):
                if 'user' not in st.session_state:
                    st.warning("Please log in to add annotations")
                else:
                    with st.form("add_annotation"):
                        annotation_text = st.text_area("Add an annotation for this page")
                        annotation_color = st.color_picker("Color", "#FFEB3B")
                        
                        if st.form_submit_button("Save Annotation"):
                            if annotation_text:
                                annotation = {
                                    'paper_id': st.session_state.selected_paper.get('id') if 'selected_paper' in st.session_state else None,
                                    'page': page_num - 1,
                                    'content': annotation_text,
                                    'color': annotation_color,
                                    'created_at': datetime.now().isoformat(),
                                    'updated_at': datetime.now().isoformat()
                                }
                                
                                save_annotation(annotation)
                                st.success("Annotation saved!")
            
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()

if __name__ == "__main__":
    load_dotenv()
    main()
