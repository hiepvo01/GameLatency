import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import importlib
import sys
import os
from pathlib import Path

# Add the project root directory to Python's module search path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from sidebar.home.welcome import show_welcome
from sidebar.demographic.dg import show_demographic
from sidebar.reports.reports import show_reports
from sidebar.support.support import show_support
from sidebar.analysis.analysis import show_analysis

pages = [
    {"name": "Home", "icon": "house"},
    {"name": "Demographic", "icon": "people"},
    {"name": "Reports", "icon": "clipboard-data"},
    {"name": "Analysis", "icon": "graph-up-arrow"},
    {"name": "Support", "icon": "question-circle"}
]

def display_page(page_name):
    if page_name == "Home":
        show_welcome()
    elif page_name == "Demographic":
        show_demographic()
    elif page_name == "Reports":
        show_reports()
    elif page_name == "Analysis":
        show_analysis()
    elif page_name == "Support":
        show_support()

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar navigation
with st.sidebar:
    st.title("Main Menu")
    
    selected_page = option_menu(
        menu_title=None,
        options=[page["name"] for page in pages],
        icons=[page["icon"] for page in pages],
        menu_icon="cast",
        default_index=next((i for i, page in enumerate(pages) if page["name"] == st.session_state.current_page), 0),
    )

    # Update current page if selection changes
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

# Display the selected page
display_page(st.session_state.current_page)

def add_footer():
    year = datetime.now().year
    footer = f"""
    <div class="footer" style="text-align: center;">
        <hr style="margin-top: 50px; border: 0; border-top: 1px solid #ccc;">
        <p>© {year} University of Technology Sydney. All rights reserved.</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

add_footer()