import streamlit as st
from .credits import show_credits
from .how_to_navigation import show_navigation
from .how_to_read_log import show_read_logs
from .how_to_participate import show_how_to_participate

def show_support():
    st.title("Support")

    with st.expander("🧭 How to Navigate the Site"):
        show_navigation()
        
    with st.expander("🙋🏽 How To Participate In The Experiment"):
        show_how_to_participate()

    with st.expander("📜 How to Read the Game Logs"):
        show_read_logs()

    with st.expander("👏 Credits"):
        show_credits()

    st.info("Click on each section above to expand or collapse its content.")
