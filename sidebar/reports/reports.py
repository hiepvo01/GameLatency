import streamlit as st
from streamlit_option_menu import option_menu
from sidebar.reports.general import latency as general_latency
from sidebar.reports.general import round as general_round
from sidebar.reports.player_specific import player as player_specific
from .general.latency import show_latency
from .general.round import show_round
from .player_specific.player import show_individual_player
from .general.all_players import show_all_players

# st.set_page_config(page_title="Game Analysis Reports", page_icon="📊", layout="wide")

def show_reports():
    st.title("Reports")
    
    # Horizontal options menu
    selected_category = option_menu(
        menu_title=None,
        options=["General", "Player-specific"],
        icons=["graph-up", "person"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected_category == "General":
        # Select box for General category
        general_option = st.selectbox(
            "Select analysis type",
            options=["Latency", "Round", "Player Performance"],
            index=0  # Default to Latency
        )

        if general_option == "Latency":
            show_latency()
        elif general_option == "Round":
            show_round()
        elif general_option == "Player Performance":
            show_all_players()

    elif selected_category == "Player-specific":
        # Select box for Player-specific category
        player_option = st.selectbox(
            "Select analysis type",
            options=["Player Performance"],
            index=0  # Only one option, so default index is 0
        )

        if player_option == "Player Performance":
            show_individual_player()
