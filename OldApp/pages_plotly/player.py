import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Player Performance", page_icon="👤", layout="wide")

# File paths
source = "final-data/player_performance.csv"
sb_summary = "final-data/round_summary_adjusted.csv"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(source)
    sb = pd.read_csv(sb_summary)
    return df, sb

df, sb = load_data()

# Data preparation
select_maps = df['map'].unique()
select_latency = df['latency'].unique()
select_latency.sort()
select_player = df['player_ip'].unique()
select_round = df['game_round'].unique()

# Page title
st.title("Player Performance")

# Player selection
option_player = st.selectbox("Select a Player", [""] + list(select_player), index=1)

# Create two columns for the charts
col1, col2 = st.columns(2)

with col1:
    player_pts = sb[sb['player_ip'] == option_player]
    st.subheader(f"{option_player} Points Earned Overview")
    
    player_pts = player_pts.drop(columns=['player_id', 'player_ip'])

    if not player_pts.empty:
        # Create an interactive bar chart for the round scoreboard
        fig = px.bar(player_pts, x='game_round', y='score',
                     title=f'Points Earned Per Round',
                     labels={'game_round': 'Game Round', 'score': 'Score'},
                     height=400)
        fig.update_layout(xaxis_title='Game Round', yaxis_title='Score')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sort the table by score in descending order and display it
        player_pts_sorted = player_pts.sort_values(by='score', ascending=False)
        st.write(player_pts_sorted)
        
with col2:
    player_deaths = sb[sb['player_ip'] == option_player]
    st.subheader(f"{option_player} Death Count Overview")
    
    player_deaths = player_deaths.drop(columns=['player_id', 'player_ip'])

    if not player_deaths.empty:
        # Create an interactive bar chart for the death count
        fig = px.bar(player_deaths, x='game_round', y='score',
                     title=f'Number of Deaths Per Round',
                     labels={'game_round': 'Game Round', 'score': 'Death Count'},
                     height=400)
        fig.update_layout(xaxis_title='Game Round', yaxis_title='Death Count')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sort the table by score in descending order and display it
        player_deaths_sorted = player_deaths.sort_values(by='score', ascending=False)
        st.write(player_deaths_sorted)

# Additional information
# st.write("Note: This page provides an overview of individual player performance across different game rounds. "
#          "The 'score' column is used to represent both points and death count in this analysis.")