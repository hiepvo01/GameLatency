import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Round Scoreboard", page_icon="🏆", layout="wide")

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

# Page title
st.title("Round Scoreboard")

# Data preparation
select_latency = df['latency'].unique()
select_latency.sort()
select_round = df['game_round'].unique()

# Round selection
option_round = st.selectbox("Select a Round", [""] + list(select_round), key='round_select', index=1)

if option_round:
    round_scoreboard = sb[sb['game_round'] == int(option_round)]
    map_name = round_scoreboard['map'].iloc[0] if not round_scoreboard.empty else "N/A"
    latency = round_scoreboard['latency'].iloc[0] if not round_scoreboard.empty else "N/A"

    st.subheader(f"Round {option_round} Scoreboard")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Map: {map_name}")
    with col2:
        st.write(f"Latency: {latency} ms")
    
    round_scoreboard = round_scoreboard.drop(columns=['game_round', 'map', 'latency'])

    # Create an interactive bar chart for the round scoreboard
    fig = px.bar(round_scoreboard, x='player_id', y='score',
                 title=f'Score by Player ID for Round {option_round}',
                 labels={'player_id': 'Player ID', 'score': 'Score'},
                 height=500)
    fig.update_layout(
        xaxis_title='Player ID',
        yaxis_title='Score',
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sort the table by score in descending order and display it
    round_scoreboard_sorted = round_scoreboard.sort_values(by='score', ascending=False)
    st.subheader("Detailed Scoreboard")
    st.dataframe(round_scoreboard_sorted)

    # Additional information
    st.write("Note: This page displays the scoreboard for a selected round, including an interactive bar chart of scores and a detailed table.")

else:
    st.warning("Please select a round to view the scoreboard.")

# Display available columns
# st.subheader("Available Data Columns")
# st.write(f"Columns in the player performance dataset: {', '.join(df.columns)}")
# st.write(f"Columns in the round summary dataset: {', '.join(sb.columns)}")