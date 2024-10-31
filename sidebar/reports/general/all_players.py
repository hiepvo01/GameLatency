import streamlit as st
import pandas as pd
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit_toggle as tog

input_path = f"{PROCESSED_DATA_FOLDER}/player_performance.csv" ##path
sb_summary = f"{PROCESSED_DATA_FOLDER}/round_summary_adjusted.csv" ##path

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(input_path)
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

def create_dual_line_chart(player_data):
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    
    # Add score line
    fig.add_trace(
        go.Scatter(x=player_data['game_round'], y=player_data['score'], mode='lines', name='Score', line=dict(color='blue')),
        secondary_y=False,
    )
    
    # Add deaths line
    deaths_data = df[(df['victim_ip'] == player_data['player_ip'].iloc[0]) & (df['game_round'].isin(player_data['game_round']))]
    deaths_per_round = deaths_data.groupby('game_round')['deaths_total'].sum().reindex(player_data['game_round']).fillna(0)
    
    fig.add_trace(
        go.Scatter(x=player_data['game_round'], y=deaths_per_round, mode='lines', name='Deaths', line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_layout(
        height=200,
        width=300,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    fig.update_xaxes(title_text="Rounds")
    fig.update_yaxes(title_text="Score", secondary_y=False, title_standoff=0)
    fig.update_yaxes(title_text="Deaths", secondary_y=True, title_standoff=0)
    
    return fig

# Get all players
all_players = sb['player_ip'].unique()

# Create a container for the scrollable area
player_container = st.container()

def show_all_players():
    
    
    # st.title("All Players")
    # col1, col2, col3 = st.columns(3)
    # col4, col5, col6 = st.columns(3)

    # for i, player in enumerate(all_players):
    #     player_data = sb[sb['player_ip'] == player]
    #     total_score = player_data['score'].sum()
    #     avg_score = player_data['score'].mean()
    #     total_deaths = df[(df['victim_ip'] == player) & (df['game_round'].isin(player_data['game_round']))]['deaths_total'].sum()

    #     # Determine which column to place the player info in
    #     col = [col1, col2, col3, col4, col5, col6][i % 6]
        
    #     with col:
    #         st.subheader(f"**{player}**", divider="gray")
    #         st.write(f"Total Score: {total_score:.0f}")
    #         st.write(f"Avg Score: {avg_score:.2f}")
    #         st.write(f"Total Deaths: {total_deaths:.0f}")
    #         st.plotly_chart(create_dual_line_chart(player_data), use_container_width=True)

    # Load data
    @st.cache_data
    def load_data():
        df = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/player_performance.csv")
        sb = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/round_summary_adjusted.csv")
        return df, sb

    df, sb = load_data()

    # Page title
    st.title("Player Performance")

    # Sidebar
    st.sidebar.subheader("Select View")

    # Radio options for line display
    display_option = st.sidebar.radio(
        "Select data to display:",
        ("Score", "Deaths"), label_visibility="collapsed"
    )

    # Get all players
    all_players = sb['player_ip'].unique()

    st.sidebar.subheader("Select Players")
    # Create checkboxes for player selection
    selected_players = []
    for player in all_players:
        if st.sidebar.checkbox(player, value=(player in all_players[:5])):  # Default to first 5 players checked
            selected_players.append(player)

    # Create the big graph
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for player in selected_players:
        player_data = sb[sb['player_ip'] == player]
        
        if display_option in ["Score"]:
            fig.add_trace(
                go.Scatter(x=player_data['game_round'], y=player_data['score'], mode='lines', name=f'{player}'),
                secondary_y=False,
            )
        
        if display_option in ["Deaths"]:
            deaths_data = df[(df['victim_ip'] == player) & (df['game_round'].isin(player_data['game_round']))]
            deaths_per_round = deaths_data.groupby('game_round')['deaths_total'].sum().reindex(player_data['game_round']).fillna(0)
            
            fig.add_trace(
                go.Scatter(x=player_data['game_round'], y=deaths_per_round, mode='lines', name=f'{player}'),
                secondary_y=True,
            )

    fig.update_layout(
        height=600,
        title_text="Player Performance Over Rounds",
        legend_title_text="Player Data"
    )

    fig.update_xaxes(title_text="Rounds")
    fig.update_yaxes(title_text="Score", secondary_y=False)
    fig.update_yaxes(title_text="Deaths", secondary_y=True)

    # Display the big graph
    st.plotly_chart(fig, use_container_width=True)

    # Create and display the table
    st.header("Key Player Statistics")

    table_data = []
    for player in all_players:
        player_data = sb[sb['player_ip'] == player]
        total_score = player_data['score'].sum()
        avg_score = player_data['score'].mean()
        
        deaths_data = df[(df['victim_ip'] == player) & (df['game_round'].isin(player_data['game_round']))]
        total_deaths = deaths_data['deaths_total'].sum()
        avg_deaths = total_deaths / len(player_data) if len(player_data) > 0 else 0
        
        table_data.append({
            "Player": player,
            "Total Score": f"{total_score:.0f}",
            "Avg Score": f"{avg_score:.2f}",
            "Total Deaths": f"{total_deaths:.0f}",
            "Avg Deaths": f"{avg_deaths:.2f}"
        })

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df.set_index('Player'), hide_index=False)    