import streamlit as st
import pandas as pd
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_individual_player():
    # File paths
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
    st.sidebar.subheader("Players")
    option_player = st.sidebar.radio("Select a Player", [""] + list(select_player), index=1, label_visibility="collapsed")

    # Calculate key statistics
    if option_player:
        player_pts = sb[sb['player_ip'] == option_player]
        player_deaths = df[df['victim_ip'] == option_player]
        
        # Merge the data for combined analysis
        combined_data = pd.merge(player_pts, player_deaths.groupby('game_round')['deaths_total'].sum().reset_index(), 
                                 on='game_round', how='outer').fillna(0)
        
        # Highest score in a single round
        highest_score_round = combined_data.loc[combined_data['score'].idxmax()]
        
        # Most deaths in a single round
        most_deaths_round = combined_data.loc[combined_data['deaths_total'].idxmax()]
        
        # Best map (highest accumulated points)
        best_map_data = player_pts.groupby('map')['score'].sum()
        best_map = best_map_data.idxmax()
        best_map_score = best_map_data.max()
        
        # Player killed by most
        killer_counts = player_deaths['killer_ip'].value_counts()
        top_killer = killer_counts.index[0] if not killer_counts.empty else "N/A"
        top_killer_count = killer_counts.iloc[0] if not killer_counts.empty else 0

        # Display key statistics
        st.header(f"{option_player} Highlights")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Best Performance")
            st.metric("Highest Score", f"{highest_score_round['score']:.0f}")
            st.metric("in Round", f"{highest_score_round['game_round']}")
            st.metric("on Map", f"{highest_score_round['map']}")
            st.metric("with Deaths", f"{highest_score_round['deaths_total']:.0f}")

        with col2:
            st.subheader("Worst Performance")
            st.metric("Most Deaths", f"{most_deaths_round['deaths_total']:.0f}")
            st.metric("in Round", f"{most_deaths_round['game_round']}")
            st.metric("on Map", f"{most_deaths_round['map']}")
            st.metric("with Score", f"{most_deaths_round['score']:.0f}")

        with col3:
            st.subheader("Overall Statistics")
            st.metric("Best Map", best_map)
            st.metric("Total Score on Best Map", f"{best_map_score:.0f}")
            st.metric("Most Killed By", top_killer)
            st.metric("Times Killed", f"{top_killer_count}")

            
    st.sidebar.subheader("View")
    view_option = st.sidebar.radio(
        "Select View",
        ("Combined Graph", "Split Graphs"),
        label_visibility="collapsed"
    )
    
    st.sidebar.subheader("Graph Background")
    background_option = st.sidebar.radio(
        "Show map detail",
        ("Hide", "Unhide"),
        label_visibility="collapsed"
)

    # Function to add background colors based on map type
    def add_map_backgrounds(fig, data):
        if background_option == "Unhide":
            for i, row in data.iterrows():
                if row['map'] == 'aggressor':
                    color = 'rgba(255, 182, 193, 0.3)'  # Light pink
                elif row['map'] == 'wrackdm17':
                    color = 'rgba(173, 216, 230, 0.3)'  # Light blue
                else:  # kaos2
                    color = 'rgba(144, 238, 144, 0.3)'  # Light green
                
                fig.add_vrect(
                    x0=row['game_round'] - 0.5,
                    x1=row['game_round'] + 0.5,
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                )
        return fig

    # Create two columns for the charts
    col1, col2 = st.columns(2)

    player_pts = sb[sb['player_ip'] == option_player]    
    player_pts = player_pts.drop(columns=['player_id', 'player_ip'])

    player_deaths = df[df['victim_ip'] == option_player].groupby('game_round')['deaths_total'].sum().reset_index()
    player_deaths = pd.merge(player_deaths, player_pts[['game_round', 'map']], on='game_round', how='left')

    if view_option == "Combined Graph":
        # Combined view code
        if not player_pts.empty and not player_deaths.empty:
            # Merge the two dataframes
            combined_data = pd.merge(player_pts, player_deaths, on=['game_round', 'map'], how='outer').fillna(0)
            
            # Create a grouped bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=combined_data['game_round'],
                y=combined_data['score'],
                name='Score',
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Bar(
                x=combined_data['game_round'],
                y=combined_data['deaths_total'],
                name='Deaths',
                marker_color='#ff7f0e'
            ))
            
            # Add background colors for different map types
            fig = add_map_backgrounds(fig, combined_data)
            
            fig.update_layout(
                barmode='group',
                title='Score and Deaths Per Round',
                xaxis_title='Game Round',
                yaxis_title='Count',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Separate views code
        col1, col2 = st.columns(2)
        
        with col1:
            if not player_pts.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=player_pts['game_round'],
                    y=player_pts['score'],
                    name='Score',
                    marker_color='#1f77b4'
                ))
                
                # Add background colors for different map types
                fig = add_map_backgrounds(fig, player_pts)
                
                fig.update_layout(
                    title='Points Earned Per Round',
                    xaxis_title='Game Round',
                    yaxis_title='Score',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                player_pts_sorted = player_pts.sort_values(by='score', ascending=False)
                st.subheader("Most Score in a Round")
                st.write(player_pts_sorted)
            
        with col2:
            if not player_deaths.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=player_deaths['game_round'],
                    y=player_deaths['deaths_total'],
                    name='Deaths',
                    marker_color='#ff7f0e'
                ))
                
                # Add background colors for different map types
                fig = add_map_backgrounds(fig, player_deaths)
                
                fig.update_layout(
                    title='Number of Deaths Per Round',
                    xaxis_title='Game Round',
                    yaxis_title='Death Count',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                player_deaths_sorted = player_deaths.sort_values(by='deaths_total', ascending=False)
                st.subheader("Most Deaths in a Round")
                st.write(player_deaths_sorted)
