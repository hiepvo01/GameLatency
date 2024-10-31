import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER, ACTIVITY_FOLDER

#st.set_page_config(page_title="Latency Analysis", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    try:
        return pd.read_csv(f'{PROCESSED_DATA_FOLDER}/round_summary_adjusted.csv')
    except Exception as e:
        st.error(f"Error loading the data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Filter out the <world> player data
    df = df[df['player_ip'] != '<world>']

    def generate_statistics(df):
        grouped = df.groupby('latency')
        result_dfs = {}
        
        for latency, group in grouped:
            unique_player_ips = group['player_ip'].unique()
            latency_df = pd.DataFrame(index=unique_player_ips)
            
            group = group[group['latency'] == latency]
            
            for game_round in group['game_round'].unique():
                round_scores = group[group['game_round'] == game_round].set_index('player_ip')['score'].rename(f'Round_{game_round}')
                latency_df = latency_df.join(round_scores, how='left')
            
            latency_df['Mean'] = latency_df.mean(axis=1)
            latency_df['StdDev'] = latency_df.std(axis=1)
            
            if 0 in result_dfs:
                latency_df['mean_difference'] = latency_df['Mean'] - result_dfs[0]['Mean']
            else:
                latency_df['mean_difference'] = 0
            
            result_dfs[latency] = latency_df
        
        return result_dfs

    result_dfs = generate_statistics(df)

    st.title('Players\' Mean Scores vs Latency Statistical Analysis')

    latency_values = list(result_dfs.keys())
    selected_latency = st.selectbox('Select Latency Value (ms)', latency_values, index=0)

    st.subheader(f'Statistics for Latency {selected_latency}')
    st.dataframe(result_dfs[selected_latency])

    # Create an interactive line plot
    fig = go.Figure()

    for player_ip in df['player_ip'].unique():
        means = [result_dfs[latency].loc[player_ip, 'Mean'] for latency in result_dfs.keys() if player_ip in result_dfs[latency].index]
        fig.add_trace(go.Scatter(
            x=list(result_dfs.keys()), 
            y=means, 
            mode='lines+markers', 
            name=f'Player {player_ip}',
            line=dict(width=6)  # Increased line thickness
        ))

    fig.update_layout(
        title={
            'text': 'Players\' Mean Scores vs Latency',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 48}  # Increased title font size
        },
        xaxis_title='Latency',
        yaxis_title='Mean Score',
        xaxis=dict(
            title_font=dict(size=36),  # Increased x-axis label font size
            tickfont=dict(size=36)  # Increased x-axis tick label font size
        ),
        yaxis=dict(
            title_font=dict(size=36),  # Increased y-axis label font size
            tickfont=dict(size=36)  # Increased y-axis tick label font size
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="center",
            x=0.5,
            font=dict(size=24),  # Slightly reduced legend font size
            bgcolor='rgba(255, 255, 255, 0.7)',  # Semi-transparent background
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1
        ),
        margin=dict(t=180, l=100, r=100, b=100),  # Adjusted margins
        height=1080,  # Set height to match the specified dimensions
        width=2420,   # Set width to match the specified dimensions
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'width': 2420,
            'height': 1080,
            'scale': 8
        }
    })
else:
    st.error("Cannot proceed with analysis due to data loading error.")

# Display available columns
# st.subheader("Available Data Columns")
# if df is not None:
#     st.write(f"Columns in the dataset: {', '.join(df.columns)}")
# else:
#     st.write("Data not available.")