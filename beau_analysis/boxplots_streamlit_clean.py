import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import pytz
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER, ACTIVITY_FOLDER

# Set the page to wide mode
st.set_page_config(layout="wide")

# Load the datasets
@st.cache_data
def load_data():
    try:
        player_performance = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/player_performance.csv')
        return player_performance
    except Exception as e:
        st.error(f"Error loading the player performance data: {str(e)}")
        return None

@st.cache_data
def load_mouse_data(ip_address):
    try:
        filename = f'{ACTIVITY_FOLDER}/{ip_address}_activity_data.csv'
        return pd.read_csv(filename, low_memory=False)
    except Exception as e:
        st.error(f"Error loading the mouse movement data for IP {ip_address}: {str(e)}")
        return None

player_performance = load_data()

if player_performance is not None:
    # Define timezones
    utc = pytz.UTC
    aest = pytz.timezone('Australia/Sydney')

    # Function to convert timestamp to AEST
    def to_aest(time_obj):
        if isinstance(time_obj, str):
            time_obj = pd.to_datetime(time_obj)
        if time_obj.tzinfo is None:
            time_obj = time_obj.tz_localize(utc)
        return time_obj.astimezone(aest)

    # Convert timestamps for player performance data
    player_performance['timestamp'] = pd.to_datetime(player_performance['timestamp']).apply(to_aest)

    # Function to calculate input frequencies between events
    def calculate_input_frequencies(player_data, mouse_keyboard_data, event_type, latency, time_range):
        events = player_data[player_data['latency'] == latency]
        events = events.sort_values('timestamp')
        
        input_frequencies = {col: [] for col in ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']}
        
        for i in range(len(events)):
            event_time = events.iloc[i]['timestamp']
            start_time = event_time + timedelta(seconds=time_range[0])
            end_time = event_time + timedelta(seconds=time_range[1])
            
            start_data = mouse_keyboard_data[mouse_keyboard_data['timestamp'] <= start_time].iloc[-1] if not mouse_keyboard_data[mouse_keyboard_data['timestamp'] <= start_time].empty else pd.Series({col: 0 for col in input_frequencies.keys()})
            end_data = mouse_keyboard_data[mouse_keyboard_data['timestamp'] <= end_time].iloc[-1] if not mouse_keyboard_data[mouse_keyboard_data['timestamp'] <= end_time].empty else pd.Series({col: 0 for col in input_frequencies.keys()})
            
            for col in input_frequencies.keys():
                input_frequencies[col].append(end_data[col] - start_data[col])
        
        return input_frequencies

    # Streamlit app
    st.title('Player Performance Box Plot Visualization')

    # Sidebar for selections
    st.sidebar.header('Filters')

    # Player selection
    all_players = pd.concat([player_performance['killer_ip'], player_performance['victim_ip']]).unique()
    selected_players = st.sidebar.multiselect('Select Players', all_players)

    # Affected players selection (persistent)
    if 'affected_players' not in st.session_state:
        st.session_state.affected_players = []
    
    affected_players = st.sidebar.multiselect('Select Affected Players', selected_players, default=st.session_state.affected_players)
    st.session_state.affected_players = affected_players

    # Latency selection (including 0ms by default)
    all_latencies = sorted(player_performance['latency'].unique())
    default_latencies = [0] if 0 in all_latencies else []
    selected_latencies = st.sidebar.multiselect('Select Latencies', all_latencies, default=default_latencies)

    # Map selection
    all_maps = player_performance['map'].unique()
    selected_maps = st.sidebar.multiselect('Select Maps', all_maps)

    # Event type selection
    event_type = st.sidebar.radio("Select Event Type", ('Deaths', 'Kills'))

    # Time range slider
    time_range = st.sidebar.slider(
        "Select Time Range Around Event (in seconds)",
        min_value=-30.0,
        max_value=30.0,
        value=(-5.0, 2.0),
        step=0.1
    )

    # Data transformation option
    data_transformation = st.sidebar.radio(
        "Select Data Transformation",
        ('Raw Data', 'Relative Values (Normalized to 0ms latency)', 'Z-Score Normalization')
    )

    # Input columns
    input_columns = ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']

    if selected_players and selected_latencies and selected_maps:
        all_player_data = []

        for player in selected_players:
            player_data = player_performance[(player_performance['killer_ip' if event_type == 'Kills' else 'victim_ip'] == player) & 
                                             (player_performance['map'].isin(selected_maps))]
            mouse_keyboard_data = load_mouse_data(player.split('_')[1])

            if mouse_keyboard_data is None:
                st.warning(f"No mouse movement data available for {player}")
                continue

            mouse_keyboard_data['timestamp'] = pd.to_datetime(mouse_keyboard_data['timestamp'], unit='s').apply(to_aest)

            player_latency_data = []

            for latency in selected_latencies:
                input_frequencies = calculate_input_frequencies(player_data, mouse_keyboard_data, event_type, latency, time_range)
                
                for input_col in input_columns:
                    player_latency_data.extend([(freq, latency, player, input_col) for freq in input_frequencies[input_col]])

            all_player_data.extend(player_latency_data)

        # Create DataFrame with all player data
        all_player_df = pd.DataFrame(all_player_data, columns=['frequency', 'latency', 'player', 'input'])

        # Data transformation
        if data_transformation == 'Relative Values (Normalized to 0ms latency)':
            baseline_data = all_player_df[all_player_df['latency'] == 0].groupby(['player', 'input'])['frequency'].mean().reset_index()
            baseline_data = baseline_data.rename(columns={'frequency': 'baseline_frequency'})
            all_player_df = pd.merge(all_player_df, baseline_data, on=['player', 'input'])
            all_player_df['transformed_frequency'] = all_player_df['frequency'] / all_player_df['baseline_frequency']
            y_axis_title = 'Relative Frequency'
        elif data_transformation == 'Z-Score Normalization':
            baseline_data = all_player_df[all_player_df['latency'] == 0].groupby(['player', 'input']).agg({'frequency': ['mean', 'std']}).reset_index()
            baseline_data.columns = ['player', 'input', 'baseline_mean', 'baseline_std']
            all_player_df = pd.merge(all_player_df, baseline_data, on=['player', 'input'])
            all_player_df['transformed_frequency'] = (all_player_df['frequency'] - all_player_df['baseline_mean']) / all_player_df['baseline_std']
            y_axis_title = 'Z-Score'
        else:
            all_player_df['transformed_frequency'] = all_player_df['frequency']
            y_axis_title = 'Frequency'

        # Calculate summary tables
        def calculate_summary_table(player_group):
            summary_data = []
            baseline_latency = 0
            
            for latency in [0, 50, 100, 150, 200]:
                if latency in selected_latencies:
                    row_data = {'Latency': latency}
                    for input_col in input_columns:
                        baseline_avg = player_group[(player_group['latency'] == baseline_latency) & (player_group['input'] == input_col)]['frequency'].mean()
                        latency_avg = player_group[(player_group['latency'] == latency) & (player_group['input'] == input_col)]['frequency'].mean()
                        pct_change = ((latency_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg != 0 else 0
                        color = 'green' if pct_change >= 0 else 'red'
                        row_data[input_col] = f'<span style="color: {color}">{pct_change:.2f}%</span>'
                    summary_data.append(row_data)
            
            return pd.DataFrame(summary_data)

        non_affected_summary = calculate_summary_table(all_player_df[~all_player_df['player'].isin(affected_players)])
        affected_summary = calculate_summary_table(all_player_df[all_player_df['player'].isin(affected_players)])

        # Display summary tables
        st.subheader("Summary Tables: Percentage Change in Inputs Relative to 0ms Latency")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Non-Affected Players")
            st.markdown(non_affected_summary.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        with col2:
            st.write("Affected Players")
            st.markdown(affected_summary.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Display merged box plots for affected and non-affected players
        st.subheader(f'Performance Comparison (Maps: {", ".join(selected_maps)})')
        st.write("Blue boxes: Non-Affected Players | Red boxes: Affected Players")
        for input_col in input_columns:
            fig = go.Figure()

            for latency in selected_latencies:
                non_affected_data = all_player_df[(all_player_df['latency'] == latency) & 
                                                  (all_player_df['input'] == input_col) & 
                                                  (~all_player_df['player'].isin(affected_players))]['transformed_frequency']
                
                affected_data = all_player_df[(all_player_df['latency'] == latency) & 
                                              (all_player_df['input'] == input_col) & 
                                              (all_player_df['player'].isin(affected_players))]['transformed_frequency']
                
                fig.add_trace(go.Box(y=non_affected_data, 
                                     name=f'Non-Affected ({latency}ms)', 
                                     boxmean=True,
                                     fillcolor='rgba(0, 0, 255, 0.3)',
                                     line=dict(color='rgba(0, 0, 255, 1)'),
                                     quartilemethod="linear",
                                     width=0.8))  # Increased width for wider boxes
                
                fig.add_trace(go.Box(y=affected_data, 
                                     name=f'Affected ({latency}ms)', 
                                     boxmean=True,
                                     fillcolor='rgba(255, 0, 0, 0.3)',
                                     line=dict(color='rgba(255, 0, 0, 1)'),
                                     quartilemethod="linear",
                                     width=0.8))  # Increased width for wider boxes

            fig.update_layout(title=f'{input_col.capitalize()} {y_axis_title} Between {event_type} (Time Range: {time_range[0]}s to {time_range[1]}s)',
                              xaxis_title='Player Group and Latency',
                              yaxis_title=y_axis_title,
                              height=600,
                              boxmode='group',
                              boxgroupgap=0.2,  # Reduced gap between groups for wider boxes
                              boxgap=0.1)       # Reduced gap within groups for wider boxes
            
            # Adjust y-axis to remove scaling
            fig.update_layout(yaxis=dict(tickformat='.2f'))

            st.plotly_chart(fig, use_container_width=True)

            # Percentage change graph
            pct_change_fig = go.Figure()
            baseline_latency = min(selected_latencies)
            
            for group in ['Non-Affected', 'Affected']:
                group_data = all_player_df[~all_player_df['player'].isin(affected_players) if group == 'Non-Affected' else all_player_df['player'].isin(affected_players)]
                baseline_avg = group_data[(group_data['latency'] == baseline_latency) & (group_data['input'] == input_col)]['frequency'].mean()
                
                pct_changes = []
                for latency in selected_latencies:
                    if latency != baseline_latency:
                        latency_avg = group_data[(group_data['latency'] == latency) & (group_data['input'] == input_col)]['frequency'].mean()
                        pct_change = ((latency_avg - baseline_avg) / baseline_avg) * 100
                        pct_changes.append(pct_change)
                
                pct_change_fig.add_trace(go.Bar(
                    x=[f'{latency}ms' for latency in selected_latencies if latency != baseline_latency],
                    y=pct_changes,
                    name=group,
                    marker_color='blue' if group == 'Non-Affected' else 'red'
                ))
            
            pct_change_fig.update_layout(
                title=f'Percentage Change in {input_col.capitalize()} Frequency from {baseline_latency}ms Latency (Time Range: {time_range[0]}s to {time_range[1]}s)',
                xaxis_title='Latency',
                yaxis_title='Percent Change',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(pct_change_fig, use_container_width=True)

    else:
        st.warning('Please select at least one player, one latency value, and one map.')

else:
    st.error("Cannot proceed with analysis due to data loading error.")