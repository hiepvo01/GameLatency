import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import pytz
from scipy import stats
import gc
import time
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER, ACTIVITY_FOLDER

# Set the page to wide mode
st.set_page_config(layout="wide")

# Load the datasets
@st.cache_data
def load_data(sample_size=None):
    try:
        # Read the CSV in chunks
        chunk_size = 10000  # Adjust this based on your memory constraints
        chunks = []
        for chunk in pd.read_csv(f'{PROCESSED_DATA_FOLDER}/player_performance.csv', chunksize=chunk_size):
            chunks.append(chunk)
            if sample_size and len(chunks) * chunk_size >= sample_size:
                break
        player_performance = pd.concat(chunks, ignore_index=True)
        if sample_size:
            player_performance = player_performance.sample(n=min(sample_size, len(player_performance)))
        return player_performance
    except Exception as e:
        st.error(f"Error loading the player performance data: {str(e)}")
        return None

@st.cache_data
def load_mouse_data(ip_address):
    try:
        filename = f'{ACTIVITY_FOLDER}/{ip_address}_activity_data.csv'
        # Read only necessary columns
        return pd.read_csv(filename, usecols=['timestamp', 'mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D'])
    except Exception as e:
        st.error(f"Error loading the mouse movement data for IP {ip_address}: {str(e)}")
        return None

# Function to convert timestamp to AEST
@st.cache_data
def to_aest(time_obj):
    utc = pytz.UTC
    aest = pytz.timezone('Australia/Sydney')
    if isinstance(time_obj, str):
        time_obj = pd.to_datetime(time_obj)
    if time_obj.tzinfo is None:
        time_obj = time_obj.tz_localize(utc)
    return time_obj.astimezone(aest)

# Function to calculate input frequencies between events
@st.cache_data
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

# Sample size selection
sample_size = st.sidebar.number_input("Sample size (0 for all data)", min_value=0, value=1000, step=100)
player_performance = load_data(sample_size if sample_size > 0 else None)

if player_performance is not None:
    # Convert timestamps for player performance data
    player_performance['timestamp'] = pd.to_datetime(player_performance['timestamp']).apply(to_aest)

    # Player selection
    all_players = pd.concat([player_performance['killer_ip'], player_performance['victim_ip']]).unique()
    selected_players = st.sidebar.multiselect('Select Players', all_players)

    # Affected players selection (persistent)
    if 'affected_players' not in st.session_state:
        st.session_state.affected_players = []
    
    affected_players = st.sidebar.multiselect('Select Affected Players', selected_players, default=st.session_state.affected_players)
    st.session_state.affected_players = affected_players

    # Latency selection
    all_latencies = sorted(player_performance['latency'].unique())
    selected_latencies = st.sidebar.multiselect('Select Latencies', all_latencies)

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

    # Relativisation option
    use_relativisation = st.sidebar.checkbox("Use Relative Values (Normalized to 0ms latency)", value=False)

    # Input columns
    input_columns = ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']

    if selected_players and selected_latencies and selected_maps:
        all_player_data = []
        
        # Calculate total iterations for progress bar
        total_iterations = len(selected_players) * len(selected_latencies) * len(input_columns)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        iterations_completed = 0

        for player in selected_players:
            player_data = player_performance[(player_performance['killer_ip' if event_type == 'Kills' else 'victim_ip'] == player) & 
                                            (player_performance['map'].isin(selected_maps))]
            mouse_keyboard_data = load_mouse_data(player.split('_')[1])

            if mouse_keyboard_data is None:
                st.warning(f"No mouse movement data available for {player}")
                continue

            mouse_keyboard_data['timestamp'] = pd.to_datetime(mouse_keyboard_data['timestamp'], unit='s').apply(to_aest)

            for latency in selected_latencies:
                input_frequencies = calculate_input_frequencies(player_data, mouse_keyboard_data, event_type, latency, time_range)
                
                for input_col in input_columns:
                    all_player_data.extend([(freq, latency, player, input_col) for freq in input_frequencies[input_col]])
                    
                    # Update progress
                    iterations_completed += 1
                    progress = iterations_completed / total_iterations
                    progress_bar.progress(progress)
                    
                    # Calculate ETA
                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / progress
                    eta = estimated_total_time - elapsed_time
                    
                    status_text.text(f"Processing... ETA: {timedelta(seconds=int(eta))}")

        # Convert all_player_data to DataFrame
        all_player_df = pd.DataFrame(all_player_data, columns=['frequency', 'latency', 'player', 'input'])
        
        # Clear progress bar and status text
        progress_bar.empty()
        status_text.empty()
        
        st.success("Data processing completed!")

        # Relativisation
        if use_relativisation:
            baseline_data = all_player_df[all_player_df['latency'] == 0].groupby(['player', 'input'])['frequency'].mean().reset_index()
            baseline_data = baseline_data.rename(columns={'frequency': 'baseline_frequency'})
            all_player_df = pd.merge(all_player_df, baseline_data, on=['player', 'input'])
            all_player_df['relative_frequency'] = all_player_df['frequency'] / all_player_df['baseline_frequency']
            plot_column = 'relative_frequency'
            y_axis_title = 'Relative Frequency'
        else:
            plot_column = 'frequency'
            y_axis_title = 'Frequency'

        # Display merged box plots for all players
        st.subheader(f'Performance Across All Selected Players (Maps: {", ".join(selected_maps)})')
        st.write("Red boxes: Affected Players | Blue boxes: Non-Affected Players")
        
        for input_col in input_columns:
            fig = go.Figure()

            for latency in selected_latencies:
                for player in selected_players:
                    player_data = all_player_df[(all_player_df['latency'] == latency) & 
                                                (all_player_df['input'] == input_col) & 
                                                (all_player_df['player'] == player)][plot_column]
                    
                    if player in affected_players:
                        box_color = 'rgba(255, 0, 0, 0.3)'  # Light red for affected players
                        line_color = 'rgba(255, 0, 0, 1)'   # Solid red border
                    else:
                        box_color = 'rgba(0, 0, 255, 0.3)'  # Light blue for non-affected players
                        line_color = 'rgba(0, 0, 255, 1)'   # Solid blue border
                    
                    fig.add_trace(go.Box(y=player_data, 
                                         name=f'{player} ({latency}ms)', 
                                         boxmean=True,
                                         fillcolor=box_color,
                                         line=dict(color=line_color)))

            fig.update_layout(title=f'{input_col.capitalize()} {y_axis_title} Between {event_type}',
                              xaxis_title='Player and Latency',
                              yaxis_title=y_axis_title,
                              height=600,
                              boxmode='group')
            
            fig.update_layout(yaxis=dict(tickformat='.2f'))
            st.plotly_chart(fig, use_container_width=True)

            # Clear the figure to free up memory
            fig = go.Figure()
            gc.collect()

        # Average graphs
        st.subheader(f'Average Performance')
        st.write("Red: Affected Players | Blue: Non-Affected Players | Purple: All Selected Players")
        
        for input_col in input_columns:
            avg_fig = go.Figure()
            for latency in selected_latencies:
                affected_data = all_player_df[(all_player_df['latency'] == latency) & 
                                              (all_player_df['input'] == input_col) & 
                                              (all_player_df['player'].isin(affected_players))][plot_column]
                non_affected_data = all_player_df[(all_player_df['latency'] == latency) & 
                                                  (all_player_df['input'] == input_col) & 
                                                  (~all_player_df['player'].isin(affected_players))][plot_column]
                all_selected_data = all_player_df[(all_player_df['latency'] == latency) & 
                                                  (all_player_df['input'] == input_col)][plot_column]
                
                avg_fig.add_trace(go.Box(y=affected_data, name=f'Affected ({latency}ms)', boxmean=True, fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')))
                avg_fig.add_trace(go.Box(y=non_affected_data, name=f'Non-Affected ({latency}ms)', boxmean=True, fillcolor='rgba(0,0,255,0.3)', line=dict(color='blue')))
                avg_fig.add_trace(go.Box(y=all_selected_data, name=f'All Selected ({latency}ms)', boxmean=True, fillcolor='rgba(128,0,128,0.3)', line=dict(color='purple')))
            
            avg_fig.update_layout(title=f'Average {input_col.capitalize()} {y_axis_title}',
                                  xaxis_title='Latency',
                                  yaxis_title=y_axis_title,
                                  height=400,
                                  boxmode='group')
            st.plotly_chart(avg_fig, use_container_width=True)

            # Clear the figure to free up memory
            avg_fig = go.Figure()
            gc.collect()

        # Percent change from 0ms latency graph
        for input_col in input_columns:
            pct_change_fig = go.Figure()
            baseline_affected = all_player_df[(all_player_df['latency'] == 0) & 
                                              (all_player_df['input'] == input_col) & 
                                              (all_player_df['player'].isin(affected_players))][plot_column].mean()
            baseline_non_affected = all_player_df[(all_player_df['latency'] == 0) & 
                                                  (all_player_df['input'] == input_col) & 
                                                  (~all_player_df['player'].isin(affected_players))][plot_column].mean()
            baseline_all_selected = all_player_df[(all_player_df['latency'] == 0) & 
                                                  (all_player_df['input'] == input_col)][plot_column].mean()
            
            for latency in selected_latencies:
                if latency != 0:
                    affected_avg = all_player_df[(all_player_df['latency'] == latency) & 
                                                 (all_player_df['input'] == input_col) & 
                                                 (all_player_df['player'].isin(affected_players))][plot_column].mean()
                    non_affected_avg = all_player_df[(all_player_df['latency'] == latency) & 
                                                     (all_player_df['input'] == input_col) & 
                                                     (~all_player_df['player'].isin(affected_players))][plot_column].mean()
                    all_selected_avg = all_player_df[(all_player_df['latency'] == latency) & 
                                                     (all_player_df['input'] == input_col)][plot_column].mean()
                    
                    pct_change_affected = ((affected_avg - baseline_affected) / baseline_affected) * 100 if baseline_affected else 0
                    pct_change_non_affected = ((non_affected_avg - baseline_non_affected) / baseline_non_affected) * 100 if baseline_non_affected else 0
                    pct_change_all_selected = ((all_selected_avg - baseline_all_selected) / baseline_all_selected) * 100 if baseline_all_selected else 0
                    
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change_affected], name=f'Affected', marker_color='red'))
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change_non_affected], name=f'Non-Affected', marker_color='blue'))
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change_all_selected], name=f'All Selected', marker_color='purple'))
            
            pct_change_fig.update_layout(title=f'Average Percent Change in {input_col.capitalize()} {y_axis_title} from 0ms Latency',
                                         xaxis_title='Latency',
                                         yaxis_title='Percent Change',
                                         height=400,
                                         barmode='group')
            st.plotly_chart(pct_change_fig, use_container_width=True)

            # Clear the figure to free up memory
            pct_change_fig = go.Figure()
            gc.collect()

        # Statistical analysis
        st.subheader("Statistical Insights")
        for input_col in input_columns:
            st.write(f"### Statistical Insights for {input_col.capitalize()}")
            baseline_latency = min(selected_latencies)
            
            affected_data = all_player_df[(all_player_df['latency'] == baseline_latency) & 
                                          (all_player_df['input'] == input_col) & 
                                          (all_player_df['player'].isin(affected_players))][plot_column]
            
            non_affected_data = all_player_df[(all_player_df['latency'] == baseline_latency) & 
                                              (all_player_df['input'] == input_col) & 
                                              (~all_player_df['player'].isin(affected_players))][plot_column]
            
            for latency in selected_latencies:
                if latency != baseline_latency:
                    affected_latency_data = all_player_df[(all_player_df['latency'] == latency) & 
                                                          (all_player_df['input'] == input_col) & 
                                                          (all_player_df['player'].isin(affected_players))][plot_column]
                    
                    non_affected_latency_data = all_player_df[(all_player_df['latency'] == latency) & 
                                                              (all_player_df['input'] == input_col) & 
                                                              (~all_player_df['player'].isin(affected_players))][plot_column]
                    
                    # Affected players analysis
                    if not affected_data.empty and not affected_latency_data.empty:
                        percent_change = ((affected_latency_data.mean() - affected_data.mean()) / affected_data.mean()) * 100
                        t_stat, p_value = stats.ttest_ind(affected_data, affected_latency_data)
                        
                        st.write(f"Affected Players at {latency}ms latency compared to {baseline_latency}ms:")
                        st.write(f"- {input_col.capitalize()} {y_axis_title.lower()} {'increased' if percent_change > 0 else 'decreased'} by {abs(percent_change):.2f}%")
                        st.write(f"- T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.write("  This change is statistically significant.")
                        else:
                            st.write("  This change is not statistically significant.")
                    
                    # Non-affected players analysis
                    if not non_affected_data.empty and not non_affected_latency_data.empty:
                        percent_change = ((non_affected_latency_data.mean() - non_affected_data.mean()) / non_affected_data.mean()) * 100
                        t_stat, p_value = stats.ttest_ind(non_affected_data, non_affected_latency_data)
                        
                        st.write(f"Non-Affected Players at {latency}ms latency compared to {baseline_latency}ms:")
                        st.write(f"- {input_col.capitalize()} {y_axis_title.lower()} {'increased' if percent_change > 0 else 'decreased'} by {abs(percent_change):.2f}%")
                        st.write(f"- T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.write("  This change is statistically significant.")
                        else:
                            st.write("  This change is not statistically significant.")
                
                st.write("---")

        # Clear unnecessary data from memory
        del player_performance
        del all_player_df
        gc.collect()

    else:
        st.warning('Please select at least one player, one latency value, and one map.')

else:
    st.error("Cannot proceed with analysis due to data loading error.")