import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import pytz
from scipy import stats
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
        return pd.read_csv(filename)
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

    # Affected players selection
    affected_players = st.sidebar.multiselect('Select Affected Players', selected_players)

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
            
            # Adjust y-axis to remove scaling
            fig.update_layout(yaxis=dict(tickformat='.2f'))

            st.plotly_chart(fig, use_container_width=True)

            # Average graphs
            st.subheader(f'Average Performance for {input_col.capitalize()}')
            
            # Absolute values average graph
            avg_fig = go.Figure()
            for latency in selected_latencies:
                avg_data = all_player_df[(all_player_df['latency'] == latency) & 
                                         (all_player_df['input'] == input_col)]['frequency']
                avg_fig.add_trace(go.Box(y=avg_data, name=f'{latency}ms', boxmean=True))
            
            avg_fig.update_layout(title=f'Average {input_col.capitalize()} Frequency',
                                  xaxis_title='Latency',
                                  yaxis_title='Frequency',
                                  height=400)
            st.plotly_chart(avg_fig, use_container_width=True)

            # Percent change from 0ms latency graph
            pct_change_fig = go.Figure()
            baseline_avg = all_player_df[(all_player_df['latency'] == 0) & 
                                         (all_player_df['input'] == input_col)]['frequency'].mean()
            
            for latency in selected_latencies:
                if latency != 0:
                    latency_avg = all_player_df[(all_player_df['latency'] == latency) & 
                                                (all_player_df['input'] == input_col)]['frequency'].mean()
                    pct_change = ((latency_avg - baseline_avg) / baseline_avg) * 100
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change], name=f'{latency}ms'))
            
            pct_change_fig.update_layout(title=f'Average Percent Change in {input_col.capitalize()} Frequency from 0ms Latency',
                                         xaxis_title='Latency',
                                         yaxis_title='Percent Change',
                                         height=400)
            st.plotly_chart(pct_change_fig, use_container_width=True)

            # Statistical analysis
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

    else:
        st.warning('Please select at least one player, one latency value, and one map.')

else:
    st.error("Cannot proceed with analysis due to data loading error.")