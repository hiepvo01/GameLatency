import streamlit as st
import polars as pl
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
        player_performance = pl.read_csv(f'{PROCESSED_DATA_FOLDER}/player_performance.csv')
        return player_performance
    except Exception as e:
        st.error(f"Error loading the player performance data: {str(e)}")
        return None

@st.cache_data
def load_mouse_data(ip_address):
    try:
        filename = f'{ACTIVITY_FOLDER}/{ip_address}_activity_data.csv'
        return pl.read_csv(filename)
    except Exception as e:
        st.error(f"Error loading the mouse movement data for IP {ip_address}: {str(e)}")
        return None

player_performance = load_data()

if player_performance is not None:
    # Define timezones
    utc = pytz.UTC
    aest = pytz.timezone('Australia/Sydney')

    # Function to convert timestamp to AEST
    def to_aest():
        return pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f").dt.replace_time_zone("UTC").dt.convert_time_zone("Australia/Sydney")

    # Convert timestamps for player performance data
    player_performance = player_performance.with_columns(
        to_aest().alias('timestamp')
    )

    # Function to calculate input frequencies between events
    def calculate_input_frequencies(player_data, mouse_keyboard_data, event_type, latency, time_range):
        events = player_data.filter(pl.col('latency') == latency).sort('timestamp')
        
        input_frequencies = {col: [] for col in ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']}
        
        for i in range(len(events)):
            event_time = events['timestamp'][i]
            start_time = event_time + timedelta(seconds=time_range[0])
            end_time = event_time + timedelta(seconds=time_range[1])
            
            start_data = mouse_keyboard_data.filter(pl.col('timestamp') <= start_time).tail(1)
            end_data = mouse_keyboard_data.filter(pl.col('timestamp') <= end_time).tail(1)
            
            if start_data.height == 0:
                start_data = pl.DataFrame({col: [0] for col in input_frequencies.keys()})
            if end_data.height == 0:
                end_data = pl.DataFrame({col: [0] for col in input_frequencies.keys()})
            
            for col in input_frequencies.keys():
                input_frequencies[col].append(end_data[col][0] - start_data[col][0])
        
        return input_frequencies

    # Streamlit app
    st.title('Player Performance Box Plot Visualization')

    # Sidebar for selections
    st.sidebar.header('Filters')

    # Player selection
    all_players = pl.concat([player_performance['killer_ip'], player_performance['victim_ip']]).unique().to_list()
    if 'selected_players' not in st.session_state:
        st.session_state.selected_players = []
    selected_players = st.sidebar.multiselect('Select Players', all_players, key='selected_players')

    # Affected players selection
    if 'affected_players' not in st.session_state:
        st.session_state.affected_players = []
    affected_players = st.sidebar.multiselect('Select Affected Players', selected_players, key='affected_players')

    # Latency selection
    all_latencies = player_performance['latency'].drop_nulls().unique().sort().to_list()
    if 'selected_latencies' not in st.session_state:
        st.session_state.selected_latencies = []
    selected_latencies = st.sidebar.multiselect('Select Latencies', all_latencies, key='selected_latencies')

    # Map selection
    all_maps = player_performance['map'].unique().to_list()
    if 'selected_maps' not in st.session_state:
        st.session_state.selected_maps = []
    selected_maps = st.sidebar.multiselect('Select Maps', all_maps, key='selected_maps')

    # Event type selection
    if 'event_type' not in st.session_state:
        st.session_state.event_type = 'Deaths'
    event_type = st.sidebar.radio("Select Event Type", ('Deaths', 'Kills'), key='event_type')

    # Time range slider
    if 'time_range' not in st.session_state:
        st.session_state.time_range = (-5.0, 2.0)
    time_range = st.sidebar.slider(
        "Select Time Range Around Event (in seconds)",
        min_value=-30.0,
        max_value=30.0,
        value=st.session_state.time_range,
        step=0.1,
        key='time_range'
    )

    # Relativisation option
    if 'use_relativisation' not in st.session_state:
        st.session_state.use_relativisation = False
    use_relativisation = st.sidebar.checkbox("Use Relative Values (Normalized to 0ms latency)", value=st.session_state.use_relativisation, key='use_relativisation')

    # Input columns
    input_columns = ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']

    if selected_players and selected_latencies and selected_maps:
        all_player_data = []

        for player in selected_players:
            player_data = player_performance.filter(
                (pl.col('killer_ip' if event_type == 'Kills' else 'victim_ip') == player) &
                (pl.col('map').is_in(selected_maps)) &
                (pl.col('latency').is_in(selected_latencies))
            )
            
            if player_data.is_empty():
                st.warning(f"No data available for {player} with the selected criteria")
                continue

            mouse_keyboard_data = load_mouse_data(player.split('_')[1])

            if mouse_keyboard_data is None:
                st.warning(f"No mouse movement data available for {player}")
                continue

            mouse_keyboard_data = mouse_keyboard_data.with_columns(
                pl.col('timestamp').apply(lambda x: to_aest(datetime.fromtimestamp(x))).alias('timestamp')
            )

            player_latency_data = []

            for latency in selected_latencies:
                input_frequencies = calculate_input_frequencies(player_data, mouse_keyboard_data, event_type, latency, time_range)
                
                for input_col in input_columns:
                    player_latency_data.extend([(freq, latency, player, input_col) for freq in input_frequencies[input_col]])

            all_player_data.extend(player_latency_data)

        if not all_player_data:
            st.warning("No data available for the selected criteria")
        else:
            # Create DataFrame with all player data
            all_player_df = pl.DataFrame(all_player_data, schema=['frequency', 'latency', 'player', 'input'])

        # Relativisation
        if use_relativisation:
            baseline_data = all_player_df.filter(pl.col('latency') == 0).groupby(['player', 'input']).agg(
                pl.col('frequency').mean().alias('baseline_frequency')
            )
            all_player_df = all_player_df.join(baseline_data, on=['player', 'input'])
            all_player_df = all_player_df.with_columns(
                (pl.col('frequency') / pl.col('baseline_frequency')).alias('relative_frequency')
            )
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
                    player_data = all_player_df.filter(
                        (pl.col('latency') == latency) &
                        (pl.col('input') == input_col) &
                        (pl.col('player') == player)
                    )[plot_column].to_list()
                    
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
            st.write("Red: Affected Players | Blue: Non-Affected Players | Green: All Players")
            
            # Absolute values average graph
            avg_fig = go.Figure()
            for latency in selected_latencies:
                affected_data = all_player_df.filter(
                    (pl.col('latency') == latency) &
                    (pl.col('input') == input_col) &
                    (pl.col('player').is_in(affected_players))
                )['frequency'].to_list()
                non_affected_data = all_player_df.filter(
                    (pl.col('latency') == latency) &
                    (pl.col('input') == input_col) &
                    (~pl.col('player').is_in(affected_players))
                )['frequency'].to_list()
                all_data = all_player_df.filter(
                    (pl.col('latency') == latency) &
                    (pl.col('input') == input_col)
                )['frequency'].to_list()
                
                avg_fig.add_trace(go.Box(y=affected_data, name=f'Affected ({latency}ms)', boxmean=True, fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')))
                avg_fig.add_trace(go.Box(y=non_affected_data, name=f'Non-Affected ({latency}ms)', boxmean=True, fillcolor='rgba(0,0,255,0.3)', line=dict(color='blue')))
                avg_fig.add_trace(go.Box(y=all_data, name=f'All Players ({latency}ms)', boxmean=True, fillcolor='rgba(0,255,0,0.3)', line=dict(color='green')))
            
            avg_fig.update_layout(title=f'Average {input_col.capitalize()} Frequency',
                                  xaxis_title='Latency',
                                  yaxis_title='Frequency',
                                  height=400,
                                  boxmode='group')
            st.plotly_chart(avg_fig, use_container_width=True)

            # Percent change from 0ms latency graph
            pct_change_fig = go.Figure()
            baseline_affected = all_player_df.filter(
                (pl.col('latency') == 0) &
                (pl.col('input') == input_col) &
                (pl.col('player').is_in(affected_players))
            )['frequency'].mean()
            baseline_non_affected = all_player_df.filter(
                (pl.col('latency') == 0) &
                (pl.col('input') == input_col) &
                (~pl.col('player').is_in(affected_players))
            )['frequency'].mean()
            baseline_all = all_player_df.filter(
                (pl.col('latency') == 0) &
                (pl.col('input') == input_col)
            )['frequency'].mean()
            
            for latency in selected_latencies:
                if latency != 0:
                    affected_avg = all_player_df.filter(
                        (pl.col('latency') == latency) &
                        (pl.col('input') == input_col) &
                        (pl.col('player').is_in(affected_players))
                    )['frequency'].mean()
                    non_affected_avg = all_player_df.filter(
                        (pl.col('latency') == latency) &
                        (pl.col('input') == input_col) &
                        (~pl.col('player').is_in(affected_players))
                    )['frequency'].mean()
                    all_avg = all_player_df.filter(
                        (pl.col('latency') == latency) &
                        (pl.col('input') == input_col)
                    )['frequency'].mean()
                    
                    pct_change_affected = ((affected_avg - baseline_affected) / baseline_affected) * 100
                    pct_change_non_affected = ((non_affected_avg - baseline_non_affected) / baseline_non_affected) * 100
                    pct_change_all = ((all_avg - baseline_all) / baseline_all) * 100
                    
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change_affected], name=f'Affected', marker_color='red'))
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change_non_affected], name=f'Non-Affected', marker_color='blue'))
                    pct_change_fig.add_trace(go.Bar(x=[f'{latency}ms'], y=[pct_change_all], name=f'All Players', marker_color='green'))
            
            pct_change_fig.update_layout(title=f'Average Percent Change in {input_col.capitalize()} Frequency from 0ms Latency',
                                         xaxis_title='Latency',
                                         yaxis_title='Percent Change',
                                         height=400,
                                         barmode='group')
            st.plotly_chart(pct_change_fig, use_container_width=True)

            # Statistical analysis
            st.write(f"### Statistical Insights for {input_col.capitalize()}")
            baseline_latency = min(selected_latencies)
            
            affected_data = all_player_df.filter(
                (pl.col('latency') == baseline_latency) &
                (pl.col('input') == input_col) &
                (pl.col('player').is_in(affected_players))
            )[plot_column].to_numpy()
            
            non_affected_data = all_player_df.filter(
                (pl.col('latency') == baseline_latency) &
                (pl.col('input') == input_col) &
                (~pl.col('player').is_in(affected_players))
            )[plot_column].to_numpy()
            
            for latency in selected_latencies:
                if latency != baseline_latency:
                    affected_latency_data = all_player_df.filter(
                        (pl.col('latency') == latency) &
                        (pl.col('input') == input_col) &
                        (pl.col('player').is_in(affected_players))
                    )[plot_column].to_numpy()
                    
                    non_affected_latency_data = all_player_df.filter(
                        (pl.col('latency') == latency) &
                        (pl.col('input') == input_col) &
                        (~pl.col('player').is_in(affected_players))
                    )[plot_column].to_numpy()
                    
                    # Affected players analysis
                    if len(affected_data) > 0 and len(affected_latency_data) > 0:
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
                    if len(non_affected_data) > 0 and len(non_affected_latency_data) > 0:
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