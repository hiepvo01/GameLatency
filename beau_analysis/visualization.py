import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import BEAU_FOLDER

# Set the page to wide mode
st.set_page_config(layout="wide")

# Cache data loading
@st.cache_data
def load_data():
    try:
        kills_data = pd.read_csv(f'{BEAU_FOLDER}/processed_kills_data.csv')
        deaths_data = pd.read_csv(f'{BEAU_FOLDER}/processed_deaths_data.csv')
        return kills_data, deaths_data
    except Exception as e:
        st.error(f"Error loading the processed data: {str(e)}")
        return None, None

# Load the data
kills_data, deaths_data = load_data()

if kills_data is not None and deaths_data is not None:
    st.title('Player Performance Analysis')

    # Simplified sidebar filters
    st.sidebar.header('Filters')

    # Event type selection
    event_type = st.sidebar.radio("Select Event Type", ('Deaths', 'Kills'))
    
    # Map selection
    all_maps = sorted(kills_data['map'].unique())
    selected_map = st.sidebar.selectbox('Select Map', all_maps)

    # Affected players selection
    all_players = sorted(kills_data['player'].unique())
    affected_players = st.sidebar.multiselect('Select Affected Players', all_players)

    # Get the appropriate dataset based on event type
    current_data = kills_data if event_type == 'Kills' else deaths_data

    if selected_map:
        # Filter data by map
        filtered_data = current_data[current_data['map'] == selected_map]

        # Display summary tables
        st.subheader("Summary Tables: Average Input Frequencies")
        
        def calculate_summary_stats(data, is_affected):
            player_filter = data['player'].isin(affected_players) if is_affected else ~data['player'].isin(affected_players)
            summary = data[player_filter].groupby(['latency', 'input_type'])['frequency'].sum().unstack()
            summary = summary.round(2)
            return summary

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Non-Affected Players")
            non_affected_summary = calculate_summary_stats(filtered_data, False)
            st.dataframe(non_affected_summary)
        
        with col2:
            st.write("Affected Players")
            affected_summary = calculate_summary_stats(filtered_data, True)
            st.dataframe(affected_summary)

        # Display box plots for each input type
        st.subheader(f'Performance Comparison (Map: {selected_map})')
        st.write("Blue boxes: Non-Affected Players | Red boxes: Affected Players")

        input_types = sorted(filtered_data['input_type'].unique())
        
        for input_type in input_types:
            input_data = filtered_data[filtered_data['input_type'] == input_type]
            
            fig = go.Figure()

            for latency in sorted(input_data['latency'].unique()):
                # Non-affected players
                non_affected_data = input_data[
                    (input_data['latency'] == latency) & 
                    (~input_data['player'].isin(affected_players))
                ]['frequency']
                
                # Affected players
                affected_data = input_data[
                    (input_data['latency'] == latency) & 
                    (input_data['player'].isin(affected_players))
                ]['frequency']
                
                # Add box plots
                fig.add_trace(go.Box(
                    y=non_affected_data,
                    name=f'Non-Affected ({latency}ms)',
                    boxmean=True,
                    fillcolor='rgba(0, 0, 255, 0.3)',
                    line=dict(color='rgba(0, 0, 255, 1)'),
                    quartilemethod="linear",
                    width=0.8
                ))
                
                fig.add_trace(go.Box(
                    y=affected_data,
                    name=f'Affected ({latency}ms)',
                    boxmean=True,
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='rgba(255, 0, 0, 1)'),
                    quartilemethod="linear",
                    width=0.8
                ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'{input_type} Frequency Distribution',
                    font=dict(size=24)
                ),
                xaxis_title=dict(text='Player Group and Latency', font=dict(size=18)),
                yaxis_title=dict(text='Frequency', font=dict(size=18)),
                height=800,
                boxmode='group',
                boxgroupgap=0.2,
                boxgap=0.1,
                legend=dict(font=dict(size=16)),
                font=dict(size=14)
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Calculate and display percentage changes
            baseline_latency = sorted(input_data['latency'].unique())[0]
            
            pct_change_fig = go.Figure()
            
            for group in ['Non-Affected', 'Affected']:
                group_data = input_data[~input_data['player'].isin(affected_players) if group == 'Non-Affected' else input_data['player'].isin(affected_players)]
                baseline_avg = group_data[group_data['latency'] == baseline_latency]['frequency'].mean()
                
                pct_changes = []
                latencies = []
                
                for latency in sorted(input_data['latency'].unique()):
                    if latency != baseline_latency:
                        latency_avg = group_data[group_data['latency'] == latency]['frequency'].mean()
                        pct_change = ((latency_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg != 0 else 0
                        pct_changes.append(pct_change)
                        latencies.append(f'{latency}ms')
                
                pct_change_fig.add_trace(go.Bar(
                    x=latencies,
                    y=pct_changes,
                    name=group,
                    marker_color='blue' if group == 'Non-Affected' else 'red'
                ))
            
            pct_change_fig.update_layout(
                title=dict(
                    text=f'Percentage Change in {input_type} Frequency from {baseline_latency}ms Latency',
                    font=dict(size=24)
                ),
                xaxis_title='Latency',
                yaxis_title='Percent Change',
                height=600
            )
            
            st.plotly_chart(pct_change_fig, use_container_width=True)

    else:
        st.warning('Please select a map to view the analysis.')

else:
    st.error("Cannot proceed with analysis due to data loading error.")