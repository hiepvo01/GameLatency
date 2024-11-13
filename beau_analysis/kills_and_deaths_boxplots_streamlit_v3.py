import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER, ACTIVITY_FOLDER

# Set the page to wide mode
st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    try:
        player_performance = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/player_performance.csv')
        return player_performance
    except Exception as e:
        st.error(f"Error loading the player performance data: {str(e)}")
        return None

player_performance = load_data()

if player_performance is not None:
    # Streamlit app
    st.title('Player Performance Box Plot Visualization')

    # Sidebar for selections
    st.sidebar.header('Filters')

    # Player selection
    all_players = pd.concat([player_performance['killer_ip'], player_performance['victim_ip']]).unique()
    selected_players = st.sidebar.multiselect('Select Players', all_players)

    # Affected players selection
    affected_players = st.sidebar.multiselect('Select Affected Players', selected_players)

    # Map selection
    all_maps = player_performance['map'].unique()
    selected_maps = st.sidebar.multiselect('Select Maps', all_maps)

    # Event type selection
    event_type = st.sidebar.radio("Select Event Type", ('Deaths', 'Kills'))

    # Normalization method selection
    normalization_method = st.sidebar.radio(
        "Select Normalization Method",
        ('Raw Counts', 'Relativization', 'Z-Score Normalization')
    )

    # Fixed latencies
    latencies = [0, 50, 100, 150, 200]

    # Color palette
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if selected_players and selected_maps:
        # Prepare data
        player_data = player_performance[
            (player_performance['killer_ip' if event_type == 'Kills' else 'victim_ip'].isin(selected_players)) &
            (player_performance['map'].isin(selected_maps))
        ]
        
        event_counts = player_data.groupby(['killer_ip' if event_type == 'Kills' else 'victim_ip', 'latency', 'map']).size().reset_index(name='count')
        
        if normalization_method == 'Relativization':
            # Calculate the mean performance at 0ms latency for each player
            baseline_performance = event_counts[event_counts['latency'] == 0].groupby(
                'killer_ip' if event_type == 'Kills' else 'victim_ip'
            )['count'].mean().reset_index(name='baseline')
            
            # Merge the baseline performance with the main dataset
            event_counts = pd.merge(
                event_counts,
                baseline_performance,
                on='killer_ip' if event_type == 'Kills' else 'victim_ip'
            )
            
            # Calculate relative performance
            event_counts['relative_count'] = event_counts['count'] / event_counts['baseline']
            plot_column = 'relative_count'
            y_axis_title = f'Relative Number of {event_type}'

        elif normalization_method == 'Z-Score Normalization':
            event_counts['normalized_count'] = event_counts.groupby(
                'killer_ip' if event_type == 'Kills' else 'victim_ip'
            )['count'].transform(lambda x: (x - x.mean()) / x.std())
            plot_column = 'normalized_count'
            y_axis_title = f'Z-Score Normalized Number of {event_type}'

        else:  # Raw Counts
            plot_column = 'count'
            y_axis_title = f'Number of {event_type}'

        # Create summary graph at the top
        st.subheader("Summary: All Selected Players at Different Latencies")
        summary_fig = go.Figure()

        for i, player in enumerate(selected_players):
            player_color = color_palette[i % len(color_palette)]
            for latency in latencies:
                player_latency_data = event_counts[
                    (event_counts['killer_ip' if event_type == 'Kills' else 'victim_ip'] == player) & 
                    (event_counts['latency'] == latency)
                ]

                summary_fig.add_trace(go.Box(
                    y=player_latency_data[plot_column],
                    name=f'{player} - {latency}ms {"(Affected)" if player in affected_players else ""}',
                    boxmean=True,
                    marker_color=player_color,
                    line=dict(color=player_color, width=2),
                    boxpoints=False,
                    jitter=0.3,
                    pointpos=-1.8,
                    width=0.6
                ))

        summary_fig.update_layout(
            title=dict(
                text=f'{y_axis_title} per Map (All Players)',
                font=dict(size=30)
            ),
            xaxis_title=dict(
                text='Player - Latency',
                font=dict(size=24)
            ),
            yaxis_title=dict(
                text=y_axis_title,
                font=dict(size=24)
            ),
            height=1080,
            width=1920,
            boxmode='group',
            boxgap=0.1,
            boxgroupgap=0.2,
            font=dict(size=18),
            legend=dict(font=dict(size=18))
        )

        summary_fig.update_xaxes(tickfont=dict(size=18))
        summary_fig.update_yaxes(tickfont=dict(size=18))

        st.plotly_chart(summary_fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'width': 2420,
                'height': 1080,
                'scale': 8
            }
        })

        # Create average plot for affected vs non-affected players
        st.subheader("Average Performance: Affected vs Non-Affected Players")
        avg_fig = go.Figure()

        for affected in [False, True]:
            players_group = affected_players if affected else [p for p in selected_players if p not in affected_players]
            group_name = "Affected" if affected else "Non-Affected"
            group_color = color_palette[0] if affected else color_palette[1]

            for latency in latencies:
                group_data = event_counts[
                    (event_counts['killer_ip' if event_type == 'Kills' else 'victim_ip'].isin(players_group)) & 
                    (event_counts['latency'] == latency)
                ]

                avg_fig.add_trace(go.Box(
                    y=group_data[plot_column],
                    name=f'{group_name} - {latency}ms',
                    boxmean=True,
                    marker_color=group_color,
                    line=dict(color=group_color, width=2),
                    boxpoints=False,
                    jitter=0.3,
                    pointpos=-1.8,
                    width=0.6
                ))

        avg_fig.update_layout(
            title=dict(
                text=f'Average {y_axis_title} per Map (Affected vs Non-Affected Players)',
                font=dict(size=30)
            ),
            xaxis_title=dict(
                text='Group - Latency',
                font=dict(size=38)
            ),
            yaxis_title=dict(
                text=y_axis_title,
                font=dict(size=38)
            ),
            height=1080,
            width=1920,
            boxmode='group',
            boxgap=0.1,
            boxgroupgap=0.2,
            font=dict(size=24),
            legend=dict(font=dict(size=24))
        )

        avg_fig.update_xaxes(tickfont=dict(size=24))
        avg_fig.update_yaxes(tickfont=dict(size=24))

        st.plotly_chart(avg_fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'width': 2420,
                'height': 1080,
                'scale': 8
            }
        })

        # Statistical analysis
        st.write("### Statistical Insights")
        baseline_latency = 0

        for affected in [True, False]:
            players_group = affected_players if affected else [p for p in selected_players if p not in affected_players]
            group_name = "Affected" if affected else "Non-Affected"

            st.write(f"#### {group_name} Players")

            baseline_data = event_counts[
                (event_counts['killer_ip' if event_type == 'Kills' else 'victim_ip'].isin(players_group)) & 
                (event_counts['latency'] == baseline_latency)
            ][plot_column]

            for latency in latencies[1:]:  # Skip the baseline latency
                latency_data = event_counts[
                    (event_counts['killer_ip' if event_type == 'Kills' else 'victim_ip'].isin(players_group)) & 
                    (event_counts['latency'] == latency)
                ][plot_column]

                if normalization_method == 'Relativization':
                    baseline_mean = 1  # The baseline for relative values is 1
                    latency_mean = latency_data.mean()
                    st.write(f"At {latency}ms latency:")
                    st.write(f"- Average relative performance: {latency_mean:.4f}")
                    percent_change = (latency_mean - 1) * 100
                    st.write(f"  Performance {'increased' if percent_change > 0 else 'decreased'} by {abs(percent_change):.2f}% compared to baseline")

                elif normalization_method == 'Z-Score Normalization':
                    latency_mean = latency_data.mean()
                    st.write(f"At {latency}ms latency:")
                    st.write(f"- Average z-score: {latency_mean:.4f}")
                    if latency_mean > 0:
                        st.write(f"  Performance is {latency_mean:.4f} standard deviations above the player's mean")
                    elif latency_mean < 0:
                        st.write(f"  Performance is {abs(latency_mean):.4f} standard deviations below the player's mean")
                    else:
                        st.write("  Performance is at the player's mean")

                else:  # Raw Counts
                    percent_change = ((latency_data.mean() - baseline_data.mean()) / baseline_data.mean()) * 100
                    st.write(f"At {latency}ms latency compared to {baseline_latency}ms:")
                    st.write(f"- Average number of {event_type.lower()} {'increased' if percent_change > 0 else 'decreased'} by {abs(percent_change):.2f}%")

                # T-test is valid for both raw and normalized data
                t_stat, p_value = stats.ttest_ind(baseline_data, latency_data)
                st.write(f"- T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

                if p_value < 0.05:
                    st.write("  This change is statistically significant.")
                else:
                    st.write("  This change is not statistically significant.")

            st.write("---")

        # Compare affected vs non-affected players
        st.write("#### Affected vs Non-Affected Players Comparison")
        for latency in latencies:
            affected_data = event_counts[
                (event_counts['killer_ip' if event_type == 'Kills' else 'victim_ip'].isin(affected_players)) & 
                (event_counts['latency'] == latency)
            ][plot_column]

            non_affected_data = event_counts[
                (event_counts['killer_ip' if event_type == 'Kills' else 'victim_ip'].isin([p for p in selected_players if p not in affected_players])) & 
                (event_counts['latency'] == latency)
            ][plot_column]

            t_stat, p_value = stats.ttest_ind(affected_data, non_affected_data)

            st.write(f"At {latency}ms latency:")
            affected_mean = affected_data.mean()
            non_affected_mean = non_affected_data.mean()
            
            if normalization_method == 'Relativization':
                st.write(f"- Affected players' average relative performance: {affected_mean:.4f}")
                st.write(f"- Non-affected players' average relative performance: {non_affected_mean:.4f}")
            elif normalization_method == 'Z-Score Normalization':
                st.write(f"- Affected players' average z-score: {affected_mean:.4f}")
                st.write(f"- Non-affected players' average z-score: {non_affected_mean:.4f}")
            else:
                st.write(f"- Affected players' average {event_type.lower()}: {affected_mean:.2f}")
                st.write(f"- Non-affected players' average {event_type.lower()}: {non_affected_mean:.2f}")

            st.write(f"- T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

            if p_value < 0.05:
                st.write("  There is a statistically significant difference between affected and non-affected players.")
            else:
                st.write("  There is no statistically significant difference between affected and non-affected players.")

        st.write("---")

    else:
        st.warning('Please select at least one player and one map.')

else:
    st.error("Cannot proceed with analysis due to data loading error.")