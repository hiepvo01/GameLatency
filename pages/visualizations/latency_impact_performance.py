import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import PROCESSED_DATA_FOLDER
from utils import DataLoader, StatisticalAnalyzer, PlotGenerator, StyleHelper

# Initialize page without set_page_config since it's handled by the main app
class PlayerPerformanceUI:
    def __init__(self):
        self.data_loader = DataLoader()
        self.player_categories = self.data_loader.load_player_categories()
        if not self.player_categories:
            st.error("Failed to load player categories")
            return
        self.plot_generator = PlotGenerator()
        st.title('Latency Impact Performance')

    def load_and_prepare_data(self):
        try:
            # Use DataLoader's cached loading method
            kills_df = self.data_loader.load_data("kills")
            deaths_df = self.data_loader.load_data("deaths")
            
            # Add categories using DataLoader's method
            kills_df = self.data_loader.add_player_categories(kills_df, self.player_categories)
            deaths_df = self.data_loader.add_player_categories(deaths_df, self.player_categories)
            
            # First aggregate by player, map, and latency to get individual player averages
            kills_by_player = (kills_df.groupby(['player', 'latency', 'map'])['frequency']
                             .mean()
                             .reset_index())
            deaths_by_player = (deaths_df.groupby(['player', 'latency', 'map'])['frequency']
                              .mean()
                              .reset_index())
            
            # Re-add player categories after aggregation
            kills_by_player = self.data_loader.add_player_categories(kills_by_player, self.player_categories)
            deaths_by_player = self.data_loader.add_player_categories(deaths_by_player, self.player_categories)
            
            # Merge kills and deaths data
            merged_df = pd.merge(
                kills_by_player,
                deaths_by_player,
                on=['player', 'latency', 'player_category', 'map'],
                suffixes=('_kills', '_deaths')
            )
            
            # Calculate K/D ratio with safety check for zero deaths
            merged_df['kd_ratio'] = merged_df['frequency_kills'] / merged_df['frequency_deaths'].replace(0, 1)
            
            return merged_df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def render_distribution_tab(self, df: pd.DataFrame, metric: str):
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            box_fig = self.plot_generator.create_box_plot(
                df, 'latency', 'metric', 'player_category',
                f"{metric} Distribution by Latency"
            )
            st.plotly_chart(box_fig, use_container_width=True)
        
        with col2:
            # Calculate averages properly - first by player, then by category
            summary = (df.groupby(['latency', 'player_category', 'player'])['metric']
                      .mean()
                      .reset_index()
                      .groupby(['latency', 'player_category'])['metric']
                      .mean()
                      .reset_index())
            
            line_fig = self.plot_generator.create_line_plot(
                summary['latency'].values,
                summary['metric'].values,
                f"Average {metric} Trend",
                "Latency (ms)",
                f"Average {metric}",
                summary['player_category'].values
            )
            st.plotly_chart(line_fig, use_container_width=True)
        
        # Summary statistics with corrected aggregation
        st.subheader("Summary Statistics")
        summary = (df.groupby(['player_category', 'latency', 'player'])['metric']
                  .mean()
                  .reset_index()
                  .groupby(['player_category', 'latency'])
                  .agg({
                      'metric': ['count', 'mean', 'std', 'median']
                  })
                  .round(2)
                  .reset_index())
        
        # Fix column names after aggregation
        summary.columns = ['player_category', 'latency', 'count', 'mean', 'std', 'median']
        
        st.dataframe(
            summary.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'median': '{:.2f}'
            }),
            use_container_width=True
        )

    def render_statistical_tab(self, df: pd.DataFrame, metric: str):
        st.subheader("Statistical Analysis")
        
        # First average by player to ensure each player contributes equally
        player_avg_df = (df.groupby(['player', 'latency', 'player_category'])['metric']
                        .mean()
                        .reset_index())
        
        # Prepare results lists
        within_group_results = []
        between_group_results = []
        baseline_latency = player_avg_df['latency'].min()
        
        # Within-group analysis
        for group in player_avg_df['player_category'].unique():
            group_data = player_avg_df[player_avg_df['player_category'] == group]
            for latency in sorted(player_avg_df['latency'].unique()):
                if latency == baseline_latency:
                    continue
                
                baseline_data = group_data[group_data['latency'] == baseline_latency]['metric'].values
                current_data = group_data[group_data['latency'] == latency]['metric'].values
                
                results = StatisticalAnalyzer.perform_analysis(baseline_data, current_data)
                if results:
                    results['Group'] = group
                    results['Latency'] = latency
                    within_group_results.append(results)
        
        # Between-group analysis
        for latency in sorted(player_avg_df['latency'].unique()):
            affected_data = player_avg_df[(player_avg_df['player_category'] == 'Affected') & 
                                        (player_avg_df['latency'] == latency)]['metric'].values
            non_affected_data = player_avg_df[(player_avg_df['player_category'] == 'Non-Affected') & 
                                            (player_avg_df['latency'] == latency)]['metric'].values
            
            results = StatisticalAnalyzer.perform_analysis(non_affected_data, affected_data)
            if results:
                results['Latency'] = latency
                between_group_results.append(results)

        # Convert results to DataFrames
        within_df = pd.DataFrame(within_group_results)
        between_df = pd.DataFrame(between_group_results)
        
        # Add significance markers
        within_df['Significance'] = within_df['p_value'].map(lambda p: "✓" if p < 0.05 else "✗")
        between_df['Significance'] = between_df['p_value'].map(lambda p: "✓" if p < 0.05 else "✗")
        
        # Display within-group analysis
        st.write(f"#### Within-group Analysis (Comparison to Baseline {metric})")
        styled_within = within_df[['Group', 'Latency', 'percent_change', 'effect_size', 
                                 'p_value', 'Significance']].style.\
            format({
                'percent_change': '{:.1f}%',
                'effect_size': '{:.3f}',
                'p_value': '{:.4f}'
            }).\
            map(StyleHelper.style_effect_size, subset=['effect_size']).\
            map(StyleHelper.style_change_percent, subset=['percent_change']).\
            map(StyleHelper.style_significance, subset=['Significance'])
        
        st.dataframe(styled_within, use_container_width=True)
        
        # Display between-group analysis
        st.write(f"#### Between-group Analysis (Affected vs Non-Affected {metric})")
        styled_between = between_df[['Latency', 'percent_change', 'effect_size', 
                                   'p_value', 'Significance']].style.\
            format({
                'percent_change': '{:.1f}%',
                'effect_size': '{:.3f}',
                'p_value': '{:.4f}'
            }).\
            map(StyleHelper.style_effect_size, subset=['effect_size']).\
            map(StyleHelper.style_change_percent, subset=['percent_change']).\
            map(StyleHelper.style_significance, subset=['Significance'])
        
        st.dataframe(styled_between, use_container_width=True)

        # Add style guides
        effect_guide, percent_guide, sig_guide = StyleHelper.get_style_guides()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(effect_guide, unsafe_allow_html=True)
            st.markdown(sig_guide, unsafe_allow_html=True)
        with col2:
            st.markdown(percent_guide, unsafe_allow_html=True)

        # Visualizations
        st.write("#### Change % Comparison")
        change_fig = self.plot_generator.create_line_plot(
            x_data=within_df['Latency'].values,
            y_data=within_df['percent_change'].values,
            title=f"Change % in {metric} Across Latencies",
            x_label="Latency (ms)",
            y_label="Change %",
            group_column=within_df['Group'].values
        )
        st.plotly_chart(change_fig, use_container_width=True)

        st.write("#### Effect Size Comparison")
        effect_fig = self.plot_generator.create_line_plot(
            x_data=within_df['Latency'].values,
            y_data=within_df['effect_size'].values,
            title=f"Effect Sizes on {metric} Across Latencies",
            x_label="Latency (ms)",
            y_label="Effect Size (Cohen's d)",
            group_column=within_df['Group'].values
        )
        st.plotly_chart(effect_fig, use_container_width=True)

    def render_combined_metrics_tab(self, df: pd.DataFrame):
        st.subheader("Combined Metrics Analysis")
        
        category = st.selectbox(
            "Select Player Category",
            df['player_category'].unique()
        )
        
        filtered_df = df[df['player_category'] == category]
        
        # Correct the aggregation to match performance_distribution.py
        metrics_fig = go.Figure()
        
        for metric, color in [('frequency_kills', 'green'), 
                            ('frequency_deaths', 'red'), 
                            ('kd_ratio', 'blue')]:
            # First average by player, then by category
            avg_data = (filtered_df.groupby(['latency', 'player'])[metric]
                       .mean()
                       .reset_index()
                       .groupby('latency')[metric]
                       .mean())
            
            metrics_fig.add_trace(go.Scatter(
                x=avg_data.index,
                y=avg_data.values,
                name=metric.replace('frequency_', '').capitalize(),
                mode='lines+markers',
                line=dict(color=color)
            ))
        
        metrics_fig.update_layout(
            title=f"{category} Players: Combined Performance Metrics",
            xaxis_title="Latency (ms)",
            yaxis_title="Value",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Correlation analysis with corrected aggregation
        st.subheader("Metric Correlations")
        corr_data = (filtered_df.groupby(['latency', 'player'])
                    .agg({
                        'frequency_kills': 'mean',
                        'frequency_deaths': 'mean',
                        'kd_ratio': 'mean'
                    })
                    .reset_index()
                    .groupby('latency')
                    .agg({
                        'frequency_kills': 'mean',
                        'frequency_deaths': 'mean',
                        'kd_ratio': 'mean'
                    })).corr().round(3)
        
        # Rename columns for better display
        corr_data.columns = ['Kills', 'Deaths', 'K/D Ratio']
        corr_data.index = ['Kills', 'Deaths', 'K/D Ratio']
        
        st.dataframe(
            corr_data.style.background_gradient(cmap='RdYlBu', vmin=-1, vmax=1),
            use_container_width=True
        )

    def run(self):
        if not self.player_categories:
            return

        df = self.load_and_prepare_data()
        if df is None:
            return

        # Sidebar controls
        st.sidebar.header('Filters')
        
        metric = st.sidebar.radio(
            "Select Analysis Metric",
            ["Kills", "Deaths", "K/D Ratio"]
        )
        
        available_maps = sorted(df['map'].unique())
        selected_maps = st.sidebar.multiselect(
            "Select Maps",
            available_maps,
            default=available_maps[0]
        )

        # Filter data
        filtered_df = df[df['map'].isin(selected_maps)].copy()
        
        # Set the analysis metric
        if metric == "Kills":
            filtered_df['metric'] = filtered_df['frequency_kills']
        elif metric == "Deaths":
            filtered_df['metric'] = filtered_df['frequency_deaths']
        else:  # K/D Ratio
            filtered_df['metric'] = filtered_df['kd_ratio']

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
            return

        # Analysis tabs
        tab1, tab2, tab3 = st.tabs([
            "Distribution Analysis",
            "Statistical Tests",
            "Combined Metrics View"
        ])

        with tab1:
            self.render_distribution_tab(filtered_df, metric)
        with tab2:
            self.render_statistical_tab(filtered_df, metric)
        with tab3:
            self.render_combined_metrics_tab(df)

# Initialize and run the app only when this file is run directly
if __name__ == "__main__":
    ui = PlayerPerformanceUI()
    ui.run()
else:
    # When imported as a page, create and run the UI
    ui = PlayerPerformanceUI()
    ui.run()