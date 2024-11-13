from utils import (
    DataLoader, PlotGenerator, NormalityStats, StyleHelper,
    StatisticalAnalyzer
)
import streamlit as st
import pandas as pd
import numpy as np
from typing import List

class KDAnalyzer:
    def __init__(self, kills_df: pd.DataFrame, deaths_df: pd.DataFrame):
        self.kills_df = kills_df
        self.deaths_df = deaths_df
        self.plot_generator = PlotGenerator()

    def calculate_kd_ratio(self, kills_data: pd.DataFrame, deaths_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate K/D ratio while preserving individual events"""
        # Get total deaths per player, latency, map combination
        deaths_sum = deaths_data.groupby(['player', 'latency', 'map'])['frequency'].sum().reset_index()
        deaths_sum = deaths_sum.rename(columns={'frequency': 'total_deaths'})
        
        # Merge death totals with kill events
        merged_data = kills_data.merge(
            deaths_sum,
            on=['player', 'latency', 'map'],
            how='left'
        )
        
        # Calculate K/D ratio for each kill event
        merged_data['frequency'] = merged_data['frequency'] / merged_data['total_deaths'].replace(0, 1)
        
        # Preserve player category
        merged_data['player_category'] = kills_data['player_category']
        
        return merged_data

    def filter_data(self, df: pd.DataFrame, players: List[str], maps: List[str], 
                   latencies: List[float]) -> pd.DataFrame:
        return df[
            (df['player'].isin(players)) &
            (df['map'].isin(maps)) &
            (df['latency'].isin(latencies))
        ]

    def get_player_data(self, filtered_df: pd.DataFrame, player: str) -> pd.DataFrame:
        return filtered_df[filtered_df['player'] == player]
    
    def get_averaged_data(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        return filtered_df.groupby(['latency', 'player_category'])['frequency'].mean().reset_index()

    def calculate_normality_results(self, filtered_df: pd.DataFrame, players: List[str], 
                                  latencies: List[float]) -> pd.DataFrame:
        results = []
        for player in players:
            for latency in latencies:
                data = filtered_df[
                    (filtered_df['player'] == player) & 
                    (filtered_df['latency'] == latency)
                ]['frequency'].values
                
                if len(data) > 0:
                    stats = NormalityStats.from_data(data)
                    results.append({
                        'Player': player,
                        'Latency': latency,
                        'K-S p-value': stats.ks_pvalue,
                        'Shapiro p-value': stats.shapiro_pvalue,
                        'Skewness': stats.skewness,
                        'Kurtosis': stats.kurtosis,
                        'Is Normal': stats.ks_pvalue > 0.05,
                        'Sample Size': stats.sample_size
                    })
        return pd.DataFrame(results)

class StreamlitUI:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.plot_generator = PlotGenerator()
        st.set_page_config(page_title="Performance Distribution", layout="wide")
        st.title("Performance Distribution")
        self.add_explanation_section()

    @staticmethod
    def add_explanation_section():
        with st.expander("📊 Understanding Normality Tests"):
            st.markdown("""
            ### Interpreting Normality Tests
            - P-value > 0.05: Data is consistent with a normal distribution
            - P-value < 0.05: Data significantly deviates from normal distribution
            
            #### Statistical Tests
            1. **Kolmogorov-Smirnov (K-S) Test**: Compares data's CDF to normal CDF
            2. **Shapiro-Wilk Test**: Better for sample sizes < 50
            
            #### Distribution Measures
            - **Skewness**: Measures symmetry (0 = symmetric)
            - **Kurtosis**: Measures tail weight (0 = normal tails)
            """)

    def run(self):
        # Load data
        kills_df = self.data_loader.load_data("kills")
        deaths_df = self.data_loader.load_data("deaths")
        player_categories = self.data_loader.load_player_categories()
        
        # Add categories to both dataframes
        kills_df = self.data_loader.add_player_categories(kills_df, player_categories)
        deaths_df = self.data_loader.add_player_categories(deaths_df, player_categories)
        
        analyzer = KDAnalyzer(kills_df, deaths_df)
        
        # Sidebar filters
        st.sidebar.markdown("### Filters")
        
        # Metric selection
        metric = st.sidebar.radio("Select Metric", ["Kills", "Deaths", "K/D Ratio"])
        
        # Get common filters
        players = sorted(set(kills_df['player']) & set(deaths_df['player']))
        maps = sorted(set(kills_df['map']) & set(deaths_df['map']))
        latencies = sorted(set(kills_df['latency']) & set(deaths_df['latency']))
        
        selected_players = st.sidebar.multiselect("Select Players", players, default=players[0])
        selected_maps = st.sidebar.multiselect("Select Maps", maps, default=maps[0])
        selected_latencies = st.sidebar.multiselect(
            "Select Latencies", 
            latencies, 
            default=latencies
        )
        
        # Get appropriate dataframe based on metric
        if metric == "Kills":
            filtered_df = analyzer.filter_data(kills_df, selected_players, selected_maps, selected_latencies)
        elif metric == "Deaths":
            filtered_df = analyzer.filter_data(deaths_df, selected_players, selected_maps, selected_latencies)
        else:  # K/D Ratio
            filtered_kills = analyzer.filter_data(kills_df, selected_players, selected_maps, selected_latencies)
            filtered_deaths = analyzer.filter_data(deaths_df, selected_players, selected_maps, selected_latencies)
            filtered_df = analyzer.calculate_kd_ratio(filtered_kills, filtered_deaths)
        
        # Main content tabs
        tab1, tab2 = st.tabs(["Distribution Analysis", "Statistical Tests"])
        
        with tab1:
            self.render_distribution_tab(analyzer, filtered_df, selected_players, selected_latencies, metric)
        with tab2:
            self.render_statistical_tests_tab(analyzer, filtered_df, selected_players, selected_latencies, metric)

    def render_distribution_tab(self, analyzer: KDAnalyzer, filtered_df: pd.DataFrame,
                              selected_players: List[str], selected_latencies: List[float],
                              metric: str):
        if not selected_players:
            st.warning("Please select at least one player from the sidebar.")
            return

        # Overall Distribution Analysis
        st.subheader("Overall Distribution Analysis")
        
        if not filtered_df.empty:
            freq_data = filtered_df['frequency'].values
            stats = NormalityStats.from_data(freq_data)
            
            # Get averaged data for the line plot
            averaged_data = analyzer.get_averaged_data(filtered_df)
            
            st.plotly_chart(analyzer.plot_generator.create_line_plot(
                averaged_data['latency'].values,
                averaged_data['frequency'].values,
                f"Average {metric} Distribution by Latency",
                "Latency (ms)",
                f"Average {metric}",
                averaged_data['player_category']
            ), use_container_width=True)

            col1, col2 = st.columns(2)
            # Distribution metrics
            metrics_df = pd.DataFrame({
                'Value': [f"{stats.mean:.2f}", f"{stats.median:.2f}", 
                         f"{stats.std_dev:.2f}", f"{stats.sample_size}"]
            }, index=['Mean', 'Median', 'Std Dev', 'Sample Size'])
            col1.dataframe(metrics_df)
            
            # Shape analysis
            shape_text = [
                f"Distribution is {'symmetric' if abs(stats.skewness) < 0.5 else 'asymmetric'}",
                f"Tails are {'normal' if abs(stats.kurtosis) < 0.5 else 'non-normal'} weight",
                f"Data {'suggests' if stats.ks_pvalue > 0.05 else 'does not suggest'} normal distribution"
            ]
            col2.markdown("\n".join(f"- {text}" for text in shape_text))

        # Individual Player Analysis
        st.subheader("Individual Player Analysis")
        selected_player = st.selectbox("Select Player", selected_players)
        
        player_data = analyzer.get_player_data(filtered_df, selected_player)
        if not player_data.empty:
            col1, col2 = st.columns(2)
            col1.plotly_chart(analyzer.plot_generator.create_cdf_plot(
                player_data['frequency'].values,
                f"CDF of {metric} for {selected_player}"
            ), use_container_width=True)
            
            col2.plotly_chart(analyzer.plot_generator.create_qq_plot(
                player_data['frequency'].values,
                f"Q-Q Plot of {metric} for {selected_player}"
            ), use_container_width=True)

            player_stats = NormalityStats.from_data(player_data['frequency'].values)
            cols = st.columns(4)
            for (label, value), col in zip(
                [("Mean", player_stats.mean), ("Median", player_stats.median),
                 ("Std Dev", player_stats.std_dev), ("Sample Size", player_stats.sample_size)],
                cols
            ):
                col.metric(label, f"{value:.2f}" if isinstance(value, float) else value)

            if len(selected_latencies) > 1:
                averaged_player_data = analyzer.get_averaged_data(player_data)
                st.plotly_chart(analyzer.plot_generator.create_box_plot(
                    averaged_player_data, 'latency', 'frequency', 'player_category',
                    f"Average {metric} by Latency for {selected_player}"
                ), use_container_width=True)

    def render_statistical_tests_tab(self, analyzer: KDAnalyzer, filtered_df: pd.DataFrame,
                                   selected_players: List[str], selected_latencies: List[float],
                                   metric: str):
        if not selected_players:
            st.warning("Please select at least one player from the sidebar.")
            return

        st.subheader("Normality Test Results")
        
        results_df = analyzer.calculate_normality_results(filtered_df, selected_players, selected_latencies)
        if not results_df.empty:
            st.dataframe(results_df.style.format({
                col: '{:.4f}' for col in ['K-S p-value', 'Shapiro p-value', 'Skewness', 'Kurtosis']
            }))
            
            normal_count = len(results_df[results_df['Is Normal']])
            total = len(results_df)
            col1, col2 = st.columns(2)
            col1.metric("Normal Distributions", f"{normal_count}/{total}")
            col2.metric("Percentage Normal", f"{(normal_count/total)*100:.1f}%")

            st.plotly_chart(analyzer.plot_generator.create_ks_test_plot(results_df),
                          use_container_width=True)
    
# Initialize and run the app only when this file is run directly
if __name__ == "__main__":
    data_loader = DataLoader()
    ui = StreamlitUI(data_loader)
    ui.run()
else:
    # When imported as a page, create and run the UI
    data_loader = DataLoader()
    ui = StreamlitUI(data_loader)
    ui.run()