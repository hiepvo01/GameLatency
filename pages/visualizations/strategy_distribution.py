from utils import (
    DataLoader, PlotGenerator, NormalityStats, StyleHelper,
    StatisticalAnalyzer
)
import streamlit as st
import pandas as pd
from typing import List
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.plot_generator = PlotGenerator()

    def filter_data(self, players: List[str], maps: List[str], latencies: List[float], 
                   input_category: str, selected_key: str = None) -> pd.DataFrame:
        # First filter by common criteria
        filtered = self.df[
            (self.df['player'].isin(players)) &
            (self.df['map'].isin(maps)) &
            (self.df['latency'].isin(latencies))
        ]
        
        # Then apply input type filtering
        if input_category == "Mouse Clicks":
            filtered = filtered[filtered['input_type'] == 'mouse_clicks']
        else:  # Keyboard Keys
            non_mouse_clicks = filtered['input_type'] != 'mouse_clicks'
            if selected_key == "All Keys":
                filtered = filtered[non_mouse_clicks]
            else:
                filtered = filtered[filtered['input_type'] == selected_key]
                
        return filtered

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
        st.set_page_config(page_title="Strategy Distribution", layout="wide")
        st.title("Strategy Distribution")
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
        # Load data and categories
        data_type = st.sidebar.radio("Select Data Type", ["kills", "deaths"])
        df = self.data_loader.load_data(data_type)
        player_categories = self.data_loader.load_player_categories()
        df = self.data_loader.add_player_categories(df, player_categories)
        analyzer = PerformanceAnalyzer(df)
        
        # Sidebar filters
        st.sidebar.markdown("### Filters")
        
        # Input type selection with conditional key selection
        input_category = st.sidebar.radio("Select Input Category", ["Mouse Clicks", "Keyboard Keys"])
        selected_key = None
        
        if input_category == "Keyboard Keys":
            # Get all input types except mouse_clicks
            other_keys = sorted([
                input_type 
                for input_type in df['input_type'].unique() 
                if input_type != 'mouse_clicks'
            ])
            
            selected_key = st.sidebar.selectbox(
                "Select Key",
                ["All Keys"] + other_keys
            )
        
        players = df['player'].unique()
        selected_players = st.sidebar.multiselect("Select Players", players, default=players[0])
        selected_maps = st.sidebar.multiselect("Select Maps", df['map'].unique(), default=df['map'].unique()[0])
        selected_latencies = st.sidebar.multiselect("Select Latencies", sorted(df['latency'].unique()), 
                                                  default=sorted(df['latency'].unique()))
        
        filtered_df = analyzer.filter_data(
            selected_players, 
            selected_maps, 
            selected_latencies,
            input_category,
            selected_key
        )
        
        # Main content tabs
        tab1, tab2 = st.tabs(["Distribution Analysis", "Statistical Tests"])
        
        with tab1:
            self.render_distribution_tab(analyzer, filtered_df, selected_players, selected_latencies)
        with tab2:
            self.render_statistical_tests_tab(analyzer, filtered_df, selected_players, selected_latencies)

    def render_distribution_tab(self, analyzer: PerformanceAnalyzer, filtered_df: pd.DataFrame,
                              selected_players: List[str], selected_latencies: List[float]):
        if not selected_players:
            st.warning("Please select at least one player from the sidebar.")
            return

        st.subheader("Overall Distribution Analysis")
        
        if not filtered_df.empty:
            freq_data = filtered_df['frequency'].values
            stats = NormalityStats.from_data(freq_data)
            
            # Create overall histogram
            st.plotly_chart(
                self.plot_generator.create_histogram(
                    filtered_df,
                    value_column='frequency',
                    category_column='player_category',
                    title="Overall Frequency Distribution",
                    nbins=30
                ),
                use_container_width=True
            )
            
            # Get averaged data for the line plot
            averaged_data = analyzer.get_averaged_data(filtered_df)
            
            st.plotly_chart(self.plot_generator.create_line_plot(
                averaged_data['latency'].values,
                averaged_data['frequency'].values,
                "Average Frequency Distribution by Latency",
                "Latency (ms)",
                "Average Frequency",
                averaged_data['player_category']
            ), use_container_width=True)

            col1, col2 = st.columns(2)
            metrics_df = pd.DataFrame({
                'Value': [f"{stats.mean:.2f}", f"{stats.median:.2f}", 
                         f"{stats.std_dev:.2f}", f"{stats.sample_size}"]
            }, index=['Mean', 'Median', 'Std Dev', 'Sample Size'])
            col1.dataframe(metrics_df)
            
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
            # Add histogram for individual player
            st.plotly_chart(
                self.plot_generator.create_histogram(
                    player_data,
                    value_column='frequency',
                    title=f"Frequency Distribution for {selected_player}",
                    nbins=30
                ),
                use_container_width=True
            )

            col1, col2 = st.columns(2)
            col1.plotly_chart(analyzer.plot_generator.create_cdf_plot(
                player_data['frequency'].values,
                f"CDF of Frequency for {selected_player}",
                player_data['player_category'].iloc[0]
            ), use_container_width=True)
            
            col2.plotly_chart(analyzer.plot_generator.create_qq_plot(
                player_data['frequency'].values,
                f"Q-Q Plot of Frequency for {selected_player}",
                player_data['player_category'].iloc[0]
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
                    f"Average Frequency by Latency for {selected_player}"
                ), use_container_width=True)

    def render_statistical_tests_tab(self, analyzer: PerformanceAnalyzer, filtered_df: pd.DataFrame,
                                   selected_players: List[str], selected_latencies: List[float]):
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

if __name__ == "__main__":
    data_loader = DataLoader()
    ui = StreamlitUI(data_loader)
    ui.run()
else:
    data_loader = DataLoader()
    ui = StreamlitUI(data_loader)
    ui.run()