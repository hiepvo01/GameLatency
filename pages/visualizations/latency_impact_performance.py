import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config import PROCESSED_DATA_FOLDER
from utils import DataLoader, PlotGenerator, StatisticalAnalyzer, StyleHelper

class PlayerPerformanceAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.plot_generator = PlotGenerator()
        
    def calculate_kd_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get the last row for each unique combination to get final counts
        df = df.sort_values('timestamp').groupby(['game_round', 'map', 'latency', 'player_ip']).last().reset_index()
        
        # Calculate total kills (excluding suicides and world deaths)
        kill_columns = [col for col in df.columns if col.startswith('killed_Player')]
        df['total_kills'] = df[kill_columns].sum(axis=1)
        
        # Deaths is already in deaths_total column
        # Calculate K/D ratio
        df['kd_ratio'] = (df['total_kills'] / df['deaths_total']).fillna(0)
        
        return df

    def calculate_all_stats(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Per round performance
        round_stats = df.groupby(['player_category', 'latency', 'game_round']).agg({
            'total_kills': 'mean',
            'deaths_total': 'mean',
            'suicide_count': 'mean'
        }).reset_index()
        
        # Total performance across all rounds
        total_stats = df.groupby(['player_category', 'latency']).agg({
            'total_kills': 'sum',
            'deaths_total': 'sum',
            'suicide_count': 'sum'
        }).reset_index()
        
        return round_stats, total_stats

class StreamlitUI:
    def __init__(self):
        self.data_loader = DataLoader()
        self.plot_generator = PlotGenerator()
        
        st.set_page_config(page_title="Player Performance Analysis", layout="wide")
        st.title("Player Performance Analysis")
        st.write("""
        Analysis of player performance metrics:
        - Total kills and deaths across all matches
        - Performance per individual match
        - Performance comparison between affected and non-affected players
        """)

    def run(self):
        # Load data
        try:
            performance_df = self.data_loader.load_data("performance")
            player_categories = self.data_loader.load_player_categories()
            
            if performance_df is None or player_categories is None:
                st.error("Failed to load data")
                return
            
            # Add categories to dataframe
            performance_df = self.data_loader.add_player_categories(performance_df, player_categories)
            
            # Initialize analyzer and process data
            analyzer = PlayerPerformanceAnalyzer(performance_df)
            performance_df = analyzer.calculate_kd_ratio(performance_df)
            
        except Exception as e:
            st.error(f"Error loading or processing data: {str(e)}")
            return

        # Sidebar filters
        st.sidebar.markdown("### Filters")
        
        # Get common filters
        maps = sorted(performance_df['map'].unique())
        latencies = sorted(performance_df['latency'].unique())
        rounds = sorted(performance_df['game_round'].unique())
        
        selected_maps = st.sidebar.multiselect("Select Maps", maps, default=maps)
        selected_latencies = st.sidebar.multiselect(
            "Select Latencies", 
            latencies, 
            default=latencies
        )
        selected_rounds = st.sidebar.multiselect(
            "Select Rounds",
            rounds,
            default=rounds
        )
        
        # Filter data
        filtered_df = performance_df[
            (performance_df['map'].isin(selected_maps)) &
            (performance_df['latency'].isin(selected_latencies)) &
            (performance_df['game_round'].isin(selected_rounds))
        ]

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Calculate both total and per-round stats
        round_stats, total_stats = analyzer.calculate_all_stats(filtered_df)
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Overall Performance", "Per-Round Analysis", "Statistical Analysis"])
        
        with tab1:
            self.render_overall_performance(total_stats)
        with tab2:
            self.render_per_round_analysis(round_stats, filtered_df)
        with tab3:
            self.render_statistical_analysis(round_stats)

    def render_overall_performance(self, df: pd.DataFrame):
        st.subheader("Overall Performance (Total Across All Matches)")
        
        # Total kills and deaths line plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Line plot for total kills
            kills_fig = self.plot_generator.create_line_plot(
                x_data=df['latency'].values,
                y_data=df['total_kills'].values,
                title="Total Kills Across All Matches",
                x_label="Latency (ms)",
                y_label="Total Kills",
                group_column=df['player_category'].values
            )
            st.plotly_chart(kills_fig, use_container_width=True)
            
        with col2:
            # Line plot for total deaths
            deaths_fig = self.plot_generator.create_line_plot(
                x_data=df['latency'].values,
                y_data=df['deaths_total'].values,
                title="Total Deaths Across All Matches",
                x_label="Latency (ms)",
                y_label="Total Deaths",
                group_column=df['player_category'].values
            )
            st.plotly_chart(deaths_fig, use_container_width=True)
        
        # Summary table of totals
        st.write("#### Total Kills and Deaths by Latency")
        summary_pivot = pd.pivot_table(
            df,
            values=['total_kills', 'deaths_total', 'suicide_count'],
            index='player_category',
            columns='latency',
            aggfunc='sum'
        ).round(0)
        
        st.dataframe(
            summary_pivot.style.format('{:.0f}'),
            use_container_width=True
        )

    def render_per_round_analysis(self, round_stats: pd.DataFrame, full_df: pd.DataFrame):
        st.subheader("Per-Round Performance Analysis")
        
        # Box plots showing distribution of kills/deaths per round
        col1, col2 = st.columns(2)
        
        with col1:
            kills_box = self.plot_generator.create_box_plot(
                round_stats, 'latency', 'total_kills', 'player_category',
                "Kills Distribution per Round"
            )
            st.plotly_chart(kills_box, use_container_width=True)
            
        with col2:
            deaths_box = self.plot_generator.create_box_plot(
                round_stats, 'latency', 'deaths_total', 'player_category',
                "Deaths Distribution per Round"
            )
            st.plotly_chart(deaths_box, use_container_width=True)
        
        # Per-map analysis
        st.write("#### Performance by Map")
        for map_name in sorted(full_df['map'].unique()):
            st.write(f"##### {map_name}")
            
            map_df = full_df[full_df['map'] == map_name]
            map_stats = map_df.groupby(['latency', 'player_category', 'game_round']).agg({
                'total_kills': 'mean',
                'deaths_total': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Line plot for kills
                kills_fig = self.plot_generator.create_line_plot(
                    x_data=map_stats['latency'].values,
                    y_data=map_stats['total_kills'].values,
                    title=f"Kills Trend - {map_name}",
                    x_label="Latency (ms)",
                    y_label="Kills",
                    group_column=map_stats['player_category'].values
                )
                st.plotly_chart(kills_fig, use_container_width=True)
                
            with col2:
                # Line plot for deaths
                deaths_fig = self.plot_generator.create_line_plot(
                    x_data=map_stats['latency'].values,
                    y_data=map_stats['deaths_total'].values,
                    title=f"Deaths Trend - {map_name}",
                    x_label="Latency (ms)",
                    y_label="Deaths",
                    group_column=map_stats['player_category'].values
                )
                st.plotly_chart(deaths_fig, use_container_width=True)
            
            # Summary statistics table
            st.write("###### Performance Summary")
            map_summary = map_stats.groupby(['player_category', 'latency']).agg({
                'total_kills': ['mean', 'std', 'min', 'max'],
                'deaths_total': ['mean', 'std', 'min', 'max']
            }).round(1)
            
            map_summary.columns = [f"{col[0]}_{col[1]}" for col in map_summary.columns]
            st.dataframe(
                map_summary.reset_index().style.format({
                    col: '{:.1f}' for col in map_summary.columns
                }),
                use_container_width=True
            )

    def render_statistical_analysis(self, df: pd.DataFrame):
        st.subheader("Statistical Analysis")
        
        metrics = {
            'total_kills': 'Kills',
            'deaths_total': 'Deaths'
        }
        
        baseline_latency = df['latency'].min()
        
        for metric, title in metrics.items():
            st.write(f"#### {title} Analysis")
            
            # Within-group analysis
            within_group_results = []
            between_group_results = []
            
            for group in ['Affected', 'Non-Affected']:
                group_data = df[df['player_category'] == group]
                for latency in sorted(df['latency'].unique()):
                    if latency == baseline_latency:
                        continue
                    
                    baseline_data = group_data[group_data['latency'] == baseline_latency][metric].values
                    current_data = group_data[group_data['latency'] == latency][metric].values
                    
                    results = StatisticalAnalyzer.perform_analysis(baseline_data, current_data)
                    if results:
                        results['Group'] = group
                        results['Latency'] = latency
                        within_group_results.append(results)
            
            # Between-group analysis
            for latency in sorted(df['latency'].unique()):
                affected_data = df[(df['player_category'] == 'Affected') & 
                                 (df['latency'] == latency)][metric].values
                non_affected_data = df[(df['player_category'] == 'Non-Affected') & 
                                     (df['latency'] == latency)][metric].values
                
                results = StatisticalAnalyzer.perform_analysis(non_affected_data, affected_data)
                if results:
                    results['Latency'] = latency
                    between_group_results.append(results)
            
            if within_group_results:
                within_df = pd.DataFrame(within_group_results)
                between_df = pd.DataFrame(between_group_results)
                
                within_df['Significance'] = within_df['p_value'].map(lambda p: "✓" if p < 0.05 else "✗")
                between_df['Significance'] = between_df['p_value'].map(lambda p: "✓" if p < 0.05 else "✗")
                
                # Display tables
                st.write("##### Within-group Analysis (Comparison to Baseline)")
                self.display_stats_table(within_df)
                
                st.write("##### Between-group Analysis (Affected vs Non-Affected)")
                self.display_stats_table(between_df, has_group=False)

        # Add effect size and significance guides
        effect_guide, percent_guide, sig_guide = StyleHelper.get_style_guides()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(effect_guide, unsafe_allow_html=True)
            st.markdown(sig_guide, unsafe_allow_html=True)
        with col2:
            st.markdown(percent_guide, unsafe_allow_html=True)

    def display_stats_table(self, df: pd.DataFrame, has_group: bool = True):
        columns = ['Latency', 'percent_change', 'effect_size', 'p_value', 'Significance']
        if has_group:
            columns = ['Group'] + columns
            
        styled = df[columns].style.\
            format({
                'percent_change': '{:.1f}%',
                'effect_size': '{:.3f}',
                'p_value': '{:.4f}'
            }).\
            map(StyleHelper.style_effect_size, subset=['effect_size']).\
            map(StyleHelper.style_change_percent, subset=['percent_change']).\
            map(StyleHelper.style_significance, subset=['Significance'])
        
        st.dataframe(styled, use_container_width=True)

# Initialize and run the app
if __name__ == "__main__":
    ui = StreamlitUI()
    ui.run()
else:
    ui = StreamlitUI()
    ui.run()