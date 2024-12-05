import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import DataLoader, PlotGenerator, StatisticalAnalyzer, StyleHelper
from typing import List

class StreamlitUI:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.plot_generator = PlotGenerator()

        # Weapon mappings for visualization
        self.weapon_mapping = {
            'MOD_ROCKET': ['MOD_ROCKET_kill_count', 'MOD_ROCKET_SPLASH_kill_count'],
            'MOD_PLASMA': ['MOD_PLASMA_kill_count', 'MOD_PLASMA_SPLASH_kill_count'],
            'MOD_GRENADE': ['MOD_GRENADE_kill_count', 'MOD_GRENADE_SPLASH_kill_count'],
            'MOD_SHOTGUN': ['MOD_SHOTGUN_kill_count'],
            'MOD_MACHINEGUN': ['MOD_MACHINEGUN_kill_count'],
            'MOD_LIGHTNING': ['MOD_LIGHTNING_kill_count'],
            'MOD_RAILGUN': ['MOD_RAILGUN_kill_count']
        }

        st.set_page_config(page_title="Weapon Range Analysis", layout="wide")
        st.title("Weapon Range Analysis by Latency")
        st.write("""
        Analysis of weapon usage patterns based on range:
        - Short Range: Shotgun, Lightning Gun, Grenade
        - Long Range: Railgun, Machine Gun, Rocket Launcher, Plasma Gun
        Note: Splash damage kills are combined with direct damage kills.
        """)

    def run(self):
        # Load data
        weapons_df = self.data_loader.load_data("killWeapons")
        player_categories = self.data_loader.load_player_categories()
        
        if weapons_df is None or player_categories is None:
            st.error("Failed to load data")
            return

        # Add categories to dataframe
        weapons_df = self.data_loader.add_player_categories(weapons_df, player_categories)
        
        # Sidebar filters
        st.sidebar.markdown("### Filters")
        
        # Get common filters
        players = sorted(set(weapons_df['player']))
        maps = sorted(set(weapons_df['map']))
        latencies = sorted(set(weapons_df['latency']))
        
        selected_players = st.sidebar.multiselect("Select Players", players, default=players)
        selected_maps = st.sidebar.multiselect("Select Maps", maps, default=maps)
        selected_latencies = st.sidebar.multiselect(
            "Select Latencies", 
            latencies, 
            default=latencies
        )
        
        # Filter data based on selections
        filtered_df = weapons_df[
            (weapons_df['player'].isin(selected_players)) &
            (weapons_df['map'].isin(selected_maps)) &
            (weapons_df['latency'].isin(selected_latencies))
        ]

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
            return
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Weapon Distribution", "Latency Impact", "Statistical Analysis"])
        
        with tab1:
            self.render_weapon_distribution(filtered_df)
        with tab2:
            self.render_latency_analysis(filtered_df)
        with tab3:
            self.render_statistical_tab(filtered_df)

    def render_weapon_distribution(self, df: pd.DataFrame):
        st.subheader("Weapon Distribution by Latency")
        
        latencies = sorted(df['latency'].unique())
        categories = ['Affected', 'Non-Affected']
        
        # Create and display the distribution plot
        fig = self.plot_generator.create_weapon_distribution_plot(
            df, latencies, categories, self.weapon_mapping
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_latency_analysis(self, df: pd.DataFrame):
        st.subheader("Latency Impact Analysis")

        # Show kills distribution with stacked bars
        st.write("#### Weapon Range Distribution by Kills")
        
        summary_df = df.groupby(['player_category', 'latency']).agg({
            'Short Range_kills': 'sum',
            'Long Range_kills': 'sum'
        }).reset_index()
        
        for category in ['Affected', 'Non-Affected']:
            category_data = summary_df[summary_df['player_category'] == category]
            
            # Create stacked bar chart
            fig = self.plot_generator.create_range_distribution_plot(
                category_data, category
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display breakdown table
            st.write(f"###### {category} Players - Usage Breakdown")
            total_kills = (category_data['Short Range_kills'] + 
                         category_data['Long Range_kills'])
            
            display_df = pd.DataFrame({
                'latency': category_data['latency'],
                'Short Range_kills': category_data['Short Range_kills'],
                'Short Range %': (category_data['Short Range_kills'] / total_kills * 100).round(1),
                'Long Range_kills': category_data['Long Range_kills'],
                'Long Range %': (category_data['Long Range_kills'] / total_kills * 100).round(1),
                'Total Kills': total_kills
            })
            
            st.dataframe(display_df.style.format({
                'Short Range %': '{:.1f}%',
                'Long Range %': '{:.1f}%',
                'Short Range_kills': '{:,.0f}',
                'Long Range_kills': '{:,.0f}',
                'Total Kills': '{:,.0f}'
            }), use_container_width=True)

        # Show percentage distributions
        st.write("""
        #### Short Range Weapon Usage Distribution
        Note: Long Range percentage is complementary (100% - Short Range %)
        """)
        
        # Single box plot for short range
        box_fig = self.plot_generator.create_box_plot(
            df, 'latency', 'Short Range_percentage', 'player_category',
            'Short Range Weapon Usage by Latency'
        )
        st.plotly_chart(box_fig, use_container_width=True)
        
        # Show trend analysis for short range only
        st.write("#### Weapon Usage Trends")
        summary = df.groupby(['latency', 'player_category'])['Short Range_percentage'].mean().reset_index()
        
        fig = self.plot_generator.create_line_plot(
            x_data=summary['latency'].values,
            y_data=summary['Short Range_percentage'].values,
            title="Average Short Range Weapon Usage",
            x_label="Latency (ms)",
            y_label="Usage Percentage (%)",
            group_column=summary['player_category'].values
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_statistical_tab(self, df: pd.DataFrame):
        st.subheader("""Statistical Analysis
        Note: Analysis shown for Short Range weapons. Long Range effects are inverse due to complementary nature.""")
        
        # Prepare results lists
        within_group_results = []
        between_group_results = []
        baseline_latency = df['latency'].min()
        
        # Within-group analysis (comparing each latency to baseline)
        for group in ['Affected', 'Non-Affected']:
            group_data = df[df['player_category'] == group]
            for latency in sorted(df['latency'].unique()):
                if latency == baseline_latency:
                    continue
                    
                # Analyze short range percentages
                baseline_data = group_data[group_data['latency'] == baseline_latency]['Short Range_percentage'].values
                current_data = group_data[group_data['latency'] == latency]['Short Range_percentage'].values
                
                results = StatisticalAnalyzer.perform_analysis(baseline_data, current_data)
                if results:
                    results['Group'] = group
                    results['Latency'] = latency
                    within_group_results.append(results)
        
        # Between-group analysis (Affected vs Non-Affected)
        for latency in sorted(df['latency'].unique()):
            affected_data = df[(df['player_category'] == 'Affected') & 
                             (df['latency'] == latency)]['Short Range_percentage'].values
            non_affected_data = df[(df['player_category'] == 'Non-Affected') & 
                                 (df['latency'] == latency)]['Short Range_percentage'].values
            
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
        st.write("#### Within-group Analysis (Comparison to Baseline)")
        styled_within = within_df[['Group', 'Latency', 'percent_change', 
                                 'effect_size', 'p_value', 'Significance']].style.\
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
        st.write("#### Between-group Analysis (Affected vs Non-Affected)")
        styled_between = between_df[['Latency', 'percent_change', 
                                   'effect_size', 'p_value', 'Significance']].style.\
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
        st.write("#### Effect Analysis")
            
        # Change % visualization
        st.write("##### Change % Comparison")
        change_fig = self.plot_generator.create_line_plot(
            x_data=within_df['Latency'].values,
            y_data=within_df['percent_change'].values,
            title="Short Range Usage Change % Across Latencies",
            x_label="Latency (ms)",
            y_label="Change %",
            group_column=within_df['Group'].values
        )
        change_fig.add_trace(go.Scatter(
            x=between_df['Latency'],
            y=between_df['percent_change'],
            name='Between Groups',
            mode='lines+markers',
            line=dict(color=PlotGenerator.COLOR_MAP['Between Groups'], dash='dash', width=2),
            marker=dict(color=PlotGenerator.COLOR_MAP['Between Groups'])
        ))
        st.plotly_chart(change_fig, use_container_width=True)

        # Effect size visualization
        st.write("##### Effect Size Comparison")
        effect_fig = self.plot_generator.create_line_plot(
            x_data=within_df['Latency'].values,
            y_data=within_df['effect_size'].values,
            title="Short Range Usage Effect Sizes Across Latencies",
            x_label="Latency (ms)",
            y_label="Effect Size (Cohen's d)",
            group_column=within_df['Group'].values
        )
        effect_fig.add_trace(go.Scatter(
            x=between_df['Latency'],
            y=between_df['effect_size'],
            name='Between Groups',
            mode='lines+markers',
            line=dict(color=PlotGenerator.COLOR_MAP['Between Groups'], dash='dash', width=2),
            marker=dict(color=PlotGenerator.COLOR_MAP['Between Groups'])
        ))
        
        # Add reference lines
        for value, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            effect_fig.add_hline(
                y=value,
                line_dash="dot",
                annotation_text=label,
                annotation_position="top right",
                line_color="gray"
            )
            effect_fig.add_hline(
                y=-value,
                line_dash="dot",
                line_color="gray"
            )
        
        st.plotly_chart(effect_fig, use_container_width=True)

# Initialize and run the app
if __name__ == "__main__":
    data_loader = DataLoader()
    ui = StreamlitUI(data_loader)
    ui.run()
else:
    data_loader = DataLoader()
    ui = StreamlitUI(data_loader)
    ui.run()