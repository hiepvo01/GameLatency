import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go  # Added this import
from config import PROCESSED_DATA_FOLDER
from utils import DataLoader, StatisticalAnalyzer, PlotGenerator, StyleHelper

class StreamlitUI:
    def __init__(self):
        self.player_categories = DataLoader.load_player_categories()
        if not self.player_categories:
            st.error("Failed to load player categories")
            return
        st.title('Latency Impact Strategy')

    def run(self):
        if not self.player_categories:
            return

        # Sidebar controls
        st.sidebar.header('Filters')
        
        # Data type selection
        data_type = st.sidebar.radio("Select Data Type", ["deaths", "kills"])
        
        try:
            df = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/processed_{data_type}_data.csv')
            df = DataLoader.add_player_categories(df, self.player_categories)
        except Exception as e:
            st.error(f"Error loading {data_type} data: {str(e)}")
            return

        if df is None:
            return

        # Map selection
        available_maps = sorted(df['map'].unique())
        selected_maps = st.sidebar.multiselect(
            "Select Maps", 
            available_maps,
            default=available_maps[0]
        )
        
        # Input type selection
        all_inputs = sorted(df['input_type'].unique())
        mouse_clicks = 'mouse_clicks'
        other_keys = [key for key in all_inputs if key != mouse_clicks]
        
        input_category = st.sidebar.radio(
            "Select Input Category",
            ["Mouse Clicks", "Keyboard Keys"]
        )
        
        if input_category == "Keyboard Keys":
            selected_key = st.sidebar.selectbox(
                "Select Key",
                ["All Keys"] + other_keys
            )
            if selected_key == "All Keys":
                filtered_df = df[df['input_type'].isin(other_keys)]
            else:
                filtered_df = df[df['input_type'] == selected_key]
        else:
            filtered_df = df[df['input_type'] == mouse_clicks]
            
        # Apply map filter
        filtered_df = filtered_df[filtered_df['map'].isin(selected_maps)]

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
            return

        # Analysis tabs
        tab1, tab2 = st.tabs(["Distribution Analysis", "Statistical Tests"])

        with tab1:
            self.render_distribution_tab(filtered_df, input_category, selected_key if input_category == "Keyboard Keys" else "Mouse Clicks")
        with tab2:
            self.render_statistical_tab(filtered_df, input_category)

    def render_distribution_tab(self, df: pd.DataFrame, input_category: str, input_type: str):
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            box_fig = PlotGenerator.create_box_plot(
                df, 'latency', 'frequency', 'player_category',
                f"{input_type} Activity Distribution"
            )
            st.plotly_chart(box_fig, use_container_width=True)
        
        with col2:
            summary = df.groupby(['latency', 'player_category'])['frequency'].mean().reset_index()
            line_fig = PlotGenerator.create_line_plot(
                summary['latency'].values,
                summary['frequency'].values,
                f"Average {input_type} Activity Trend",
                "Latency (ms)",
                "Average Activity Frequency",
                summary['player_category'].values
            )
            st.plotly_chart(line_fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        summary = df.groupby(['player_category', 'latency'])['frequency'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(2).reset_index()
        
        st.dataframe(
            summary.style.format({
                'mean': '{:.2f}',
                'std': '{:.2f}',
                'median': '{:.2f}'
            }),
            use_container_width=True
        )

    def render_statistical_tab(self, df: pd.DataFrame, input_category: str):
        st.subheader("Statistical Analysis")
        
        # Prepare results lists
        within_group_results = []
        between_group_results = []
        baseline_latency = df['latency'].min()
        
        # Within-group analysis
        for group in ['Affected', 'Non-Affected']:
            group_data = df[df['player_category'] == group]
            for latency in sorted(df['latency'].unique()):
                if latency == baseline_latency:
                    continue
                
                baseline_data = group_data[group_data['latency'] == baseline_latency]['frequency'].values
                current_data = group_data[group_data['latency'] == latency]['frequency'].values
                
                results = StatisticalAnalyzer.perform_analysis(baseline_data, current_data)
                if results:
                    results['Group'] = group
                    results['Latency'] = latency
                    within_group_results.append(results)
        
        # Between-group analysis
        for latency in sorted(df['latency'].unique()):
            affected_data = df[(df['player_category'] == 'Affected') & 
                             (df['latency'] == latency)]['frequency'].values
            non_affected_data = df[(df['player_category'] == 'Non-Affected') & 
                                 (df['latency'] == latency)]['frequency'].values
            
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
        st.write("#### Between-group Analysis (Affected vs Non-Affected)")
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

        # Change % visualization
        st.write("#### Change % Comparison")
        change_fig = PlotGenerator.create_line_plot(
            x_data=within_df['Latency'].values,
            y_data=within_df['percent_change'].values,
            title="Change % Across Latencies",
            x_label="Latency (ms)",
            y_label="Change %",
            group_column=within_df['Group'].values
        )
        change_fig.add_trace(go.Scatter(
            x=between_df['Latency'],
            y=between_df['percent_change'],
            name='Between Groups',
            mode='lines+markers',
            line=dict(
                color=PlotGenerator.COLOR_MAP['Between Groups'],  # Use color from COLOR_MAP
                dash='dash',
                width=2
            ),
            marker=dict(
                color=PlotGenerator.COLOR_MAP['Between Groups']  # Match marker color
            )
        ))
        st.plotly_chart(change_fig, use_container_width=True)

        # Effect size visualization
        st.write("#### Effect Size Comparison")
        effect_fig = PlotGenerator.create_line_plot(
            x_data=within_df['Latency'].values,
            y_data=within_df['effect_size'].values,
            title="Effect Sizes Across Latencies",
            x_label="Latency (ms)",
            y_label="Effect Size (Cohen's d)",
            group_column=within_df['Group'].values
        )
        effect_fig.add_trace(go.Scatter(
            x=between_df['Latency'],
            y=between_df['effect_size'],
            name='Between Groups',
            mode='lines+markers',
            line=dict(
                color=PlotGenerator.COLOR_MAP['Between Groups'],  # Use color from COLOR_MAP
                dash='dash',
                width=2
            ),
            marker=dict(
                color=PlotGenerator.COLOR_MAP['Between Groups']  # Match marker color
            )
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
        
# Initialize and run the app only when this file is run directly
if __name__ == "__main__":
    ui = StreamlitUI()
    ui.run()
else:
    # When imported as a page, create and run the UI
    ui = StreamlitUI()
    ui.run()