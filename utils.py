import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from dataclasses import dataclass
from typing import List
import json
from config import PROCESSED_DATA_FOLDER
import streamlit as st

@dataclass
class NormalityStats:
    sample_size: int
    mean: float
    median: float
    std_dev: float
    skewness: float
    kurtosis: float
    shapiro_pvalue: float
    ks_pvalue: float

    @classmethod
    def from_data(cls, data: np.ndarray) -> 'NormalityStats':
        return cls(
            sample_size=len(data),
            mean=np.mean(data),
            median=np.median(data),
            std_dev=np.std(data),
            skewness=stats.skew(data),
            kurtosis=stats.kurtosis(data),
            shapiro_pvalue=stats.shapiro(data)[1] if len(data) >= 3 else np.nan,
            ks_pvalue=stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))[1]
        )

class DataLoader:
    def __init__(self):
        self.data_folder = PROCESSED_DATA_FOLDER

    def load_data(self, data_type: str = "kills") -> pd.DataFrame:
        return self._cached_load_data(self.data_folder, data_type)

    @staticmethod
    @st.cache_data
    def _cached_load_data(data_folder: str, data_type: str) -> pd.DataFrame:
        return pd.read_csv(f'{data_folder}/processed_{data_type}_data.csv')

    @staticmethod
    @st.cache_data
    def load_player_categories() -> dict:
        with open(f'{PROCESSED_DATA_FOLDER}/player_categories.json', 'r') as f:
            return json.load(f)

    @staticmethod
    def add_player_categories(df: pd.DataFrame, player_categories: dict) -> pd.DataFrame:
        df['player_category'] = df['player'].apply(
            lambda x: 'Affected' if x in player_categories['affected']
            else 'Non-Affected' if x in player_categories['non_affected']
            else 'Unknown'
        )
        return df

class StatisticalAnalyzer:
    @staticmethod
    def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se if pooled_se != 0 else 0

    @staticmethod
    def perform_analysis(baseline_data: np.ndarray, comparison_data: np.ndarray):
        """Perform statistical analysis comparing two groups"""
        if len(baseline_data) == 0 or len(comparison_data) == 0:
            return None
            
        t_stat, p_value = stats.ttest_ind(baseline_data, comparison_data)
        effect_size = StatisticalAnalyzer.calculate_effect_size(comparison_data, baseline_data)
        percent_change = ((np.mean(comparison_data) - np.mean(baseline_data)) / 
                         np.mean(baseline_data) * 100) if np.mean(baseline_data) != 0 else 0
        
        return {
            'baseline_mean': np.mean(baseline_data),
            'comparison_mean': np.mean(comparison_data),
            'percent_change': percent_change,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'n_baseline': len(baseline_data),
            'n_comparison': len(comparison_data)
        }

class PlotGenerator:
    # Define color scheme as a class variable for consistency
    COLOR_MAP = {
        'Affected': 'red',
        'Non-Affected': 'blue',
        'Unknown': 'gray',
        'Between Groups': 'green'  # Added for between groups comparison
    }

    @staticmethod
    def create_cdf_plot(data: np.ndarray, title: str, player_category: str = None) -> go.Figure:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        mu, sigma = np.mean(sorted_data), np.std(sorted_data)
        x_theoretical = np.linspace(min(sorted_data), max(sorted_data), 100)
        cdf_theoretical = stats.norm.cdf(x_theoretical, mu, sigma)
        
        # Use category-specific color if provided
        empirical_color = PlotGenerator.COLOR_MAP.get(player_category, 'blue')
        
        fig = go.Figure([
            go.Scatter(x=sorted_data, y=cdf, mode='lines', name='Empirical CDF', 
                      line_color=empirical_color),
            go.Scatter(x=x_theoretical, y=cdf_theoretical, mode='lines', name='Normal CDF', 
                      line=dict(color='black', dash='dash'))
        ])
        fig.update_layout(
            title=title, 
            xaxis_title="Value", 
            yaxis_title="Cumulative Probability",
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey')
        )
        return fig

    @staticmethod
    def create_qq_plot(data: np.ndarray, title: str, player_category: str = None) -> go.Figure:
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        sorted_data = np.sort(data)
        min_val, max_val = min(min(theoretical_quantiles), min(sorted_data)), max(max(theoretical_quantiles), max(sorted_data))
        
        # Use category-specific color if provided
        marker_color = PlotGenerator.COLOR_MAP.get(player_category, 'blue')
        
        fig = go.Figure([
            go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', 
                      name='Q-Q Plot', marker_color=marker_color),
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', 
                      name='Reference Line', line=dict(color='black', dash='dash'))
        ])
        fig.update_layout(
            title=title, 
            xaxis_title="Theoretical Quantiles", 
            yaxis_title="Sample Quantiles",
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey')
        )
        return fig

    @staticmethod
    def create_line_plot(x_data: np.ndarray, y_data: np.ndarray, 
                        title: str, x_label: str, y_label: str,
                        group_column: str = None) -> go.Figure:
        fig = go.Figure()
        
        if group_column is not None:
            for group in sorted(set(group_column)):
                mask = group_column == group
                fig.add_trace(go.Scatter(
                    x=x_data[mask],
                    y=y_data[mask],
                    name=group,
                    mode='lines+markers',
                    line=dict(
                        color=PlotGenerator.COLOR_MAP.get(group, 'gray'),
                        width=2
                    ),
                    marker=dict(
                        color=PlotGenerator.COLOR_MAP.get(group, 'gray')
                    )
                ))
        else:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                line=dict(width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=400,
            showlegend=True if group_column is not None else False,
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey')
        )
        return fig
    
    @staticmethod
    def create_histogram(data: pd.DataFrame, value_column: str, 
                        category_column: str = None, title: str = "Frequency Distribution",
                        nbins: int = 30) -> go.Figure:
        """
        Create a histogram of the data with optional category-based coloring
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataframe containing the data
        value_column : str
            The column name containing the values to plot
        category_column : str, optional
            The column name containing categories for coloring
        title : str
            Title of the plot
        nbins : int
            Number of bins in the histogram
        
        Returns:
        --------
        go.Figure
            Plotly figure object containing the histogram
        """
        fig = go.Figure()
        
        if category_column is not None:
            # Create separate histogram for each category
            for category in sorted(data[category_column].unique()):
                category_data = data[data[category_column] == category][value_column]
                fig.add_trace(go.Histogram(
                    x=category_data,
                    name=category,
                    opacity=0.75,
                    nbinsx=nbins,
                    marker_color=PlotGenerator.COLOR_MAP.get(category, 'gray')
                ))
            fig.update_layout(barmode='overlay')
        else:
            # Single histogram for all data
            fig.add_trace(go.Histogram(
                x=data[value_column],
                nbinsx=nbins,
                marker_color='blue'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=value_column,
            yaxis_title="Count",
            plot_bgcolor='white',
            showlegend=category_column is not None,
            xaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey')
        )
        
        return fig

    @staticmethod
    def create_box_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_col: str, title: str) -> go.Figure:
        fig = go.Figure()
        
        for group in sorted(df[color_col].unique()):
            group_data = df[df[color_col] == group]
            fig.add_trace(go.Box(
                x=group_data[x_col],
                y=group_data[y_col],
                name=group,
                boxpoints='outliers',
                marker_color=PlotGenerator.COLOR_MAP.get(group, 'gray'),
                line_color=PlotGenerator.COLOR_MAP.get(group, 'gray')
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500,
            showlegend=True,
            boxmode='group',
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey', zerolinecolor='lightgrey')
        )
        return fig

    @staticmethod
    def create_ks_test_plot(results_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        # Define a color sequence for latencies
        colors = px.colors.qualitative.Set3  # or any other color sequence you prefer
        
        # Add bars for each latency with distinct colors
        for i, latency in enumerate(sorted(results_df['Latency'].unique())):
            latency_data = results_df[results_df['Latency'] == latency]
            color_idx = i % len(colors)  # Cycle through colors if more latencies than colors
            
            fig.add_trace(go.Bar(
                name=f'{latency} ms',
                x=latency_data['Player'],
                y=latency_data['K-S p-value'],
                text=latency_data['K-S p-value'].round(3),
                textposition='outside',
                marker_color=colors[color_idx]
            ))
        
        # Update layout for grouped bars
        fig.update_layout(
            barmode='group',
            title='K-S Test p-values by Player and Latency',
            xaxis_title='Player',
            yaxis_title='K-S p-value',
            xaxis_tickangle=-45,
            showlegend=True,
            legend=dict(
                title="Latency",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            bargap=0.2,
            bargroupgap=0.1,
            margin=dict(l=50, r=50, t=80, b=100),
            height=500,
            plot_bgcolor='white',
            yaxis=dict(
                range=[0, max(results_df['K-S p-value']) * 1.1],
                gridcolor='lightgrey',
                zerolinecolor='lightgrey'
            ),
            xaxis=dict(
                gridcolor='lightgrey',
                zerolinecolor='lightgrey'
            )
        )
        
        # Add significance threshold line
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red"
        )
        
        # Add annotation for threshold line
        fig.add_annotation(
            x=1,
            y=0.05,
            text="α = 0.05",
            showarrow=False,
            yshift=10,
            xshift=10,
            font=dict(size=10, color="red")
        )
        
        return fig

class StyleHelper:
    @staticmethod
    def style_numeric_value(v, color_func):
        try:
            val = float(v)
            color = color_func(val)
            return f'background-color: {color}; color: black'
        except:
            return ''

    @staticmethod
    def style_effect_size(v):
        def get_color(val):
            abs_val = abs(val)
            if abs_val >= 0.8:
                return '#ff0000'
            elif abs_val >= 0.5:
                return '#ff6666'
            elif abs_val >= 0.2:
                return '#ffcccc'
            else:
                return '#f8f8f8'
        return StyleHelper.style_numeric_value(v, get_color)

    @staticmethod
    def style_change_percent(v):
        def get_color(val):
            if val > 0:
                if val >= 30:
                    return '#00cc00'
                elif val >= 20:
                    return '#40ff40'
                elif val >= 10:
                    return '#80ff80'
                else:
                    return '#e6ffe6'
            else:
                abs_val = abs(val)
                if abs_val >= 30:
                    return '#ff0000'
                elif abs_val >= 20:
                    return '#ff4040'
                elif abs_val >= 10:
                    return '#ff8080'
                else:
                    return '#ffe6e6'
        return StyleHelper.style_numeric_value(v, get_color)

    @staticmethod
    def style_significance(v):
        if v == "✓":
            return 'background-color: #98fb98; color: black'
        return 'background-color: #ffcccb; color: black'

    @staticmethod
    def get_style_guides():
        effect_size_guide = """
            **Effect Size (Cohen's d) Guide:**
            - <span style='background-color: #ff0000; padding: 2px 6px;'>|d| ≥ 0.8: Large effect</span>
            - <span style='background-color: #ff6666; padding: 2px 6px;'>|d| ≥ 0.5: Medium effect</span>
            - <span style='background-color: #ffcccc; padding: 2px 6px;'>|d| ≥ 0.2: Small effect</span>
            - <span style='background-color: #f8f8f8; padding: 2px 6px;'>|d| < 0.2: Negligible effect</span>
        """
        
        percent_change_guide = """
            **Percent Change Guide:**
            Increases:
            - <span style='background-color: #00cc00; padding: 2px 6px;'>↑ ≥ 30%: Large increase</span>
            - <span style='background-color: #40ff40; padding: 2px 6px;'>↑ ≥ 20%: Medium increase</span>
            - <span style='background-color: #80ff80; padding: 2px 6px;'>↑ ≥ 10%: Small increase</span>
            - <span style='background-color: #e6ffe6; padding: 2px 6px;'>↑ < 10%: Minimal increase</span>
            
            Decreases:
            - <span style='background-color: #ff0000; padding: 2px 6px;'>↓ ≥ 30%: Large decrease</span>
            - <span style='background-color: #ff4040; padding: 2px 6px;'>↓ ≥ 20%: Medium decrease</span>
            - <span style='background-color: #ff8080; padding: 2px 6px;'>↓ ≥ 10%: Small decrease</span>
            - <span style='background-color: #ffe6e6; padding: 2px 6px;'>↓ < 10%: Minimal decrease</span>
        """
        
        significance_guide = """
            **Significance:**
            - <span style='background-color: #98fb98; padding: 2px 6px;'>✓ = Statistically significant (p < 0.05)</span>
            - <span style='background-color: #ffcccb; padding: 2px 6px;'>✗ = Not statistically significant</span>
        """
        
        return effect_size_guide, percent_change_guide, significance_guide