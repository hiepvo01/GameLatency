import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go  # Added this import
import plotly.express as px
from config import PROCESSED_DATA_FOLDER
from utils import DataLoader, StatisticalAnalyzer, PlotGenerator, StyleHelper
from typing import List

class KillingWeaponsAnalyzer:
    def __init__(self, weapons_df: pd.DataFrame):
        self.weapons_df = weapons_df
        self.plot_generator = PlotGenerator()
        
    def filter_data(self, df: pd.DataFrame, players: List[str], maps: List[str], 
                   latencies: List[float]) -> pd.DataFrame:
        return df[
            (df['player'].isin(players)) &
            (df['map'].isin(maps)) &
            (df['latency'].isin(latencies))
        ]
        
    def get_player_data(self, filtered_df: pd.DataFrame, player: str) -> pd.DataFrame:
        return filtered_df[filtered_df['player'] == player]

class StreamlitUI:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.plot_generator = PlotGenerator()
        st.set_page_config(page_title="Weapons used for Kill", layout="wide")
        st.title("Weapons used for Kill")
        
    def run(self):
        # Load data
        killWeapons_df = self.data_loader.load_data("killWeapons")
        player_categories = self.data_loader.load_player_categories()
        
        # Add categories to both dataframes
        killWeapons_df = self.data_loader.add_player_categories(killWeapons_df, player_categories)

        analyzer = KillingWeaponsAnalyzer(killWeapons_df)
        
        # Sidebar filters
        st.sidebar.markdown("### Filters")
        
        # Metric selection
        #metric = st.sidebar.radio("Select Metric", ["Kills", "Deaths", "K/D Ratio"])
        
        # Get common filters
        players = sorted(set(killWeapons_df['player']))
        maps = sorted(set(killWeapons_df['map']))
        latencies = sorted(set(killWeapons_df['latency']))
        
        selected_players = st.sidebar.multiselect("Select Players", players, default=players[0])
        selected_maps = st.sidebar.multiselect("Select Maps", maps, default=maps[0])
        selected_latencies = st.sidebar.multiselect(
            "Select Latencies", 
            latencies, 
            default=latencies
        )
        
        filtered_df = analyzer.filter_data(killWeapons_df, selected_players, selected_maps, selected_latencies)
        
        self.render_kill_weapon_count_table(filtered_df, killWeapons_df['player_category'].unique())
        
    def render_kill_weapon_count_table(self, filtered_df: pd.DataFrame, player_category: List[str],
                                        ):
        # Initialize dictionaries to hold weapon counts
        affected_data = {}
        non_affected_data = {}

        for category in player_category:
            category_df = filtered_df[filtered_df['player_category'] == category]
            category_kill_counts = category_df.iloc[:, 3:-1].sum(axis=0)

            if category == "Affected":
                for weapon, count in category_kill_counts.items():
                    affected_data[weapon] = affected_data.get(weapon, 0) + count
            elif category == "Non-Affected":
                for weapon, count in category_kill_counts.items():
                    non_affected_data[weapon] = non_affected_data.get(weapon, 0) + count


        # Create DataFrames for plotting pie charts
        affected_df = pd.DataFrame(list(affected_data.items()), columns=['Weapon', 'Count'])
        non_affected_df = pd.DataFrame(list(non_affected_data.items()), columns=['Weapon', 'Count'])

        # Generate a custom color scale with enough distinct colors for all weapon types
        custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#2e1fb4','#f5f2f2', '#080101','#2e1fb4']

        # Create two pie charts for affected and non-affected players with the custom color scale
        fig_affected = px.pie(affected_df, names='Weapon', values='Count', title='Affected Players Killer Weapons Count',
                            color='Weapon', color_discrete_sequence=custom_colors)
        
        fig_non_affected = px.pie(non_affected_df, names='Weapon', values='Count', title='Non-affected Players Killer Weapons Count',
                                color='Weapon', color_discrete_sequence=custom_colors)

        # Display the pie charts
        st.plotly_chart(fig_affected)
        st.plotly_chart(fig_non_affected)


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

