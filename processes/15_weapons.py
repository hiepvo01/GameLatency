import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from tqdm import tqdm
import json
import glob

class WeaponStatsExtractor:
    def __init__(self, input_path, output_path):
        self.input_csv = pd.read_csv(input_path)
        self.output_csv = output_path
        self.weapons = self.input_csv['weapon'].unique()
        self.players = self.input_csv["killer_ip" and "victim_ip"].unique()
        self.maps = self.input_csv["map"].unique()
        self.input_csv.loc[:, 'player'] = self.input_csv['player_ip']

        # Define weapon categories
        self.weapon_categories = {
            'Short Range': ['MOD_SHOTGUN', 'MOD_MACHINEGUN', 'MOD_LIGHTNING', 
                          'MOD_GRENADE', 'MOD_GRENADE_SPLASH'],
            'Long Range': ['MOD_RAILGUN', 'MOD_ROCKET', 'MOD_ROCKET_SPLASH', 
                          'MOD_PLASMA', 'MOD_PLASMA_SPLASH']
        }

        # Initialize kill count columns
        for weapon in self.weapons:
            self.input_csv.loc[:, f'{weapon}_kill_count'] = 0
            
        # Initialize category columns
        for category in self.weapon_categories.keys():
            self.input_csv.loc[:, f'{category}_kills'] = 0
        
    def safe_str(self, x):
        """Function to safely convert to string and handle NaN values"""
        return str(x) if pd.notnull(x) else np.nan
    
    def filter_player(self):
        self.players_stats = {}
        for player in self.players:
            self.players_stats[player] = {}
            for map_name in self.input_csv['map'].unique():
                self.players_stats[player][map_name] = {}
                for latency in self.input_csv['latency'].unique():
                    # Create a copy to avoid SettingWithCopyWarning
                    mask = ((self.input_csv["player_ip"] == player) & 
                           (self.input_csv["latency"] == latency) & 
                           (self.input_csv['map'] == map_name))
                    self.players_stats[player][map_name][latency] = self.input_csv[mask].copy()

    def weapon_count(self):
        for player in self.players:
            for map_name in self.input_csv['map'].unique():
                for latency in self.input_csv['latency'].unique():
                    player_data = self.players_stats[player][map_name][latency]
                    if not player_data.empty:
                        # Count individual weapons
                        player_weapons_count = player_data["weapon"].value_counts()
                        for weapon, count in player_weapons_count.items():
                            player_data.loc[:, f'{weapon}_kill_count'] = count
                        
                        # Calculate category totals
                        for category, weapons in self.weapon_categories.items():
                            category_cols = [col for col in player_data.columns 
                                           if any(weapon in col and '_kill_count' in col 
                                                for weapon in weapons)]
                            player_data.loc[:, f'{category}_kills'] = player_data[category_cols].sum(axis=1)
                        
                        # Update the stats dictionary
                        self.players_stats[player][map_name][latency] = player_data

    def calculate_percentages(self, df):
        """Calculate weapon usage percentages"""
        df.loc[:, 'total_kills'] = df['Short Range_kills'] + df['Long Range_kills']
        
        for category in self.weapon_categories.keys():
            mask = df['total_kills'] != 0  # Avoid division by zero
            df.loc[mask, f'{category}_percentage'] = (
                df.loc[mask, f'{category}_kills'] / df.loc[mask, 'total_kills'] * 100
            )
            df.loc[~mask, f'{category}_percentage'] = 0  # Set to 0 where total_kills is 0
            
        return df

    def aggregate(self):
        # Create empty DataFrame with all necessary columns
        first_valid_df = None
        for player in self.players:
            for map_name in self.input_csv['map'].unique():
                for latency in self.input_csv['latency'].unique():
                    if not self.players_stats[player][map_name][latency].empty:
                        first_valid_df = self.players_stats[player][map_name][latency]
                        break
                if first_valid_df is not None:
                    break
            if first_valid_df is not None:
                break
                
        if first_valid_df is None:
            raise ValueError("No valid data found")
            
        aggregated_df = pd.DataFrame(columns=first_valid_df.columns)
        
        # Concatenate all data
        for player in self.players:
            for map_name in self.input_csv['map'].unique():
                for latency in self.input_csv['latency'].unique():
                    player_data = self.players_stats[player][map_name][latency]
                    if not player_data.empty:
                        aggregated_df = pd.concat([aggregated_df, player_data], ignore_index=True)
            
        # Drop unnecessary columns
        columns_to_drop = [
            'timestamp', 'game_round', 'killer_ip', 'victim_ip', 'weapon_id',
            'log_line', 'event', 'killer_id', 'victim_id', 'weapon', 
            'score', 'points', 'player_id', 'log_score', 'player_ip'
        ]
        aggregated_df.drop(columns=columns_to_drop, inplace=True)
        
        # Remove duplicates and calculate percentages
        aggregated_df = aggregated_df.drop_duplicates()
        aggregated_df = self.calculate_percentages(aggregated_df)
        
        return aggregated_df

    def extract(self):
        print("Filtering players...")
        self.filter_player()
        print("Counting weapon usage...")
        self.weapon_count()
        print("Aggregating data...")
        aggregated_csv = self.aggregate()
        aggregated_csv.to_csv(self.output_csv, index=False)
        print(f"Processed data saved to {self.output_csv}")

if __name__ == "__main__":
    from config import PROCESSED_DATA_FOLDER
    input_path = f'{PROCESSED_DATA_FOLDER}/ignore_suicides.csv'
    output_path = f'{PROCESSED_DATA_FOLDER}/processed_killWeapons_data.csv'
    processor = WeaponStatsExtractor(input_path, output_path)
    processor.extract()