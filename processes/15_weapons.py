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
        self.input_csv['player'] = self.input_csv['player_ip']
        for weapon in self.weapons:
            self.input_csv[f'{weapon}_kill_count'] = 0
        
# Function to safely convert to string and handle NaN values
    def safe_str(self, x):
        return str(x) if pd.notnull(x) else np.nan
    
    def filter_player(self):
        #self.players = self.input_csv["killer_ip" and "victim_ip"].unique()
        #print(self.players)
        self.players_stats = {}
        for player in self.players:
            self.players_stats[player] = {}
            for map in self.input_csv['map'].unique():
                self.players_stats[player][map] = {}
                for latency in self.input_csv['latency'].unique():
                    self.players_stats[player][map][latency] = self.input_csv[(self.input_csv["player_ip"] == player) & (self.input_csv["latency"] == latency) & (self.input_csv['map'] == map)]                 
    def weapon_count(self):
        for player in self.players:
            for map in self.input_csv['map'].unique():
                for latency in self.input_csv['latency'].unique():
                    player_weapons_count = self.players_stats[player][map][latency]["weapon"].value_counts()
                    for i, player_weapon_count in enumerate(player_weapons_count):
                        self.players_stats[player][map][latency].loc[:,f'{player_weapons_count.keys()[i]}_kill_count'] = player_weapon_count

    def aggregate(self):
        aggregated_df = pd.DataFrame()
        for player in self.players:
            for map in self.input_csv['map'].unique():
                for latency in self.input_csv['latency'].unique():
                    aggregated_df = pd.concat([aggregated_df, self.players_stats[player][map][latency]],ignore_index = True)
            
        aggregated_df.drop(columns={
            'timestamp', 'game_round', 'killer_ip', 'victim_ip', 'weapon_id',
            'log_line', 'event', 'killer_id', 'victim_id', 'weapon', 
            'score', 'points', 'player_id', 'log_score', 'player_ip'
        }, inplace = True)
        aggregated_df = aggregated_df.drop_duplicates()
        return aggregated_df

    def extract(self):
        #print(len(self.input_csv))
        self.filter_player()
        self.weapon_count()
        aggregated_csv = self.aggregate()
        aggregated_csv.to_csv(output_path, index=False)
        print(f"Weapon kill counts, player use counts, game round, map, latency are mapped into {output_path}")
        


# Columns needed: 
# game_round,map,latency, weapon+_id, weapon, 
# Create rows: f'{weapon}_count', f'{player}_count


# Final DF
# round, map, latency, weapon_count, player_count
if __name__ == "__main__":
    from config import PROCESSED_DATA_FOLDER
    input_path = f'{PROCESSED_DATA_FOLDER}/ignore_suicides.csv' ##path
    output_path = f'{PROCESSED_DATA_FOLDER}/processed_killWeapons_data.csv' ##path
    processor = WeaponStatsExtractor(input_path, output_path)
    processor.extract()

    
    
'''
 What to do:
 
 use player_IP as a filter, and in each player get the total number of weapon kill counts
 so that it could be used to filtering in streamlit
 '''   