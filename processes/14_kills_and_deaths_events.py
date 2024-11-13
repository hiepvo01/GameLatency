import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
from tqdm import tqdm
import json
import glob

class DataProcessor:
    def __init__(self, processed_data_folder, activity_folder, output_folder, raw_data_folder):
        self.processed_data_folder = processed_data_folder
        self.activity_folder = activity_folder
        self.output_folder = output_folder
        self.raw_data_folder = raw_data_folder
        self.input_columns = ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']
        self.time_range = (-5.0, 1.0)
        self.affected_players = None
        self.log_file = None
        self._find_log_file()
        
    def _find_log_file(self):
        """Find the .log file in the raw data folder."""
        log_files = glob.glob(os.path.join(self.raw_data_folder, '*.log'))
        if not log_files:
            raise FileNotFoundError("No .log file found in raw data folder")
        self.log_file = os.path.basename(log_files[0])

    def _format_timestamp(self, timestamp):
        """Convert Unix timestamp to formatted datetime string."""
        try:
            # Try to parse as a float Unix timestamp
            datetime_obj = datetime.fromtimestamp(float(timestamp))
            date_str = datetime_obj.strftime('%Y-%m-%d')
            time_str = datetime_obj.strftime('%H:%M:%S')
            return f"{date_str} {time_str}"
        except ValueError:
            # If it fails, assume the timestamp is already formatted
            return timestamp

    def _load_affected_players(self):
        """Load affected players from the text file based on the log file name."""
        affected_file_path = os.path.join(self.raw_data_folder, 'effected_players.txt')
        
        try:
            with open(affected_file_path, 'r') as f:
                lines = f.readlines()
            
            # Clean up lines
            lines = [line.strip() for line in lines if line.strip()]
            log_name = self.log_file.replace('.log', '')
            
            print(f"Looking for affected players for log file: {log_name}")
            
            # Process the lines
            affected_ips = []
            i = 0
            while i < len(lines):
                current_line = lines[i].strip()
                if current_line == log_name:
                    # Next two lines contain the IPs
                    if i + 2 <= len(lines):
                        ip1 = lines[i + 1].strip()
                        ip2 = lines[i + 2].strip()
                        affected_ips.extend([f"Player_{ip1}", f"Player_{ip2}"])
                i += 1
            
            # Remove duplicates while preserving order
            self.affected_players = list(dict.fromkeys(affected_ips))
            
            print(f"Found {len(self.affected_players)} affected players for log file {self.log_file}")
            if self.affected_players:
                print("Affected players:", self.affected_players)
            else:
                print("No affected players found for this log file")
            
        except Exception as e:
            print(f"Error loading affected players: {str(e)}")
            self.affected_players = []

    def _save_player_categories(self):
        """Save player categories to JSON file."""
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Create categories dictionary
        categories = {
            "affected": self.affected_players,
            "non_affected": []  # Will be filled later with other players
        }
        
        with open(f'{self.output_folder}/player_categories.json', 'w') as f:
            json.dump(categories, f, indent=4)

    def process_all_data(self):
        """Process all player data and save results."""
        print(f"Processing data for log file: {self.log_file}")
        
        # Load affected players first
        print("Loading affected players list...")
        self._load_affected_players()
        
        # Load player performance data
        print("Loading player performance data...")
        player_performance = pd.read_csv(f'{self.processed_data_folder}/player_performance.csv')
        player_performance['timestamp'] = player_performance['timestamp'].apply(self._format_timestamp)
        player_performance['timestamp'] = pd.to_datetime(player_performance['timestamp'])

        # Get unique players, maps, and latencies
        all_players = pd.concat([player_performance['killer_ip'], player_performance['victim_ip']]).unique()
        
        # Update non-affected players list
        non_affected = [p for p in all_players if p not in self.affected_players]
        
        # Update and save player categories with complete list
        categories = {
            "affected": self.affected_players,
            "non_affected": non_affected
        }
        with open(f'{self.output_folder}/player_categories.json', 'w') as f:
            json.dump(categories, f, indent=4)

        all_maps = player_performance['map'].unique()
        all_latencies = sorted(player_performance['latency'].unique())

        # Process kills data
        print("\nProcessing kills data...")
        kills_data = self._process_event_type(player_performance, all_players, all_maps, all_latencies, "Kills")
        
        # Process deaths data
        print("\nProcessing deaths data...")
        deaths_data = self._process_event_type(player_performance, all_players, all_maps, all_latencies, "Deaths")

        # Save processed data
        print("\nSaving processed data...")
        self._save_processed_data(kills_data, deaths_data)

        print("Data processing complete!")

    def _process_event_type(self, player_performance, all_players, all_maps, all_latencies, event_type):
        """Process data for either kills or deaths."""
        all_data = []
        player_role = 'killer' if event_type == 'Kills' else 'victim'

        for player in tqdm(all_players, desc=f'Processing {event_type}'):
            try:
                activity_data = self._load_activity_data(player.split('_')[1])
                if activity_data is None:
                    continue

                for map_name in all_maps:
                    for latency in all_latencies:
                        # Get relevant events
                        events = player_performance[
                            (player_performance[f'{player_role}_ip'] == player) &
                            (player_performance['map'] == map_name) &
                            (player_performance['latency'] == latency)
                        ].sort_values('timestamp')

                        if len(events) == 0:
                            continue

                        # Calculate input frequencies
                        input_freqs = self._calculate_input_frequencies(events, activity_data)
                        
                        # Add to results
                        for input_col in self.input_columns:
                            if input_freqs[input_col]:  # Only add if we have data
                                for freq in input_freqs[input_col]:
                                    all_data.append({
                                        'player': player,
                                        'map': map_name,
                                        'latency': latency,
                                        'input_type': input_col,
                                        'frequency': freq,
                                        'timestamp': events.iloc[0]['timestamp']
                                    })

            except Exception as e:
                print(f"Error processing {player}: {str(e)}")
                continue

        return pd.DataFrame(all_data)

    def _load_activity_data(self, ip_address):
        """Load activity data for a player."""
        try:
            activity_data = pd.read_csv(f'{self.activity_folder}/{ip_address}_activity_data.csv')
            activity_data['timestamp'] = activity_data['timestamp'].apply(self._format_timestamp)
            activity_data['timestamp'] = pd.to_datetime(activity_data['timestamp'])
            return activity_data
        except Exception as e:
            print(f"Error loading activity data for IP {ip_address}: {str(e)}")
            return None

    def _calculate_input_frequencies(self, events, activity_data):
        """Calculate input frequencies around events."""
        input_frequencies = {col: [] for col in self.input_columns}
        
        for _, event in events.iterrows():
            event_time = event['timestamp']
            start_time = event_time + timedelta(seconds=self.time_range[0])
            end_time = event_time + timedelta(seconds=self.time_range[1])
            
            start_data = activity_data[activity_data['timestamp'] <= start_time].iloc[-1] if not activity_data[activity_data['timestamp'] <= start_time].empty else pd.Series({col: 0 for col in self.input_columns})
            end_data = activity_data[activity_data['timestamp'] <= end_time].iloc[-1] if not activity_data[activity_data['timestamp'] <= end_time].empty else pd.Series({col: 0 for col in self.input_columns})
            
            for col in self.input_columns:
                input_frequencies[col].append(end_data[col] - start_data[col])

        return input_frequencies

    def _save_processed_data(self, kills_data, deaths_data):
        """Save processed data to CSV files."""
        os.makedirs(self.output_folder, exist_ok=True)
        
        kills_data.to_csv(f'{self.output_folder}/processed_kills_data.csv', index=False)
        deaths_data.to_csv(f'{self.output_folder}/processed_deaths_data.csv', index=False)

if __name__ == "__main__":
    from config import PROCESSED_DATA_FOLDER, ACTIVITY_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER
    
    processor = DataProcessor(PROCESSED_DATA_FOLDER, ACTIVITY_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER)
    processor.process_all_data()