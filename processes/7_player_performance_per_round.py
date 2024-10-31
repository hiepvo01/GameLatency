import pandas as pd
import os
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER

input_path = f'{PROCESSED_DATA_FOLDER}/no_blanks.csv'  # path to input CSV
output_dir = f'{PROCESSED_DATA_FOLDER}/player_performance_per_round'  # directory to store output files

# Read the CSV file
df = pd.read_csv(input_path)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create death events for each kill event
death_events = df[df['event'] == 'Kill'].copy()
death_events['event'] = 'Death'
death_events['player_ip'] = death_events['victim_ip']
death_events['player_id'] = death_events['victim_id']

# Combine kill and death events
combined_df = pd.concat([df, death_events], ignore_index=True)

# Sort the combined dataframe by timestamp
combined_df = combined_df.sort_values('timestamp')

# Group by game_round and player_ip and save each group to a separate CSV file
for (game_round, player_ip), group in combined_df.groupby(['game_round', 'player_ip']):
    output_path = os.path.join(output_dir, f'round_{game_round}_player_{player_ip}.csv')
    group.to_csv(output_path, index=False)
    # print(f'Saved round {game_round} for player {player_ip} to {output_path}')

print("DataFrames for each round and player_ip have been saved.")