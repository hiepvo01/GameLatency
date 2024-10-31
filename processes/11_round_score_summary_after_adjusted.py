import pandas as pd
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER

# Create summary dataframes for each round

input_path = f'{PROCESSED_DATA_FOLDER}/ignore_suicides.csv' ##path
output_path = f'{PROCESSED_DATA_FOLDER}/round_summary_adjusted.csv' ##path

# Read the CSV file
df = pd.read_csv(input_path)

# Dynamically create player_ip_map
unique_players = df['player_ip'].unique()
player_ip_map = {i+1: player_ip for i, player_ip in enumerate(sorted(unique_players))}

# Unique player IDs to include in the summary
player_ids = list(range(1, len(player_ip_map) + 1))

# Initialize a list to store the summary data
summary_data = []

# Group by game_round and iterate over each group
for game_round, group in df.groupby('game_round'):
    current_map = group['map'].iloc[0] if pd.notna(group['map'].iloc[0]) else ''
    current_latency = group['latency'].iloc[0] if pd.notna(group['latency'].iloc[0]) else ''
    
    for player_id in player_ids:
        player_ip = player_ip_map[player_id]
        player_data = group[group['player_ip'] == player_ip]
        if not player_data.empty:
            last_record = player_data.iloc[-1]
            score = last_record['score'] if pd.notna(last_record['score']) else 0
        else:
            score = 0
        
        summary_data.append({
            'game_round': game_round,
            'map': current_map,
            'latency': current_latency,
            'player_id': player_id,
            'player_ip': player_ip,
            'score': score
        })

# Create a summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save the summary DataFrame to a new CSV file
summary_df.to_csv(output_path, index=False)

print(f"Round score summary saved to {output_path}")

# Print the player_ip_map for reference
print("\nPlayer ID to IP mapping:")
for player_id, ip in player_ip_map.items():
    print(f"Player {player_id}: {ip}")