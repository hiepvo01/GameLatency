import pandas as pd
import numpy as np
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER

input_path = f'{PROCESSED_DATA_FOLDER}/ignore_suicides.csv'  # path
output_path = f'{PROCESSED_DATA_FOLDER}/player_performance.csv'  # path

# Load the input CSV file
df = pd.read_csv(input_path)

# Drop the specified columns
drop = {
    'log_line', 'event', 'killer_id', 'victim_id', 'weapon_id',
    'weapon', 'score', 'points', 'player_id', 'log_score'
}
df.drop(columns=drop, inplace=True)

# Adding the new columns with initial values set to 0
df['suicide_count'] = 0
df['deaths_total'] = 0

# Function to safely convert to string and handle NaN values
def safe_str(x):
    return str(x) if pd.notnull(x) else np.nan

# Apply safe_str to killer_ip and victim_ip columns
df['killer_ip'] = df['killer_ip'].apply(safe_str)
df['victim_ip'] = df['victim_ip'].apply(safe_str)

# Dynamically identify unique players, excluding NaN values
players = sorted(set(pd.concat([df['killer_ip'], df['victim_ip']]).dropna().unique()))

for player in players:
    df[f'killed_by_{player}'] = 0
    df[f'killed_{player}'] = 0

# Function to calculate the counts for each round
def calculate(group):
    # Mark suicides
    group['is_suicide'] = group['killer_ip'] == group['victim_ip']
    
    # Calculate cumulative suicides
    group['suicide_count'] = group.groupby('player_ip')['is_suicide'].cumsum()
    
    # Check and update killed_Player counters
    for player in players:
        killed_mask = group['victim_ip'] == player
        group[f'killed_{player}'] = killed_mask.cumsum()

    return group

def calculate_killed_by(group):
    for player in players:
        killed_by_mask = group['killer_ip'] == player
        group[f'killed_by_{player}'] = killed_by_mask.cumsum()
        
    return group

def calculate_total_deaths(group):
    # Calculate deaths total
    group['deaths_total'] = group.groupby('victim_ip').cumcount() + 1
    return group

# Applying the function to each game round
df = df.groupby(['game_round', 'player_ip'], group_keys=False).apply(calculate)
df = df.groupby(['game_round', 'victim_ip'], group_keys=False).apply(calculate_killed_by)
df = df.groupby('game_round', group_keys=False).apply(calculate_total_deaths)

# Drop the 'is_suicide' column as it's no longer needed
df.drop(columns=['is_suicide'], inplace=True)

# Saving the updated and filtered dataframe to the output CSV file
df.to_csv(output_path, index=False)

print(f"Suicide counts, total deaths and deaths from each player by round have been recorded in {output_path}")

# Print the list of players for reference
print("\nPlayers identified in the data:")
for player in players:
    print(player)