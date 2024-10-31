import re
from collections import defaultdict
import csv
from datetime import datetime

def parse_quake3_log(log_content):
    timestamp_pattern = re.compile(r'^(\d+\.\d+):')
    events = []

    for line in log_content.split('\n'):
        timestamp_match = timestamp_pattern.match(line)
        if timestamp_match:
            timestamp = float(timestamp_match.group(1))
            events.append((timestamp, line))

    return sorted(events, key=lambda x: x[0])

def process_events(events, interval=1.0):
    games = []
    current_game = {
        'start_time': events[0][0],
        'end_time': None,
        'map': "Unknown",
        'player_data': defaultdict(lambda: {
            'kills': 0,
            'deaths': 0,
            'suicides': 0,
            'items_picked': 0,
            'weapons_picked': 0,
            'score': 0,
            'name': 'Unknown',
            'latency': None,
            'weapon_events': defaultdict(int),
            'last_weapon_event': None,
            'items_inventory': defaultdict(int),
            'awards': defaultdict(int)
        }),
        'time_series': []
    }

    kill_pattern = re.compile(r'Kill: (\d+) (\d+) (\d+): (.+) killed (.+) by (\w+)')
    item_pattern = re.compile(r'Item: (\d+) (\w+)')
    score_pattern = re.compile(r'PlayerScore: (\d+) (\d+):')
    player_name_pattern = re.compile(r'ClientUserinfoChanged: (\d+) n\\(.+?)\\t')
    init_game_pattern = re.compile(r'InitGame:')
    map_loading_pattern = re.compile(r'loaded maps/(\w+)\.aas')
    challenge_pattern = re.compile(r'Challenge: (\d+) (\d+) (\d+):')
    award_pattern = re.compile(r'Award: (\d+) (\d+): (.+) gained the (\w+) award!')
    shutdown_game_pattern = re.compile(r'ShutdownGame:')
    latency_pattern = re.compile(r'Network egress latency: (\d+) ms')

    map_info = {
        'kaos2': 'break round, enclosed arena',
        'aggressor': 'indoor map, players can die from lava (MOD_LAVA)',
        'wrackdm17': 'outdoor map, players can fall to death (MOD_FALLING)'
    }

    award_mapping = {
        '202': 'IMPRESSIVE',
        '203': 'EXCELLENT',
        '204': 'GAUNTLET',
        '205': 'FRAGS',
        '206': 'PERFECT',
        '207': 'CAPTURE',
        '208': 'ACCURACY'
    }

    current_time = events[0][0]
    end_time = events[-1][0]
    event_index = 0
    current_latency = None

    while current_time <= end_time:
        events_this_second = []
        while event_index < len(events) and events[event_index][0] < current_time + interval:
            timestamp, line = events[event_index]
            events_this_second.append((timestamp, line.strip()))
            
            kill_match = kill_pattern.search(line)
            if kill_match:
                killer, victim, _, killer_name, victim_name, weapon = kill_match.groups()
                killer, victim = int(killer), int(victim)
                if killer != 1022:  # 1022 is world kills
                    current_game['player_data'][killer]['kills'] += 1
                    current_game['player_data'][killer]['weapon_events'][weapon] += 1
                    current_game['player_data'][killer]['last_weapon_event'] = weapon
                current_game['player_data'][victim]['deaths'] += 1
                current_game['player_data'][victim]['items_inventory'] = defaultdict(int)  # Reset items on death
                if killer == victim:
                    current_game['player_data'][victim]['suicides'] += 1

            item_match = item_pattern.search(line)
            if item_match:
                player, item = item_match.groups()
                player = int(player)
                if item.startswith('weapon_'):
                    current_game['player_data'][player]['weapons_picked'] += 1
                    current_game['player_data'][player]['weapon_events'][item] += 1
                    current_game['player_data'][player]['last_weapon_event'] = item
                else:
                    current_game['player_data'][player]['items_picked'] += 1
                    current_game['player_data'][player]['items_inventory'][item] += 1

            score_match = score_pattern.search(line)
            if score_match:
                player, score = map(int, score_match.groups())
                current_game['player_data'][player]['score'] = score

            name_match = player_name_pattern.search(line)
            if name_match:
                player, name = name_match.groups()
                player = int(player)
                current_game['player_data'][player]['name'] = name

            challenge_match = challenge_pattern.search(line)
            if challenge_match:
                player, award_id, _ = map(int, challenge_match.groups())
                current_game['player_data'][player]['awards'][award_id] += 1

            award_match = award_pattern.search(line)
            if award_match:
                player, _, _, award_name = award_match.groups()
                player = int(player)
                current_game['player_data'][player]['awards'][award_name] += 1

            latency_match = latency_pattern.search(line)
            if latency_match:
                current_latency = int(latency_match.group(1))

            map_loading_match = map_loading_pattern.search(line)
            if map_loading_match:
                current_game['map'] = map_loading_match.group(1)

            init_game_match = init_game_pattern.search(line)
            if init_game_match:
                if current_game['time_series']:
                    current_game['end_time'] = timestamp
                    games.append(current_game)
                current_game = {
                    'start_time': timestamp,
                    'end_time': None,
                    'map': "Unknown",
                    'player_data': defaultdict(lambda: {
                        'kills': 0,
                        'deaths': 0,
                        'suicides': 0,
                        'items_picked': 0,
                        'weapons_picked': 0,
                        'score': 0,
                        'name': 'Unknown',
                        'latency': None,
                        'weapon_events': defaultdict(int),
                        'last_weapon_event': None,
                        'items_inventory': defaultdict(int),
                        'awards': defaultdict(int)
                    }),
                    'time_series': []
                }

            shutdown_game_match = shutdown_game_pattern.search(line)
            if shutdown_game_match:
                current_game['end_time'] = timestamp
                games.append(current_game)
                current_game = {
                    'start_time': None,
                    'end_time': None,
                    'map': "Unknown",
                    'player_data': defaultdict(lambda: {
                        'kills': 0,
                        'deaths': 0,
                        'suicides': 0,
                        'items_picked': 0,
                        'weapons_picked': 0,
                        'score': 0,
                        'name': 'Unknown',
                        'latency': None,
                        'weapon_events': defaultdict(int),
                        'last_weapon_event': None,
                        'items_inventory': defaultdict(int),
                        'awards': defaultdict(int)
                    }),
                    'time_series': []
                }

            event_index += 1

        # Update latency for all players
        for player_data in current_game['player_data'].values():
            player_data['latency'] = current_latency

        # Store the state and all events for this second
        current_game['time_series'].append((current_time, events_this_second, {player: data.copy() for player, data in current_game['player_data'].items()}))
        current_time += interval

    # Ensure the last game has an end time
    if current_game['time_series'] and current_game['end_time'] is None:
        current_game['end_time'] = events[-1][0]
        games.append(current_game)

    return games, map_info, award_mapping

def write_to_csv(games, filename, award_mapping):
    # Collect all unique item types
    all_items = set()
    for game in games:
        for _, _, state in game['time_series']:
            for player_data in state.values():
                all_items.update(player_data['items_inventory'].keys())
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['game_id', 'timestamp', 'exact_timestamp', 'event', 'map', 'player', 'name', 'kills', 'deaths', 'suicides', 'items_picked', 'weapons_picked', 'score', 'latency', 'last_weapon_event', 'most_frequent_weapon_event', 'awards'] + list(all_items)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for game_id, game in enumerate(games, 1):
            for timestamp, events_this_second, state in game['time_series']:
                if not events_this_second:
                    # If no events occurred this second, write one row with the current state
                    for player, data in state.items():
                        row = create_row(game_id, timestamp, timestamp, "No event", game['map'], player, data, state, award_mapping, all_items)
                        writer.writerow(row)
                else:
                    # If events occurred, write a row for each event
                    for exact_timestamp, event in events_this_second:
                        for player, data in state.items():
                            row = create_row(game_id, timestamp, exact_timestamp, event, game['map'], player, data, state, award_mapping, all_items)
                            writer.writerow(row)

def create_row(game_id, timestamp, exact_timestamp, event, map_name, player, data, state, award_mapping, all_items):
    if data['weapon_events']:
        most_frequent_weapon_event = max(data['weapon_events'], key=data['weapon_events'].get)
        last_weapon_event = data['last_weapon_event']
    else:
        most_frequent_weapon_event = 'None'
        last_weapon_event = 'None'
    
    awards_list = []
    for award_id, count in data['awards'].items():
        award_name = award_mapping.get(str(award_id), f"Unknown Award {award_id}")
        awards_list.append(f"{award_name}: {count}")
    awards_str = ', '.join(awards_list) if awards_list else 'None'
    
    row = {
        'game_id': game_id,
        'timestamp': f"{timestamp:.1f}",
        'exact_timestamp': f"{exact_timestamp:.6f}",
        'event': event,
        'map': map_name,
        'player': player,
        'name': data['name'],
        'kills': data['kills'],
        'deaths': data['deaths'],
        'suicides': data['suicides'],
        'items_picked': data['items_picked'],
        'weapons_picked': data['weapons_picked'],
        'score': data['score'],
        'latency': data['latency'] if data['latency'] is not None else 'Unknown',
        'last_weapon_event': last_weapon_event,
        'most_frequent_weapon_event': most_frequent_weapon_event,
        'awards': awards_str
    }
    # Add item counts
    for item in all_items:
        row[item] = data['items_inventory'].get(item, 0)
    return row

# Read the log file
with open('quake3_server.log', 'r') as file:
    log_content = file.read()

# Parse events
events = parse_quake3_log(log_content)

# Process events and get game data
games, map_info, award_mapping = process_events(events)

# Write time series data to CSV
write_to_csv(games, 'quake3_detailed_game_data.csv', award_mapping)

# Print summary
print(f"CSV file 'quake3_detailed_game_data.csv' has been created.")
print(f"Total number of games: {len(games)}")

# Print map information
print("\nMap Information:")
for map_name, description in map_info.items():
    print(f"  {map_name}: {description}")

# Print summary for each game
for game_id, game in enumerate(games, 1):
    print(f"\nGame {game_id}:")
    print(f"  Map: {game['map']}")
    if game['end_time'] is not None and game['start_time'] is not None:
        print(f"  Duration: {game['end_time'] - game['start_time']:.2f} seconds")
    else:
        print("  Duration: Unknown (game may not have ended properly)")
    print(f"  Players:")
    for player, data in game['player_data'].items():
        print(f"    Player {player} ({data['name']}):")
        print(f"      Kills: {data['kills']}, Deaths: {data['deaths']}, Suicides: {data['suicides']}")
        print(f"      Score: {data['score']}")
        print(f"      Items collected: {dict(data['items_inventory'])}")
        print(f"      Most frequent weapon event: {max(data['weapon_events'], key=data['weapon_events'].get) if data['weapon_events'] else 'None'}")
        print(f"      Awards: {', '.join(f'{k}: {v}' for k, v in data['awards'].items())}")
    print(f"  Time series entries: {len(game['time_series'])}")

print("\nAward Explanations:")
for award_id, award_name in award_mapping.items():
    print(f"  {award_id}: {award_name}")

print("\nNote on timestamps and events:")
print("  The 'timestamp' column shows the artificial second-by-second timestamps.")
print("  The 'exact_timestamp' column shows the precise timestamp from the log file.")
print("  The 'event' column contains the exact log entry for each event.")
print("  If multiple events occur within the same second, each event will have its own row")
print("  with the same 'timestamp' but different 'exact_timestamp' and 'event' values.")
print("  Rows with 'No event' in the 'event' column represent the state at that second")
print("  when no new events occurred.")

print("\nNote on weapon events:")
print("  'last_weapon_event' shows the most recent weapon involved in a kill or pickup.")
print("  'most_frequent_weapon_event' shows the weapon most frequently involved in kills or pickups.")
print("  These do not necessarily represent the player's current weapon or all weapon usage.")
print("  They are based on recorded events (kills and pickups) in the log.")

print("\nNote on latency:")
print("  Latency is reported as 'Network egress latency' in milliseconds.")
print("  This value represents the server-side network latency.")
print("  The latency value is updated whenever a new latency measurement is encountered in the log.")
print("  All players will have the same latency value at any given time, as it's a server-wide measurement.")
print("  If no latency information is available for a particular time point, it will be reported as 'Unknown'.")

print("\nNote on item columns:")
print("  Each unique item type encountered in the log will have its own column in the CSV output.")
print("  The value in each item column represents the number of that item type the player currently has.")
print("  Item counts are reset to 0 when a player dies.")
print("  Weapon pickups are not included in these item counts, but are reflected in the 'weapons_picked' column.")

print("\nCSV column explanations:")
print("  game_id: Unique identifier for each game session")
print("  timestamp: Artificial second-by-second time of the event")
print("  exact_timestamp: Precise time of the event from the log file")
print("  event: Exact log entry for the event")
print("  map: Current map being played")
print("  player: Player identifier")
print("  name: Player's name")
print("  kills: Number of kills by the player")
print("  deaths: Number of times the player has died")
print("  suicides: Number of times the player has killed themselves")
print("  items_picked: Total number of non-weapon items picked up")
print("  weapons_picked: Total number of weapons picked up")
print("  score: Player's current score")
print("  latency: Server's network egress latency")
print("  last_weapon_event: Most recent weapon involved in a kill or pickup")
print("  most_frequent_weapon_event: Weapon most frequently involved in kills or pickups")
print("  awards: List of awards received by the player")
print("  [item_name]: Count of specific items held by the player")

print("\nNote on data interpretation:")
print("  This analysis is based on available log data and may not capture all aspects of gameplay.")
print("  Weapon events are limited to kills and pickups, and don't reflect all weapon usage or switches.")
print("  Item counts represent the current inventory and are reset upon player death.")
print("  Latency is a server-wide measurement and may not reflect individual player experiences.")
print("  For the most accurate interpretation, consider these limitations when analyzing the data.")

print("\nNote on events:")
print("  The 'event' column now contains the exact log entry for each event.")
print("  This preserves the original data and allows for detailed analysis of specific events.")
print("  Events are logged at their exact timestamp, allowing for precise tracking of game occurrences.")