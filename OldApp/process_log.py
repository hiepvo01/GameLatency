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

def process_events(events):
    games = []
    current_game = create_new_game(events[0][0])

    kill_pattern = re.compile(r'Kill: (\d+) (\d+) (\d+): (.+) killed (.+) by (\w+)')
    item_pattern = re.compile(r'Item: (\d+) (\w+)')
    score_pattern = re.compile(r'PlayerScore: (\d+) (\d+):')
    player_name_pattern = re.compile(r'ClientUserinfoChanged: (\d+) n\\(.+?)\\t')
    init_game_pattern = re.compile(r'InitGame:')
    shutdown_game_pattern = re.compile(r'ShutdownGame:')
    challenge_pattern = re.compile(r'Challenge: (\d+) (\d+) (\d+):')
    award_pattern = re.compile(r'Award: (\d+) (\d+): (.+) gained the (\w+) award!')
    map_pattern = re.compile(r'\\mapname\\(\w+)')

    for timestamp, line in events:
        init_game_match = init_game_pattern.search(line)
        if init_game_match:
            if current_game['events']:
                current_game['end_time'] = timestamp
                games.append(current_game)
            current_game = create_new_game(timestamp)
            map_match = map_pattern.search(line)
            if map_match:
                current_game['map'] = map_match.group(1)

        shutdown_game_match = shutdown_game_pattern.search(line)
        if shutdown_game_match:
            current_game['end_time'] = timestamp
            games.append(current_game)
            current_game = create_new_game(None)
            continue

        current_game['events'].append((timestamp, line))

        kill_match = kill_pattern.search(line)
        if kill_match:
            process_kill_event(current_game, kill_match)

        item_match = item_pattern.search(line)
        if item_match:
            process_item_event(current_game, item_match)

        score_match = score_pattern.search(line)
        if score_match:
            process_score_event(current_game, score_match)

        name_match = player_name_pattern.search(line)
        if name_match:
            process_name_event(current_game, name_match)

        challenge_match = challenge_pattern.search(line)
        if challenge_match:
            process_challenge_event(current_game, challenge_match)

        award_match = award_pattern.search(line)
        if award_match:
            process_award_event(current_game, award_match)

    if current_game['events']:
        current_game['end_time'] = events[-1][0]
        games.append(current_game)

    return games

def create_new_game(start_time):
    return {
        'start_time': start_time,
        'end_time': None,
        'map': "Unknown",
        'player_data': defaultdict(lambda: {
            'kills': 0, 'deaths': 0, 'suicides': 0, 'score': 0,
            'name': 'Unknown', 'items_picked': 0, 'weapons_picked': 0,
            'awards': defaultdict(int)
        }),
        'events': []
    }

def process_kill_event(game, match):
    killer, victim, _, _, _, weapon = match.groups()
    killer, victim = int(killer), int(victim)
    if killer != 1022:  # 1022 is world kills
        game['player_data'][killer]['kills'] += 1
    game['player_data'][victim]['deaths'] += 1
    if killer == victim:
        game['player_data'][victim]['suicides'] += 1

def process_item_event(game, match):
    player, item = match.groups()
    player = int(player)
    if item.startswith('weapon_'):
        game['player_data'][player]['weapons_picked'] += 1
    else:
        game['player_data'][player]['items_picked'] += 1

def process_score_event(game, match):
    player, score = map(int, match.groups())
    game['player_data'][player]['score'] = score

def process_name_event(game, match):
    player, name = match.groups()
    game['player_data'][int(player)]['name'] = name

def process_challenge_event(game, match):
    player, award_id, _ = map(int, match.groups())
    game['player_data'][player]['awards'][award_id] += 1

def process_award_event(game, match):
    player, _, _, award_name = match.groups()
    game['player_data'][int(player)]['awards'][award_name] += 1

def write_to_csv(games, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['game_id', 'timestamp', 'event', 'map', 'player_id', 'player_name', 
                      'kills', 'deaths', 'suicides', 'score', 'items_picked', 'weapons_picked']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for game_id, game in enumerate(games, 1):
            for timestamp, event in game['events']:
                for player_id, data in game['player_data'].items():
                    row = {
                        'game_id': game_id,
                        'timestamp': timestamp,
                        'event': event.strip(),
                        'map': game['map'],
                        'player_id': player_id,
                        'player_name': data['name'],
                        'kills': data['kills'],
                        'deaths': data['deaths'],
                        'suicides': data['suicides'],
                        'score': data['score'],
                        'items_picked': data['items_picked'],
                        'weapons_picked': data['weapons_picked']
                    }
                    writer.writerow(row)

# Main execution
with open('quake3_server.log', 'r') as file:
    log_content = file.read()

events = parse_quake3_log(log_content)
games = process_events(events)
write_to_csv(games, 'quake3_game_data.csv')

print(f"CSV file 'quake3_game_data.csv' has been created.")
print(f"Total number of games: {len(games)}")

for game_id, game in enumerate(games, 1):
    print(f"\nGame {game_id}:")
    print(f"  Map: {game['map']}")
    if game['start_time'] is not None and game['end_time'] is not None:
        duration = game['end_time'] - game['start_time']
        print(f"  Duration: {duration:.2f} seconds")
    else:
        print("  Duration: Unknown (start or end time missing)")
    print(f"  Players:")
    for player, data in game['player_data'].items():
        print(f"    Player {player} ({data['name']}):")
        print(f"      Kills: {data['kills']}, Deaths: {data['deaths']}, Suicides: {data['suicides']}")
        print(f"      Score: {data['score']}")
        print(f"      Items picked: {data['items_picked']}, Weapons picked: {data['weapons_picked']}")
        print(f"      Awards: {dict(data['awards'])}")
    print(f"  Total events: {len(game['events'])}")