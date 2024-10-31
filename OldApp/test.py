import re

def parse_quake3_log(log_content):
    # Regular expressions for game start and end
    start_pattern = re.compile(r'InitGame:')
    end_pattern = re.compile(r'ShutdownGame:')

    games = []
    current_game = []
    in_game = False

    for line in log_content.split('\n'):
        if start_pattern.search(line):
            in_game = True
            current_game = [line]
        elif end_pattern.search(line) and in_game:
            current_game.append(line)
            games.append('\n'.join(current_game))
            in_game = False
            current_game = []
        elif in_game:
            current_game.append(line)

    return games

# Read the log file
with open('quake3_server.log', 'r') as file:
    log_content = file.read()

# Parse the log file
games = parse_quake3_log(log_content)

# Print the number of games found
print(f"Number of games found: {len(games)}")

# Print the first game log (if any games were found)
if games:
    print("\nFirst game log:")
    print(games[2])
else:
    print("No games found in the log.")