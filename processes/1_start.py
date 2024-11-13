import re
from datetime import datetime
import subprocess
import time
import os
import glob
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER, ACTIVITY_FOLDER

if not os.path.exists(LOG_FOLDER):
   os.makedirs(LOG_FOLDER)
if not os.path.exists(PROCESSED_DATA_FOLDER):
   os.makedirs(PROCESSED_DATA_FOLDER)
if not os.path.exists(RAW_DATA_FOLDER):
   os.makedirs(RAW_DATA_FOLDER)
if not os.path.exists(ACTIVITY_FOLDER):
   os.makedirs(ACTIVITY_FOLDER)
   
def format_datetime(timestamp):
    datetime_obj = datetime.fromtimestamp(float(timestamp))
    date_str = datetime_obj.strftime('%Y-%m-%d')
    time_str = datetime_obj.strftime('%H:%M:%S')
    return date_str, time_str

# Added code to ignore the beginning messy log without latency control
def find_clean_game_start(lines):
    for i, line in enumerate(lines):
        if '\\x08------------ Map Loading ------------' in line:
            # Look for Network egress latency within next 20 lines
            for j in range(i, min(i + 20, len(lines))):
                if 'Network egress latency:' in lines[j]:
                    return i
    return 0

# Find the input file
import_dir = RAW_DATA_FOLDER
log_files = glob.glob(os.path.join(import_dir, '*.log'))

if not log_files:
    raise FileNotFoundError("No .log file found in the import directory.")

if len(log_files) > 1:
    print("Warning: Multiple .log files found. Using the first one.")

input_path = log_files[0]
output_path = f'{LOG_FOLDER}/start.log'

# Read the input file
with open(input_path, 'r') as file:
    lines = file.readlines()

# Find the start of the clean game log
start_index = find_clean_game_start(lines)

# Process the file and write to the output file focusing on kill events
with open(output_path, 'w') as output_file:
    for line in lines[start_index:]:  # Start from clean game log
        if '\\x08Kill' in line or '\\x08PlayerScore' in line:
            # Extract timestamp and format it
            parts = line.split(': ', 1)
            timestamp = parts[0]
            date_str, time_str = format_datetime(timestamp)
            formatted_line = f"{date_str} {time_str}: {parts[1].strip()}"
            output_file.write(formatted_line + '\n')
        elif 'Network egress latency:' in line or '\\x08loaded maps' in line:
            # Extract timestamp and format it
            parts = line.split(': ', 1)
            timestamp = parts[0]
            date_str, time_str = format_datetime(timestamp)
            formatted_line = f"{date_str} {time_str}: {parts[1].strip()}"
            output_file.write(formatted_line + '\n')