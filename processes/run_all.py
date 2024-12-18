import subprocess

scripts = [
    "processes/1_start.py",
    "processes/2_separate.py",
    "processes/3_merge.py",
    "processes/4_create_df.py",
    "processes/5_remove_break_rounds.py",
    "processes/6_no_blanks.py",
    "processes/7_player_performance_per_round.py",
    "processes/8_round_score_summary.py",
    "processes/9_ignore_suicides.py",
    "processes/10_player_performance_per_round_adjusted.py",
    "processes/11_round_score_summary_after_adjusted.py",
    "processes/12_additional_counters.py",
    "processes/13_additional_counters_round_summary.py",
    "processes/14_kills_and_deaths_events.py",
    "processes/15_weapons.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script}: {result.stderr}")
    else:
        print(f"{script} completed successfully.")
    print(result.stdout)