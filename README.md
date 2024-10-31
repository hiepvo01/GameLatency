# About

This repo processes a log file to extract and organise relevant data, ensuring that all player activities are recorded. The output is a CSV file that includes timestamps, game rounds, latencies, maps, killers, victims, weapons, and points for each player.

## Directory Walkthrough

- **App** | Main app and uploaded files

- **Assets** | Images and logos

- **Data** | Data cleansing processed files

- **Processes** | Data cleansing processes

- **Sidebar** | Main content holder

# How to run the app

** First upload the log file

There are 2 ways to run the app:

1. Deploy zeta.py into a website hosted by Streamlit.

2. On command line, run ```streamlit run app/zeta.py```

- Make sure you have Streamlit installed in your environment. If not, you can install it

- Make sure you install environment packages using ```pip install -r requirements.txt```

# Other notes

- Take care in modifying pathnames and python modules used within directories.