# PDD9 - Game Latency Impact Study

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gamelatency.streamlit.app/)

## Overview
This repository contains a data processing pipeline and visualization dashboard for analyzing game log files as part of the PDD9 Game Latency Impact Study. The system extracts player activities and generates structured data including timestamps, game rounds, latencies, maps, player interactions (kills/deaths), weapons used, and scoring information to understand the impact of network latency on player performance.

## Live Demo
Access the interactive dashboard at: [https://gamelatency.streamlit.app/](https://gamelatency.streamlit.app/)

## Features
- Automated log file processing and data extraction
- CSV output generation with detailed player statistics
- Interactive visualization dashboard built with Streamlit
- Support for multiple data processing pipelines
- Configurable file paths and processing parameters
- Real-time latency impact analysis
- Player performance metrics visualization

## Directory Structure
```
.
├── pages/              # Streamlit sidebar pages for visualization
├── images/             # Static assets (logos, images)
├── data/              
│   ├── activity_data/  # Player activity CSV files
│   └── raw_log/       # Raw log files and player lists
├── processes/          # Data processing scripts
├── processed/          # Output directory for cleaned data
└── utils/             # Helper functions and utilities
```

## Prerequisites
- Python 3.8 or higher
- Virtual environment management tool (venv, conda)
- Access to raw data files (contact UTS TRU lab for download instructions)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

1. Create required data directories:
```bash
mkdir -p data/activity_data data/raw_log
```

2. Add required input files:
   - Place participant `*_activity_data.csv` files in `data/activity_data/`
   - Add raw log file and `effected_players.txt` to `data/raw_log/`

## Usage

1. Process the raw data:
```bash
python processes/run_all.py
```
This will:
- Execute all data processing scripts
- Generate cleaned data in the `processed/` directory
- Overwrite any existing processed data

2. Launch the visualization dashboard locally:
```bash
streamlit run app.py
```

## Configuration
- Environment variables are managed through `.env` file
- Processing parameters can be modified in `config.py`
- File paths and other settings can be customized as needed

## Contributing

### Adding New Processing Scripts
1. Add your processing script to the `processes/` directory
2. Update `run_all.py` to include your script in the processing pipeline
3. Ensure your script follows the project's input/output conventions

### Adding New Visualizations
1. Create core visualization functions in `utils.py`
2. Add new Streamlit pages in `pages/visualizations/`
3. Import visualization functions from `utils.py` to maintain consistency
4. Follow existing templates and coding standards

## Troubleshooting
- Ensure all required directories exist before running the pipeline
- Check file permissions if experiencing access issues
- Verify input file formats match expected schemas
- Consult logs in case of processing errors

## Support
For access to raw data files or technical support, please contact the team at UTS TRU lab.

## Deployment
The application is deployed using Streamlit Cloud and can be accessed at [https://gamelatency.streamlit.app/](https://gamelatency.streamlit.app/)