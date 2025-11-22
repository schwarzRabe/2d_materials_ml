"""
Configuration file for 2D Materials ML project
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'output'
PLOTS_DIR = OUTPUT_DIR / 'plots'
LOGS_DIR = PROJECT_ROOT / 'logs'
MODELS_DIR = PROJECT_ROOT / 'models'

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  OUTPUT_DIR, PLOTS_DIR, LOGS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration dictionary
CONFIG = {
    # Data sources
    'data_sources': ['jarvis', 'c2db'],  # Available: 'jarvis', 'c2db', 'mp'
    
    # Data filtering
    'stability_threshold': 0.1,  # eV/atom above hull
    'band_gap_min': 0.0,         # Minimum band gap (eV)
    'band_gap_max': 10.0,        # Maximum band gap (eV)
    
    # File paths
    'raw_data_file': str(RAW_DATA_DIR / 'all_materials_raw.csv'),
    'processed_data_file': str(PROCESSED_DATA_DIR / 'materials_processed.csv'),
    'ml_data_file': str(PROCESSED_DATA_DIR / 'ml_ready_data.pkl'),
    
    # Visualization
    'plot_style': 'seaborn-v0_8-darkgrid',
    'figure_dpi': 100,
    
    # Logging
    'log_file': str(LOGS_DIR / 'download.log'),
    'verbose': True,
}

# API Keys (set as environment variables)
API_KEYS = {
    'materials_project': os.environ.get('MP_API_KEY', ''),
    'citrine': os.environ.get('CITRINE_API_KEY', ''),
}

# Material properties to extract
MATERIAL_PROPERTIES = [
    'band_gap',
    'band_gap_direct',
    'formation_energy',
    'energy_above_hull',
    'thickness',
    'exfoliation_energy',
    'magnetic_moment',
    'fermi_energy',
    'elastic_modulus',
    'poisson_ratio'
]
