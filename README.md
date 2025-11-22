# 2D Materials ML: Band Gap & Formation Energy Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine Learning models for predicting band gap and formation energy of 2D materials using data from multiple computational databases.

## 🎯 Features

- **Automated data collection** from multiple 2D materials databases
- **Data preprocessing** and standardization pipeline  
- **ML-ready dataset** preparation
- **Visualization tools** for data exploration
- **Support for multiple data sources**: JARVIS-2D, C2DB, Materials Project

## 📊 Data Sources

| Database | Materials | Properties | Access |
|----------|-----------|------------|---------|
| JARVIS-2D | ~2,500 | Band gap, Formation energy, Stability | Free |
| C2DB | ~4,000 | Electronic, Mechanical, Magnetic | Free |
| Materials Project | ~1,000 | Band gap, Formation energy | API Key |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/2d-materials-ml.git
cd 2d-materials-ml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Run complete pipeline
python main.py

# Or run individual steps:
python -m src.download_data  # Download data
python -m src.check_data     # Validate and visualize
```

### Use the Data

```python
import pickle
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/materials_processed.csv')

# Load ML-ready data
with open('data/processed/ml_ready_data.pkl', 'rb') as f:
    ml_data = pickle.load(f)

X = ml_data['features']
y_bandgap = ml_data['band_gap']
y_formation = ml_data['formation_energy']
```

## 📁 Project Structure

```
2d-materials-ml/
├── src/                    # Source code
│   ├── download_data.py   # Data collection
│   ├── check_data.py       # Validation & visualization
│   └── config.py          # Configuration
├── data/                   # Data directory
│   ├── raw/               # Raw downloaded data
│   └── processed/         # Processed ML-ready data
├── notebooks/             # Jupyter notebooks
├── output/                # Results and plots
└── docs/                  # Documentation
```

## 📈 Dataset Statistics

After processing, the dataset typically contains:

- **Total materials**: ~2,000-3,000
- **Features**: Band gap, formation energy, stability metrics
- **Complete samples**: Materials with all required properties

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
CONFIG = {
    'data_sources': ['jarvis', 'c2db'],  # Data sources to use
    'stability_threshold': 0.1,          # eV/atom above hull
    'band_gap_range': (0, 10),          # Valid band gap range
}
```

## 📊 Visualization

The pipeline generates automatic visualizations in `output/plots/` directory.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{2d_materials_ml,
  title = {2D Materials ML: Band Gap and Formation Energy Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/2d-materials-ml}
}
```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [JARVIS](https://jarvis.nist.gov/) - NIST database
- [C2DB](https://c2db.fysik.dtu.dk/) - DTU database
- [Materials Project](https://materialsproject.org/) - Berkeley database

## 📧 Contact

- GitHub Issues: [Create an issue](https://github.com/yourusername/2d-materials-ml/issues)

---
⭐ Star this repository if you find it helpful!
