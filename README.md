# Don't Swim in Data: Real-Time Microbial Forecasting for New Zealand Recreational Waters

<div align="center">

![Don't Swim in Data Logo](https://img.shields.io/badge/🌊-Don't%20Swim%20in%20Data-blue?style=for-the-badge)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/powered%20by-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED.svg)](https://www.docker.com/)

*Forecasting Enterococci levels in Canterbury's recreational waters through advanced machine learning*

</div>

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Docker Installation](#docker-installation-alternative)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training Models](#training-models)
  - [Running Predictions](#running-predictions)
  - [Launching the Dashboard](#launching-the-dashboard)
- [Experimental Results](#experimental-results)
- [Key Features](#key-features)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

Traditional water quality monitoring, characterized by sparse sampling and multi-day lab delays, often overlooks rapid contamination changes, posing risks to recreational water users. To address this gap, this repository implements two novel machine learning frameworks for real-time Enterococci forecasting in Canterbury, New Zealand:

1. **The Probabilistic Forecasting Framework** employs quantile regression (5th–98th percentiles), a gradient boosting meta-learner, and Conformalized Quantile Regression to provide precise single-point forecasts and calibrated 90% prediction intervals.

![image](https://github.com/user-attachments/assets/6e7617fe-d9fa-4c39-9f33-f6f79e8b09fb)

2. **The Matrix Decomposition Framework** leverages Non-negative Matrix Factorization to decompose complex spatio-temporal water quality data into lower-dimensional latent factors, modeled via multi-target Random Forests—enhancing generalization and addressing cold-start challenges at new coastal sites.

![image](https://github.com/user-attachments/assets/a2ae4daf-fd7c-43c8-9078-a16208ecff4a)

Evaluated on a dataset from 15 locations (2021–2024, 1047 samples, 100 exceedance events), our frameworks achieve:
- Exceedance sensitivity up to 67.0% (increasing to 75.7% in 2023–2024)
- Precautionary sensitivity up to 77.0%
- Specificity up to 92.3%
- WMAPE as low as 17.2% during exceedance events

Together, these frameworks exceed USGS guidelines, outperform traditional operational methods and standard ML benchmarks, while displaying highly competitive performance relative to state-of-the-art nowcasting systems such as Auckland's Safeswim.

## Project Structure

```
Don't Swim in Data/
│
├── data/                               # Data directory
│   ├── raw/                            # Raw data files
│   ├── processed/                      # Processed datasets
│   └── external/                       # External data sources
│
├── models/                             # Trained model files
│   ├── probabilistic/                  # Probabilistic Framework models
│   └── matrix_decomposition/           # Matrix Decomposition Framework models
│
├── notebooks/                          # Jupyter notebooks
│   ├── exploratory/                    # Data exploration
│   ├── model_development/              # Model training and tuning
│   └── analysis/                       # Results analysis
│
├── src/                                # Source code
│   ├── data/                           # Data processing scripts
│   │   ├── preprocessing.py            # Data preprocessing utilities
│   │   └── feature_engineering.py      # Feature engineering pipelines
│   │
│   ├── models/                         # Model implementations
│   │   ├── probabilistic/              # Probabilistic Framework
│   │   │   ├── quantile_models.py      # Quantile regression models
│   │   │   ├── meta_learner.py         # Meta-learner implementation
│   │   │   └── cqr.py                  # Conformalized Quantile Regression
│   │   │
│   │   └── matrix_decomposition/       # Matrix Decomposition Framework
│   │       ├── nmf.py                  # Non-negative Matrix Factorization
│   │       ├── time_model.py           # Time model implementation
│   │       └── space_model.py          # Space model implementation
│   │
│   ├── evaluation/                     # Evaluation utilities
│   │   ├── metrics.py                  # Evaluation metrics
│   │   └── cross_validation.py         # Cross-validation strategies
│   │
│   ├── visualization/                  # Visualization utilities
│   │   └── shap_analysis.py            # SHAP analysis tools
│   │
│   └── deployment/                     # Deployment utilities
│       ├── pipeline.py                 # Real-time data pipeline
│       └── drift_detection.py          # Drift detection implementation
│
├── app/                                # Web application
│   ├── dashboard.py                    # Streamlit dashboard
│   └── static/                         # Static assets
│
├── tests/                              # Unit tests
│
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── setup.py                            # Package installation script
└── README.md                           # Project documentation
```

## Installation

### Prerequisites

- Python 3.9+
- pip
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/asif-jc/Dont-Swim-in-Data.git
   cd Dont-Swim-in-Data
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation (Alternative)

For a containerized setup:

```bash
docker build -t microbial-forecasting .
docker run -p 8501:8501 microbial-forecasting
```

## Usage

### Data Preprocessing

To preprocess raw data and generate features:

```bash
python src/data/preprocessing.py --input data/raw --output data/processed
python src/data/feature_engineering.py --input data/processed --output data/processed
```

### Training Models

To train the Probabilistic Forecasting Framework:

```bash
python src/models/probabilistic/train.py --data data/processed/features.csv --output models/probabilistic
```

To train the Matrix Decomposition Framework:

```bash
python src/models/matrix_decomposition/train.py --data data/processed/features.csv --output models/matrix_decomposition
```

### Running Predictions

To generate forecasts using trained models:

```bash
python src/deployment/forecast.py --model models/probabilistic --input data/new_data.csv --output predictions.csv
```

### Launching the Dashboard

To launch the interactive Streamlit dashboard for real-time monitoring:

```bash
streamlit run app/dashboard.py
```

The dashboard will be accessible at http://localhost:8501.

## Experimental Results

Our frameworks achieve impressive performance on Enterococci concentration forecasting:

| Model | RMSE (MPN/100mL) | WMAPE Safe (%) | WMAPE Exc. (%) | Sensitivity (%) | Specificity (%) |
|-------|------------------|----------------|----------------|-----------------|-----------------|
| **Probabilistic Framework** | 872 | 52.3 | **17.2** | **67.0** | 92.3 |
| **Matrix Decomposition** | **870** | 54.2 | 20.3 | 61.0 | 90.6 |
| Virtual Beach (Baseline) | 934 | 48.2 | 42.4 | 38.0 | 92.9 |
| Linear Regression | 867 | 42.6 | 38.3 | 40.0 | **93.9** |
| MLP | 893 | 115.3 | 20.4 | 56.0 | 91.8 |

*Note: Best performance for each metric shown in bold.*

Both frameworks significantly outperform conventional methods, providing robust uncertainty quantification and enhanced interpretability through SHAP analysis. The Probabilistic Framework excels at exceedance sensitivity, while the Matrix Decomposition Framework demonstrates superior spatial generalization capabilities.

## Key Features

- **Robust High-Risk Detection**: Exceedance sensitivity of 67.0% (Probabilistic) and 61.0% (Matrix), substantially outperforming traditional methods
  ![Forecast Plot (Church Bay Beach)](https://github.com/user-attachments/assets/382e6ad8-bfbb-4a3e-92ad-fb53b353c5ae)
- **Uncertainty Quantification**: Calibrated 90% prediction intervals through Conformalized Quantile Regression
  ![Confidence Interval Calibration Plot](https://github.com/user-attachments/assets/17f8f66b-74d8-4464-854d-f52af59a53e2)
- **Model Interpretability**: Comprehensive SHAP-based feature importance and interaction analysis
  ![Global Feature Importance](https://github.com/user-attachments/assets/94985c77-a199-42bb-ae2c-27d5ce30575b)
- **Spatial Generalization**: Demonstrated ability to predict contamination levels at unseen locations
  ![Akaroa Sites](https://github.com/user-attachments/assets/b3b60491-74b0-4922-8e33-4ac8afa343f4) ![Lyttelton Sites](https://github.com/user-attachments/assets/0afce100-504e-4990-a804-81d86e5b40d6)
- **Temporal Robustness**: Consistent performance across different sampling seasons
- **Real-Time Capability**: Sub-second inference times with drift detection and missing data handling
- **Explainable Predictions**: Dashboard visualization showing feature contributions to individual forecasts

## Documentation

For more detailed information about the frameworks and methodologies:

- **Probabilistic Forecasting Framework**:
  - [Framework Overview](notebooks/model_development/probabilistic_framework.ipynb): Architecture and implementation details
  - [Quantile Regression Models](notebooks/model_development/quantile_models.ipynb): Implementation of individual quantile predictors
  - [Meta-Learner Development](notebooks/model_development/meta_learner.ipynb): Integration of quantile predictions

- **Matrix Decomposition Framework**:
  - [Framework Overview](notebooks/model_development/matrix_decomposition_framework.ipynb): NMF approach and implementation
  - [Latent Factor Analysis](notebooks/analysis/latent_factors.ipynb): Interpretation of learned factors
  - [Multi-Target Models](notebooks/model_development/multi_target_rf.ipynb): Time and Space model implementation

- **Evaluation & Analysis**:
  - [Performance Metrics](notebooks/analysis/performance_metrics.ipynb): Detailed evaluation methodology
  - [SHAP Analysis](notebooks/analysis/interpretability.ipynb): Feature importance and interaction study
  - [Cross-Validation](notebooks/analysis/cross_validation.ipynb): Temporal and spatial generalization assessment

- **Deployment**:
  - [Real-Time Pipeline](notebooks/deployment/pipeline.ipynb): Data ingestion and processing workflow
  - [Dashboard Implementation](notebooks/deployment/dashboard.ipynb): Streamlit interface development
  - [Drift Detection](notebooks/deployment/drift_detection.ipynb): ADWIN algorithm implementation

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{cheena2025dontswim,
  title={Don't Swim in Data: Real-Time Microbial Forecasting for New Zealand Recreational Waters},
  author={Cheena, Asif Juzar},
  year={2025},
  school={The University of Auckland}
}
```

For the research paper:

```bibtex
@article{cheena2025dont,
  title={Don't Swim in Data: Real-Time Microbial Forecasting for New Zealand Recreational Waters},
  author={Cheena, Asif Juzar and Dost, Katharina and Sarris, Theo and Straathof, Nina and Wicker, Jörg Simon},
  journal={Environmental Modelling & Software},
  year={2025},
  publisher={Elsevier}
}
```

## Contributing

Contributions to improve the code or extend the frameworks are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

Before submitting, please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) - see the LICENSE file for details.

## Acknowledgements

This research would not have been possible without the support and contributions of:

- **Academic Supervision**:
  - **Dr. Jörg Simon Wicker** - Primary Supervisor, School of Computer Science, The University of Auckland
  - **Dr. Katharina Dost** - Co-supervisor, Department of Knowledge Technologies, Jožef Stefan Institute

- **Data and Domain Expertise**:
  - **Nina Straathof** - Senior Scientist, Environment Canterbury
  - **Theo Sarris** - Institute for Environmental Science and Research, Christchurch

- **Data Sources**:
  - **Environment Canterbury** - Water quality sampling data
  - **NIWA** - Weather station data
  - **Lyttelton Port Company** - Weather and oceanographic data
  - **Land Information New Zealand (LINZ)** - Tide predictions

I would also like to express my gratitude to the School of Computer Science at the University of Auckland for their support throughout this Master's research project.

## Contact

For any questions or inquiries about this project, please contact:

**Asif Juzar Cheena**  
Email: [ache234@aucklanduni.ac.nz](mailto:ache234@aucklanduni.ac.nz)

![image](https://github.com/user-attachments/assets/0d817147-c67a-4afb-aef6-3dc84f7437cb)


---

*This README was last updated on April 6, 2025.*
