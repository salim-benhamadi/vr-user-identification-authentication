# VR User Identification & Authentication

A machine learning system for identifying and authenticating users based on VR behavioral biometrics from movement and network traffic patterns.

## Overview

This project implements user authentication systems using behavioral biometrics extracted from VR gameplay sessions. The system analyzes head and controller movements along with network traffic patterns to create unique user profiles for identification and authentication purposes.

## Project Structure

```
vr-user-identification/
├── data/                           # Data storage
│   ├── processed/                  # Cleaned and processed datasets
│   └── raw/                        # Raw VR session data
├── env/                            # Virtual environment
├── models/                         # Trained models and scalers
│   ├── scalers/                    # Feature scaling models
│   └── user_authentification/     # User-specific models
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_exploratory_data_analysis/
│   │   └── 01-sh-exploratory-data-analysis.ipynb
│   ├── 02_user_identification/
│   │   ├── 01-sh-user-identification-by-game.ipynb
│   │   ├── 02-sh-user-identification-mov+traffic.ipynb
│   │   ├── 02-sh-user-identification-by-group-without-height.ipynb
│   │   ├── 03-sh-user-identification-by-game-mov+traffic-without-h...
│   │   └── 04-sh-user-identification-by-group.ipynb
│   └── 03_user_authentification/
│       ├── 01-sh-standard-authentification.ipynb
│       ├── 02-sh-standard-authentification-without-height.ipynb
│       ├── 03-sh-standard-authentification-traffic.ipynb
│       ├── 04-sh-standard-authentification-traffic-without-height...
│       └── 05-sh-hierarchical-authentification.ipynb
├── results/                        # Evaluation results and plots
│   └── user_authentification/
│       ├── intruders_evaluation/   # Intruder attack results
│       └── training_evaluation/    # Model performance metrics
└── src/                           # Source code
    ├── 01_data_pipeline/          # Data preprocessing
    │   ├── data_cleaner.py         # Feature scaling and cleaning
    │   └── user_feature_generator.py # Feature extraction
    ├── 02_user_identification/    # User identification models  
    │   ├── model_trainer.py        # Multi-classifier training
    │   └── model_visualizer.py     # Performance visualization
    └── 03_user_authentification/ # Authentication system
        ├── user_authentification.py      # Main authentication system
        ├── intruder_authentification.py  # Attack simulation
        ├── intruder_feature_generator.py # Adversarial sample generation
        └── user_hierarchical_auth.py     # Scalable clustering approach
```

## Used Data
The datasets used in this project are available [here](https://drive.google.com/drive/folders/1IgKOyrNcIql0BSkDvzMz7xP5ZsHrioKC?usp=sharing). They include VR session data from four games: Beat Saber, a Forklift Simulator, and two others, capturing both movement and network traffic patterns.

## License

This project is for research purposes. Please cite appropriately if used in academic work.
