# ML-Based Wildfire Detection
![Python](https://img.shields.io/badge/Python--3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn--F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas--150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=Jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ML-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter%20Tuning-Optimization-9C27B0?style=flat)

A machine learning project for predicting wildfire occurrences using NASA satellite measurement data. This project implements and compares multiple classification algorithms with hyperparameter tuning.

## Project Overview

### Problem Statement

Wildfires pose significant threats to ecosystems, property, and human lives. Early detection and prediction of wildfire occurrences can help in resource allocation, evacuation planning, and fire prevention strategies. This project addresses the challenge of predicting wildfire occurrences using environmental and meteorological data collected from NASA satellites.

### Objectives

- Develop and compare multiple machine learning classification models to predict wildfire occurrences
- Identify the most important environmental features contributing to wildfire risk
- Optimize model performance through systematic hyperparameter tuning
- Evaluate models using comprehensive performance metrics including accuracy, precision, recall, F1-score, and ROC-AUC

### Dataset

The project uses a combined dataset consisting of:

- **Wildfire Occurrence Data**: Historical wildfire records from the US Forest Service National Fire Occurrence Point dataset, containing over 500,000 fire incidents across the contiguous United States (1950-2024)
- **NASA Satellite Data**: Meteorological and environmental measurements from NASA's Prediction of Worldwide Energy Resources (POWER) project

**Dataset Characteristics:**
- **Total Instances**: ~30,000 training instances (balanced dataset: 15,000 fire=0, 14,950 fire=1)
- **Features**: 12 environmental variables (excluding Date and target variable)
- **Train/Test Split**: 80/20 split with fixed random state (1234) for reproducibility


### Methodology

1. **Data Preprocessing**:
   - Geographic filtering to contiguous United States boundaries
   - Date range filtering (1950-2024)
   - Data quality checks and cleaning
   - Feature extraction and preparation

2. **Model Development**:
   - **Baseline Models**: Initial implementation without hyperparameter tuning for quick comparison
   - **Hyperparameter Tuning**: Systematic optimization using grid search and cross-validation techniques
   - **Model Evaluation**: Comprehensive performance assessment using multiple metrics

3. **Models Implemented**: Random Forest, Decision Tree, Logistic Regression, Multilayer Perceptron (MLP), K-Nearest Neighbors (KNN)

4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

**Detailed documentation and results:** See the [Project Report PDF](https://github.com/Uchswas/ml-based-wildfire-detection/blob/main/Report%20and%20Documentation.pdf) for comprehensive analysis, results, and findings.



## How to Run in Jupyter Notebook

### Prerequisites

1. **Install Python** 

2. **Install required packages:**
   ```bash
   pip install jupyter pandas numpy matplotlib seaborn scikit-learn
   ```
   

### Running the Notebooks

1. **Start Jupyter Notebook:**
   ```bash
   cd /path/to/ml-based-wildfire-detection
   jupyter notebook
   ```
   
   This will open Jupyter in your web browser (typically at `http://localhost:8888`)

2. **Navigate to the `src/` directory** in the Jupyter interface

3. **Run notebooks:** Start with `baseline_models.ipynb` for baseline comparisons, then run any of the hyperparameter tuning notebooks (`decision_tree_*`, `knn_*`, `logistic_regression_*`, `multilayer_perceptron_*`, `random_forest_*`) independently.



