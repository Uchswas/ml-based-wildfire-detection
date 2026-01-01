# ML-Based Wildfire Detection

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

3. **Run notebooks:** Start with `Baseline_Models.ipynb` for baseline comparisons, then run any of the hyperparameter tuning notebooks (`Decision_Tree_*`, `KNN_*`, `Logistic_Regression_*`, `Multilayer_Perceptron_*`, `Random_Forest_*`) independently.



