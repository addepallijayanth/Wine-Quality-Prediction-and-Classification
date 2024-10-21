# Wine Quality Prediction and Classification

This project involves building a machine learning model to predict and classify wine quality using various supervised learning algorithms. The dataset used for this project is a red wine quality dataset (`winequality-red.csv`), which contains physicochemical properties of red wine and their corresponding quality scores. The project includes both regression (to predict wine quality scores) and classification (to categorize wine quality into different levels).

## Table of Contents
- [Overview](#overview)
- [Algorithms Used](#algorithms-used)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
  - [XGBoost Regression Model](#xgboost-regression-model)
  - [Classification Models](#classification-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [Conclusion](#conclusion)

## Overview
This project aims to predict wine quality based on various chemical properties like acidity, residual sugar, chlorides, and alcohol content. The prediction is treated as both a regression and a classification problem. The code explores and visualizes the data, applies feature scaling, and builds models using several machine learning algorithms, including XGBoost, Decision Tree, Random Forest, and Gradient Boosting.

## Algorithms Used
The following algorithms have been used in this project:

1. **XGBoost (Extreme Gradient Boosting)** - For regression, XGBoost helps predict the exact wine quality score.
2. **Decision Tree Classifier** - A non-parametric supervised learning method for classifying wine quality.
3. **Random Forest Classifier** - An ensemble learning method based on multiple decision trees for more robust classification.
4. **Gradient Boosting Classifier** - Another ensemble learning method that optimizes for classification accuracy.
5. **Sigmoid Activation Function and Neural Network** - Basic implementation of forward propagation and backpropagation in a simple neural network.

## Dataset
The dataset `winequality-red.csv` is available in the repository under the path `winequality-red.csv`. This dataset contains 11 features related to the physicochemical properties of wine and one target feature (`quality`), which is the wine quality score ranging from 0 to 10.

Features include:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## Project Workflow
The project follows a structured approach:
1. **Data Loading**: Load the dataset from Google Drive.
2. **Data Cleaning**: Handle missing values and remove duplicates.
3. **Exploratory Data Analysis (EDA)**: Perform visualizations to understand feature distributions and relationships.
4. **Feature Scaling**: Standardize the features to ensure all variables are on the same scale.
5. **Model Training**: Train models for both regression and classification.
6. **Model Evaluation**: Evaluate the model performance using different metrics.

## Data Preprocessing
- **Scaling**: Standardization is performed using `StandardScaler` to ensure that all features have a mean of 0 and a standard deviation of 1. This is crucial for models that are sensitive to the scale of input data.
- **Train-Test Split**: The dataset is split into training and testing sets using an 80-20 split for regression and a 70-30 split for classification.

## Exploratory Data Analysis
Key EDA steps:
- **Histograms**: Display the distribution of each feature.
- **Pairplots**: Visualize the relationship between features based on wine quality.
- **Correlation Heatmap**: Show the correlation between different features.

## Modeling

### XGBoost Regression Model
For predicting wine quality as a continuous variable, XGBoost regression is used. A Grid Search is performed to find the optimal hyperparameters for the XGBoost model, including:
- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`

The best hyperparameters are then used to fit the model on the training data and predict the wine quality on the test set.

### Classification Models
For classifying wine quality into categories, the following models are used:
- **Decision Tree Classifier**: A simple tree-based model that splits data based on feature values.
- **Random Forest Classifier**: An ensemble of decision trees that improves classification accuracy by reducing overfitting.
- **Gradient Boosting Classifier**: An iterative approach where each tree corrects the errors of the previous trees.

Each classifier is evaluated using a confusion matrix, classification report, and ROC AUC score.

## Evaluation Metrics
- **MSE (Mean Squared Error)**: Evaluates the prediction error for regression models.
- **MAE (Mean Absolute Error)**: Measures the average absolute errors in predictions.
- **RMSE (Root Mean Squared Error)**: Square root of MSE to bring error values to the same unit as the target variable.
- **R² (R-squared)**: Measures how well the predictions fit the actual data.
- **Confusion Matrix**: Shows the performance of classification models in a tabular format.
- **Classification Report**: Includes precision, recall, f1-score, and accuracy for classification models.
- **ROC AUC Score**: Evaluates the performance of a classification model by plotting the ROC curve.

## Results
- The XGBoost regression model performed best for predicting continuous wine quality scores, with optimal hyperparameters found through Grid Search.
- The Random Forest and Gradient Boosting classifiers provided the best classification results in terms of accuracy and ROC AUC scores.
  
## Requirements
Install the necessary Python libraries by running:
```bash
pip install -r requirements.txt
```

## How to Use
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/wine-quality-prediction.git
    ```
2. Load the dataset by connecting your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Follow the steps in the `wine-quality-prediction.ipynb` notebook for data loading, preprocessing, and model training.
4. Run the code to view the performance of the models.

## Conclusion
This project showcases the use of both regression and classification models for predicting wine quality. The XGBoost model performed well in regression, and Random Forest and Gradient Boosting were effective in classifying wine quality. Additionally, EDA provided valuable insights into feature relationships and their impact on wine quality.
