# File: connect_gdrive.py
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Load the dataset
path = '/content/drive/MyDrive/winequality-red[1].csv'
data = pd.read_csv(path)

# Data Cleaning
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# Descriptive Statistics
print(data.describe())

# Exploratory Data Analysis (EDA)

# Histograms for each feature
data.hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograms of Features')
plt.show()

# Pairplot for features
sns.pairplot(data, hue='quality')
plt.suptitle('Pairplot of Features by Quality')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Histogram of Alcohol Content
plt.figure(figsize=(8, 6))
sns.histplot(data['alcohol'], kde=True)
plt.title('Distribution of Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot of Alcohol vs Quality
plt.figure(figsize=(8, 6))
sns.scatterplot(x='alcohol', y='quality', data=data)
plt.title('Alcohol Content vs Wine Quality')
plt.xlabel('Alcohol Content')
plt.ylabel('Wine Quality')
plt.show()

# Prepare features (X) and target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost Model with Grid Search
param_grid = {
    'objective': ['reg:squarederror'],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# XGBoost Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Visualization of XGBoost Model Performance
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

plt.figure(figsize=(12, 8))
xgb.plot_importance(best_model, importance_type='weight', title='Feature Importance')
plt.show()

# Model Training and Evaluation for Classification Models
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for model_name, model in models.items():
    model.fit(X_train_class_scaled, y_train_class)
    y_pred_class = model.predict(X_test_class_scaled)
    print(f'\n{model_name} Performance:')
    print('Confusion Matrix:\n', confusion_matrix(y_test_class, y_pred_class))
    print('Classification Report:\n', classification_report(y_test_class, y_pred_class))

    if len(np.unique(y_test_class)) > 2:
        roc_auc = roc_auc_score(y_test_class, model.predict_proba(X_test_class_scaled), multi_class='ovr')
        print(f'ROC AUC Score: {roc_auc:.2f}')
    else:
        roc_auc = roc_auc_score(y_test_class, model.predict_proba(X_test_class_scaled)[:,1])
        print(f'ROC AUC Score: {roc_auc:.2f}')

    print(f'Accuracy Score: {accuracy_score(y_test_class, y_pred_class):.2f}')

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances for {model_name}')
        plt.barh(range(X.shape[1]), importances[indices], align='center')
        plt.yticks(range(X.shape[1]), X.columns[indices])
        plt.xlabel('Feature Importance')
        plt.show()
