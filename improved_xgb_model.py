import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('iPD_foundin1_dan_snp_features.csv')

# Data exploration and preprocessing
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values if any
df = df.dropna()

# Select features and target
features = ['pval', 'acc_mean', 'acc_PD', 'acc_HC', 'acc_diff', 'cell_specificity', 'width', 'tss_distance']
X = df[features]
y = df['effect_size']

# Check for outliers in target variable
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = 1.5 * IQR
outliers = ((y < (Q1 - outlier_threshold)) | (y > (Q3 + outlier_threshold))).sum()
print(f"\nNumber of outliers in target variable: {outliers}")

# Basic statistics
print("\nTarget variable statistics:")
print(y.describe())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize base model
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# Grid search with cross-validation
cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=cv_folds,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Cross-validation evaluation
print("\nCross-validation results:")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                           cv=cv_folds, scoring='neg_mean_squared_error')
cv_r2_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                              cv=cv_folds, scoring='r2')

print(f"CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"CV R²: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")

# Train final model on full training set
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

# Evaluate model performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nModel Performance:")
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Testing MAE: {test_mae:.4f}")

# Check for overfitting
if train_r2 - test_r2 > 0.1:
    print("\nWarning: Potential overfitting detected!")
else:
    print("\nModel appears to generalize well.")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Feature importance plot
axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Feature Importance')
axes[0, 0].invert_yaxis()

# 2. Predicted vs Actual (test set)
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Effect Size')
axes[0, 1].set_ylabel('Predicted Effect Size')
axes[0, 1].set_title(f'Predicted vs Actual (Test Set)\nR² = {test_r2:.3f}')

# 3. Residuals plot
residuals = y_test - y_test_pred
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Effect Size')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residuals Plot')

# 4. Distribution of residuals
axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Residuals')

plt.tight_layout()
plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Predict effect sizes for all SNPs
X_scaled = scaler.transform(X)
df['predicted_effect_size'] = best_model.predict(X_scaled)

# Calculate prediction intervals (approximate)
# Using standard deviation of residuals as uncertainty estimate
residual_std = np.std(residuals)
df['prediction_lower'] = df['predicted_effect_size'] - 1.96 * residual_std
df['prediction_upper'] = df['predicted_effect_size'] + 1.96 * residual_std

# Rank SNPs by predicted effect size
df['abs_predicted_effect_size'] = df['predicted_effect_size'].abs()
ranked_snps = df[['rsID', 'predicted_effect_size', 'abs_predicted_effect_size', 
                  'prediction_lower', 'prediction_upper']].sort_values(
    by='abs_predicted_effect_size', ascending=False)

# Output top ranked SNPs
print("\nTop 20 Ranked SNPs based on predicted effect size:")
print(ranked_snps.head(20)[['rsID', 'predicted_effect_size']])

# Save results
ranked_snps.to_csv('ranked_snps_improved.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

# Model interpretation: SHAP values (optional, requires shap library)
try:
    import shap
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled[:100])  # Sample for visualization
    
    # Summary plot
    shap.summary_plot(shap_values, X_test.iloc[:100], feature_names=features, show=False)
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except ImportError:
    print("\nSHAP library not installed. Install with: pip install shap")
    print("SHAP values provide detailed feature importance for individual predictions.")

print(f"\nModel saved successfully!")
print(f"Files created:")
print(f"- ranked_snps_improved.csv")
print(f"- feature_importance.csv") 
print(f"- model_evaluation_plots.png")
print(f"- shap_summary_plot.png (if SHAP is available)")

# Additional diagnostics
print("\nAdditional Model Diagnostics:")
print(f"Mean prediction: {df['predicted_effect_size'].mean():.4f}")
print(f"Std prediction: {df['predicted_effect_size'].std():.4f}")
print(f"Min prediction: {df['predicted_effect_size'].min():.4f}")
print(f"Max prediction: {df['predicted_effect_size'].max():.4f}")

# Correlation between actual and predicted
if 'effect_size' in df.columns:
    correlation = df['effect_size'].corr(df['predicted_effect_size'])
    print(f"Overall correlation (actual vs predicted): {correlation:.4f}")
