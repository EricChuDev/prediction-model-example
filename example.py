import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('example.csv')

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Environmental Factors vs Plant Height')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Prepare features and target
X = df.drop('plant_height_cm', axis=1)
y = df['plant_height_cm']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
feature_importance = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {'R2 Score': r2, 'RMSE': rmse}
    
    # Get feature importance
    if name == 'Random Forest':
        importance = dict(zip(X.columns, model.feature_importances_))
        feature_importance = {k: v for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)}

# Print results
print("\nModel Performance:")
print("-----------------")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"R² Score: {metrics['R2 Score']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")

print("\nFeature Importance (Random Forest):")
print("----------------------------------")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

# Create feature importance plot
plt.figure(figsize=(10, 6))
plt.bar(feature_importance.keys(), feature_importance.values())
plt.xticks(rotation=45)
plt.title('Feature Importance in Predicting Plant Height')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Example prediction for a new experimental condition
print("\nExample Prediction:")
print("-----------------")
example_condition = pd.DataFrame({
    'temperature': [24.0],
    'humidity': [65],
    'light_intensity': [800],
    'water_ml_daily': [50],
    'nutrient_concentration': [300],
    'co2_levels': [450]
})

for name, model in models.items():
    predicted_height = model.predict(example_condition)[0]
    print(f"{name} predicts plant height: {predicted_height:.2f} cm")
