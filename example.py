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
plt.title('Correlation Heatmap of Factors vs Sliding Time')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Prepare features and target
X = df.drop('sliding_time_s', axis=1)
y = df['sliding_time_s']

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
plt.title('Feature Importance in Predicting Sliding Time')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Example prediction for a new experimental setup
print("\nExample Prediction:")
print("-----------------")
example_condition = pd.DataFrame({
    'starting_position_m': [2.0],
    'mass_kg': [0.5],
    'friction_coefficient': [0.15],
    'surface_area_cm2': [25],
    'angle_degrees': [30]
})

for name, model in models.items():
    predicted_time = model.predict(example_condition)[0]
    print(f"{name} predicts sliding time: {predicted_time:.2f} seconds")

# Additional physics-based visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['angle_degrees'], df['sliding_time_s'])
plt.xlabel('Angle (degrees)')
plt.ylabel('Sliding Time (s)')
plt.title('Angle vs Sliding Time')

plt.subplot(1, 2, 2)
plt.scatter(df['friction_coefficient'], df['sliding_time_s'])
plt.xlabel('Friction Coefficient')
plt.ylabel('Sliding Time (s)')
plt.title('Friction vs Sliding Time')

plt.tight_layout()
plt.savefig('physics_relationships.png')
plt.close()
