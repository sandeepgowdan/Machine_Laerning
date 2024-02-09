import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Set the working directory
os.chdir(r'c:\Users\windows\OneDrive\Desktop\ml')

# Import phenotype file
phenotype = pd.read_csv('phenotype.csv', index_col='Taxa')

# Import genotype file
genotype = pd.read_csv('genotype.csv', index_col='Taxa')

# Initialize figure and axes
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Flatten the axes for easy iteration
axs = axs.flatten()

# Initialize DataFrame to store actual and predicted values
results_df = pd.DataFrame()

# Initialize DataFrame to store hyperparameters
hyperparameters_df = pd.DataFrame(columns=['Trait', 'Hyperparameters'])

# Initialize DataFrame to store correlation coefficients
correlation_df = pd.DataFrame(columns=['Trait', 'Correlation'])

# Iterate over each column (trait) in the phenotype dataset
for i, column in enumerate(phenotype.columns[1:]):  # Exclude the first column which is 'Taxa'
    # Split the data into training and testing sets for the current trait
    X_train, X_test, y_train, y_test = train_test_split(genotype, phenotype[column], test_size=0.2, random_state=42)
    
    # Define the Random Forest regressor
    rf_regressor = RandomForestRegressor(random_state=42)
    
    # Fit the model
    rf_regressor.fit(X_train, y_train)
    
    # Get the hyperparameters
    hyperparameters = rf_regressor.get_params()
    
    # Store hyperparameters
    hyperparameters_df = hyperparameters_df.append({'Trait': column, 'Hyperparameters': hyperparameters}, ignore_index=True)
    
    # Make predictions
    predictions = rf_regressor.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(y_test, predictions)[0, 1]
    
    # Store correlation coefficient
    correlation_df = correlation_df.append({'Trait': column, 'Correlation': correlation}, ignore_index=True)
    
    # Plot actual vs predicted values
    axs[i].scatter(y_test, predictions, label='Actual vs Predicted', color='blue')
    axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal', color='red')
    axs[i].set_xlabel('Actual')
    axs[i].set_ylabel('Predicted')
    axs[i].set_title(f'Trait: {column}')
    axs[i].legend()
    
    # Add text box with MSE, R^2 score, and correlation coefficient
    axs[i].text(0.7, 0.8, f'MSE: {mse:.2f}\nR^2: {r2:.2f}\nCorrelation: {correlation:.2f}', transform=axs[i].transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    # Store actual and predicted values in results DataFrame
    results_df[f'Actual_{column}'] = y_test
    results_df[f'Predicted_{column}'] = predictions

# Adjust layout
plt.tight_layout()

# Save the plot as a PDF file
plt.savefig('plots.pdf')

# Save the actual vs predicted values as a CSV file
results_df.to_csv('actual_vs_predicted.csv')

# Save the hyperparameters as a CSV file
hyperparameters_df.to_csv('hyperparameters.csv')

# Save the correlation coefficients as a CSV file
correlation_df.to_csv('accuaracy.csv')
