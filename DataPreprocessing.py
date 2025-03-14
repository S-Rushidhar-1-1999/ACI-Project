import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer: Handles Outliers using IQR Method
class OutlierHandler(BaseEstimator, TransformerMixin):
    """Custom Outlier Handler using IQR method."""
    def __init__(self, fold=1.5):
        self.fold = fold
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        """Fit method calculates IQR and outlier bounds for each numerical column."""
        X_df = pd.DataFrame(X)
        for col in range(X_df.shape[1]):
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds[col] = q1 - self.fold * iqr
            self.upper_bounds[col] = q3 + self.fold * iqr
        return self

    def transform(self, X):
        """Transform method clips the values to within the calculated IQR bounds."""
        X_df = pd.DataFrame(X)
        for col in self.lower_bounds.keys():
            X_df[col] = np.clip(X_df[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_df.values

# Load Dataset from CSV
def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset Loaded Successfully!")
        print(data.head())
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Handle Missing Values
def handle_missing_values(data, numerical_strategy='mean', categorical_strategy='most_frequent'):
    """Handle missing values in the dataset."""
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    print("\nBefore Missing Values Handling:")
    print(data.head())

    numerical_imputer = SimpleImputer(strategy=numerical_strategy)
    data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])

    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    print("\nAfter Missing Values Handling:")
    print(data.head())
    return data

# Detect and Handle Outliers
def detect_outliers(data, fold=1.5):
    """Detect and handle outliers using the IQR method."""
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    outlier_handler = OutlierHandler(fold=fold)
    outlier_handler.fit(data[numerical_cols].values)
    data[numerical_cols] = outlier_handler.transform(data[numerical_cols].values)

    print("\nOutlier Handling Summary (using IQR):")
    for col in numerical_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        print(f"Column: {col}")
        print(f"IQR: {iqr}, Q1: {q1}, Q3: {q3}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    return data

# Normalize Data
def normalize_data(data, scaling_method='standard'):
    """Normalize numerical data."""
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        print("No scaling applied.")
        return data

    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    print("\nNormalization (after scaling):")
    print(data.head())
    return data

# Feature Engineering
def feature_engineering(data):
    """Perform dynamic feature engineering on the dataset."""
    print("\nDynamic Feature Engineering:")

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    new_cols = {}  # Dictionary to store new columns
    
    if len(numerical_cols) >= 2:
        # Example: Create squared features for all numerical columns
        for col in numerical_cols:
            new_cols[f'{col}_squared'] = data[col] ** 2

        # Example: Create interaction features for pairs of numerical columns
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                new_cols[f'{col1}_times_{col2}'] = data[col1] * data[col2]
                
        # Concatenate new columns to the DataFrame
        data = pd.concat([data, pd.DataFrame(new_cols)], axis=1)
    else:
        print("Insufficient numerical columns for dynamic feature engineering.")

    print("Dynamic Feature Engineering Completed.")
    return data

# Save Preprocessed Data to CSV
def save_data(data, file_name):
    """Save the preprocessed data to a CSV file."""
    try:
        data.to_csv(file_name, index=False)
        print(f"Preprocessed dataset saved as '{file_name}' ✅")
    except Exception as e:
        print(f"Error saving the dataset: {e}")

# Main Execution Block
if __name__ == "__main__":
    file_path = "C:/Users/rushi/OneDrive/Desktop/M.Tech/!st year 1st term/ACI/Project/Titanic - Machine Learning from Disaster.csv"
    data = load_data(file_path)

    if data is not None:
        data = handle_missing_values(data)
        data = detect_outliers(data)
        data = normalize_data(data)
        data = feature_engineering(data)
        # save_data(data, "preprocessed_data.csv")
        print("\nPreprocessed Data:")
        print(data.head())
