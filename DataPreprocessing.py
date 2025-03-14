import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer: Converts NumPy array to DataFrame
class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Converts NumPy array to DataFrame."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

# Custom Transformer: Handles Outliers using IQR Method
class OutlierHandler(BaseEstimator, TransformerMixin):
    """Custom Outlier Handler using IQR method."""
    def __init__(self, fold=1.5):
        self.fold = fold
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        """Fit method calculates IQR and outlier bounds for each numerical column."""
        X_df = pd.DataFrame(X)  # Convert NumPy array to DataFrame
        for col in range(X_df.shape[1]):  # Iterate over column indices
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            # Calculate lower and upper bounds based on IQR
            self.lower_bounds[col] = q1 - self.fold * iqr
            self.upper_bounds[col] = q3 + self.fold * iqr
        return self

    def transform(self, X):
        """Transform method clips the values to within the calculated IQR bounds."""
        X_df = pd.DataFrame(X)  # Convert NumPy array to DataFrame
        for col in self.lower_bounds.keys():  # Iterate over column indices
            X_df[col] = np.clip(X_df[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_df.values  # Convert DataFrame back to NumPy array

# Load Dataset
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

# Preprocess Data
def preprocess_data(data, numerical_imputation_strategy='mean', categorical_imputation_strategy='most_frequent', scaling_method='standard'):
    """Preprocess the dataset: handle missing values, outliers, and scaling."""
    
    # Drop non-informative columns (e.g., IDs)
    non_informative_cols = [col for col in data.columns if data[col].nunique() == len(data)]  # Columns with all unique values
    if non_informative_cols:
        print(f"Dropping non-informative columns: {non_informative_cols}")
        data = data.drop(non_informative_cols, axis=1)

    # Detect numerical and categorical columns dynamically
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Logging preprocessing steps
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Missing Values Handling: Before Imputation
    print("\nBefore Missing Values Handling:")
    print(data[numerical_cols].isnull().sum())

    # Numerical Pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=numerical_imputation_strategy)),  # Handle missing values
        ('outlier_handler', OutlierHandler()),  # Handle outliers
        ('scaler', StandardScaler() if scaling_method == 'standard' else MinMaxScaler() if scaling_method == 'minmax' else None)  # Normalize
    ])

    # Categorical Pipeline (only if categorical columns exist)
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=categorical_imputation_strategy)),  # Handle missing values
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))  # One-hot encode
        ])
    else:
        categorical_pipeline = None

    # Create ColumnTransformer
    transformers = [
        ('numerical', numerical_pipeline, numerical_cols)
    ]
    if categorical_pipeline:
        transformers.append(('categorical', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers)
    
    # Fit and Transform data
    print("\nStarting preprocessing...")
    processed_data = preprocessor.fit_transform(data)
    print("Preprocessing complete!")

    # After Missing Values Handling
    print("\nAfter Missing Values Handling:")
    # Convert transformed data back to DataFrame to check for missing values
    processed_data_df = pd.DataFrame(processed_data)
    print(processed_data_df.isnull().sum())

    # Outlier Handling Summary (using IQR)
    print("\nOutlier Handling Summary (using IQR):")
    X_df = pd.DataFrame(processed_data)  # Convert NumPy array to DataFrame
    for col in range(X_df.shape[1]):  # Iterate over column indices
        q1 = X_df[col].quantile(0.25)
        q3 = X_df[col].quantile(0.75)
        iqr = q3 - q1
        # Calculate lower and upper bounds based on IQR
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        print(f"Column: {col}")
        print(f"IQR: {iqr}, Q1: {q1}, Q3: {q3}, Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Normalization (after scaling)
    print("\nNormalization (after scaling):")
    if scaling_method:
        scaled_data = StandardScaler().fit_transform(processed_data)  # Use StandardScaler for normalization
        print(pd.Series(scaled_data.flatten()))  # Flatten for easy printing
    else:
        print("No scaling applied.")

    # Feature Engineering Placeholder
    print("\nFeature Engineering Placeholder:")
    print("Feature Engineering could include transformations, combinations of features, and other domain-specific features.")

    # Dynamically assign column names based on transformed data
    feature_names = numerical_cols + list(preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols)) if categorical_cols else numerical_cols
    processed_data_df = pd.DataFrame(processed_data, columns=feature_names)

    return processed_data_df

# Save Preprocessed Data
def save_data(data, file_name):
    """Save the preprocessed data to a CSV file."""
    try:
        data.to_csv(file_name, index=False)
        print(f"Preprocessed dataset saved as '{file_name}' ✅")
    except Exception as e:
        print(f"Error saving the dataset: {e}")

# Main Execution
if __name__ == "__main__":
    # Load dataset
    file_path = "C:/Users/rushi/OneDrive/Desktop/M.Tech/!st year 1st term/ACI/Project/Credit Card Fraud Detection.csv"
    data = load_data(file_path)

    if data is not None:
        # Preprocess the data with customizable imputation and scaling
        preprocessed_data = preprocess_data(
            data,
            numerical_imputation_strategy='mean',  # Options: 'mean', 'median', 'most_frequent'
            categorical_imputation_strategy='most_frequent',  # Options: 'most_frequent', 'constant'
            scaling_method='standard'  # Options: 'standard', 'minmax', None
        )
        
        print("\nPreprocessed Data:")
        print(preprocessed_data.head())
        
        # Save the preprocessed data
        save_data(preprocessed_data, "preprocessed_data.csv")
