import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 30, np.nan, 40, 35, 60, 28],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', np.nan],
    'Income': [50000, 60000, 45000, np.nan, 70000, 55000, 48000],
    'Product Purchased': ['Laptop', 'Smartphone', 'Laptop', 'Tablet', 'Smartphone', 'Tablet', 'Laptop']
})

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

# Feature Engineering Functions
def feature_engineering(data):
    print("\nFeature Engineering Step:")
    
    # 1. Creating Interaction Features: Combine Age and Income into a WealthIndex
    data['WealthIndex'] = data['Age'] * data['Income']
    print("\nAfter Creating WealthIndex:")
    print(data[['Age', 'Income', 'WealthIndex']])

    # 2. Binning Age into categories: Young, Middle-Aged, Senior
    bins = [0, 30, 50, 100]  # Define bins
    labels = ['Young', 'Middle-Aged', 'Senior']  # Corresponding labels
    data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    print("\nAfter Binning Age into AgeGroup:")
    print(data[['Age', 'AgeGroup']])

    # 3. Creating Polynomial Feature: Age squared (capturing non-linear relationships)
    data['AgeSquared'] = data['Age'] ** 2
    print("\nAfter Creating AgeSquared Feature:")
    print(data[['Age', 'AgeSquared']])

    # 4. Extracting useful time-based features (example with Product Purchased)
    data['TechSavvy'] = data['Product Purchased'].apply(lambda x: 1 if x in ['Laptop', 'Smartphone'] else 0)
    print("\nAfter Creating TechSavvy Feature:")
    print(data[['Product Purchased', 'TechSavvy']])
    
    return data

# Main Preprocessing Function
def preprocess_data(data, numerical_imputation_strategy='mean', categorical_imputation_strategy='most_frequent', scaling_method='standard'):
    print("\nOriginal Dataset:")
    print(data)

    # Impute missing values
    data_imputed = data.copy()

    # Impute missing Age with mean
    print("\nImputing missing values for Age (mean):")
    data_imputed['Age'].fillna(data_imputed['Age'].mean(), inplace=True)
    print(data_imputed[['Age']])

    # Impute missing Income with mean
    print("\nImputing missing values for Income (mean):")
    data_imputed['Income'].fillna(data_imputed['Income'].mean(), inplace=True)
    print(data_imputed[['Income']])

    # Impute missing Gender with most frequent value
    print("\nImputing missing values for Gender (most frequent):")
    data_imputed['Gender'].fillna(data_imputed['Gender'].mode()[0], inplace=True)
    print(data_imputed[['Gender']])

    # Handle outliers using IQR
    print("\nHandling outliers for Age and Income:")
    numerical_data = data_imputed[['Age', 'Income']].values
    outlier_handler = OutlierHandler()
    outlier_handler.fit(numerical_data)
    data_no_outliers = outlier_handler.transform(numerical_data)

    # Put the transformed data back into the DataFrame
    data_imputed[['Age', 'Income']] = data_no_outliers
    print("\nAfter Outlier Handling (Clipping Age and Income):")
    print(data_imputed[['Age', 'Income']])

    # Feature Engineering
    data_imputed = feature_engineering(data_imputed)

    # Scaling Numerical Features
    if scaling_method == 'standard':
        print("\nApplying Standard Scaling to Age and Income:")
        scaler = StandardScaler()
        data_imputed[['Age', 'Income']] = scaler.fit_transform(data_imputed[['Age', 'Income']])
        print(data_imputed[['Age', 'Income']])

    # One-Hot Encoding for Categorical Features
    print("\nApplying One-Hot Encoding to Categorical Features (Gender and Product Purchased):")
    categorical_cols = ['Gender', 'Product Purchased']
    preprocessor = ColumnTransformer([
        ('gender', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Gender']),
        ('product', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Product Purchased'])
    ])

    encoded_data = preprocessor.fit_transform(data_imputed)
    encoded_feature_names = preprocessor.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
    print(encoded_df)

    # Concatenate with the numerical data (Age and Income)
    final_data = pd.concat([data_imputed[['Age', 'Income']], encoded_df], axis=1)
    print("\nFinal Preprocessed Dataset:")
    print(final_data)

    return final_data

# Main Execution
if __name__ == "__main__":
    # Load and Preprocess Data
    preprocessed_data = preprocess_data(data)

    print("\nFinal Preprocessed Dataset with Feature Engineering and Transformations:")
    print(preprocessed_data)
