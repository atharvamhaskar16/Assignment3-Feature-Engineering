
import pandas as pd


def load_dataset(file_path):
    """
    Load the CSV dataset and display basic information.
    """
    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully!")
    print("\nOriginal Dataset Shape:", df.shape)

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def clean_dataset(df):
    """
    Remove duplicate records and identify numerical
    and categorical features.
    """
    initial_count = len(df)

    df = df.drop_duplicates().copy()

    removed = initial_count - len(df)

    print("\nDuplicate Records Removed:", removed)
    print("Cleaned Dataset Shape:", df.shape)

    return df


def identify_feature_types(df, target_column):
    """
    Identify numerical and categorical columns,
    excluding the target variable.
    """

    X = df.drop(columns=[target_column])

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    return numerical_features, categorical_features
