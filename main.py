
import os
import joblib
import pandas as pd

from src.preprocessing import (
    load_dataset,
    clean_dataset,
    identify_feature_types
)

from src.pipeline import create_feature_pipeline


# ------------------------------------------------
# 1. Define paths and configuration
# ------------------------------------------------

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/feature_pipeline.pkl"
OUTPUT_PATH = "output/transformed_data.csv"

TARGET_COLUMN = "Survived"

# Features selected for our Titanic experiment
numerical_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# ------------------------------------------------
# 2. Load dataset
# ------------------------------------------------

df = load_dataset(DATA_PATH)


# ------------------------------------------------
# 3. Clean dataset
# ------------------------------------------------

df = clean_dataset(df)
identify_feature_types(df, TARGET_COLUMN)

# ------------------------------------------------
# 4. Separate features and target
# ------------------------------------------------

X = df[numerical_features + categorical_features]
y = df[TARGET_COLUMN]

print("\nSelected Input Features:")
print(X.columns.tolist())


# ------------------------------------------------
# 5. Create feature engineering pipeline
# ------------------------------------------------

pipeline = create_feature_pipeline(
    numerical_features,
    categorical_features,
    k=8
)


# ------------------------------------------------
# 6. Transform dataset
# ------------------------------------------------

transformed_data = pipeline.fit_transform(X, y)

# Get names of features after preprocessing
preprocessor = pipeline.named_steps["preprocessor"]

all_feature_names = preprocessor.get_feature_names_out()

# Identify selected features
selector = pipeline.named_steps["feature_selection"]

selected_features = all_feature_names[
    selector.get_support()
]

print("\nSelected Features:")
for feature in selected_features:
    print(feature)

# Create transformed DataFrame
transformed_df = pd.DataFrame(
    transformed_data,
    columns=selected_features
)

# Add target column
transformed_df[TARGET_COLUMN] = y.reset_index(drop=True)

print("\nOriginal Dataset Shape:", df.shape)
print("Transformed Dataset Shape:", transformed_df.shape)


# ------------------------------------------------
# 7. Save transformed dataset
# ------------------------------------------------

os.makedirs("output", exist_ok=True)

transformed_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nTransformed Dataset Saved Successfully!")


# ------------------------------------------------
# 8. Save pipeline
# ------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, MODEL_PATH)

print("Pipeline Saved Successfully!")


# ------------------------------------------------
# 9. Reuse saved pipeline
# ------------------------------------------------

loaded_pipeline = joblib.load(MODEL_PATH)

new_sample = X.head(5)

new_transformed_data = loaded_pipeline.transform(new_sample)

print("\nNew Sample Transformed Successfully!")

print("New Sample Shape:", new_transformed_data.shape)

print("\nPipeline Reuse Successful!")
