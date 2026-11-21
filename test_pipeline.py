
import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("models/feature_pipeline.pkl")

# Load new sample data
df = pd.read_csv("data/dataset.csv")

# Use five new passenger records
new_data = pd.DataFrame({
    "Pclass": [3, 1],
    "Age": [22, 38],
    "SibSp": [1, 1],
    "Parch": [0, 0],
    "Fare": [7.25, 71.28],
    "Sex": ["male", "female"],
    "Embarked": ["S", "C"]
})

X_new = new_data

# Transform using the saved pipeline
transformed_data = pipeline.transform(X_new)

print("Saved Pipeline Loaded Successfully!")

print("\nOriginal Sample:")
print(X_new)

print("\nTransformed Sample:")
print(transformed_data)

print("\nTransformed Shape:", transformed_data.shape)
print("\nPipeline Reuse Test Passed!")
