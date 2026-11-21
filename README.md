# Feature Engineering Pipeline for Machine Learning

## 1. Project Overview

This project implements a reusable Feature Engineering Pipeline using Python and Scikit-learn. The pipeline automatically preprocesses raw data and transforms it into a machine-learning-ready dataset.

The Titanic dataset is used to demonstrate different preprocessing techniques, including missing value imputation, categorical encoding, numerical scaling, and feature selection.

## 2. Objectives

- Handle missing numerical and categorical values.
- Remove duplicate records.
- Scale numerical features.
- Encode categorical variables.
- Perform feature selection.
- Build a reusable preprocessing pipeline.
- Save and load the pipeline using Joblib.
- Transform new data using the saved pipeline.

## 3. Dataset

**Dataset:** Titanic Dataset

**Source:** [Titanic Dataset – DataScienceDojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)

- Total records: 891
- Total columns: 12
- Target variable: `Survived`

The target variable indicates whether a passenger survived.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## 5. Feature Engineering Techniques

### Numerical Features
- Missing value imputation using Median.
- Feature scaling using StandardScaler.

### Categorical Features
- Missing value imputation using Most Frequent.
- Categorical encoding using OneHotEncoder.

### Feature Selection
- SelectKBest
- Mutual Information (`mutual_info_classif`)
- Number of selected features: 8

### Pipeline Architecture
- Pipeline
- ColumnTransformer

## 6. Project Structure

```text
Assignment3/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── preprocessing.py
│   ├── pipeline.py
│   └── feature_selection.py
│
├── models/
│   └── feature_pipeline.pkl
│
├── output/
│   └── transformed_data.csv
│
├── screenshots/
│
├── main.py
├── check_dataset.py
├── test_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 7. Installation and Execution

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Assignment3
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows:

```bash
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Execute the pipeline

```bash
python main.py
```

### Step 6: Test the saved pipeline

```bash
python test_pipeline.py
```

## 8. Expected Results

| Description | Result |
|---|---|
| Original Dataset Shape | (891, 12) |
| Transformed Dataset Shape | (891, 9) |
| Selected Features | 8 |
| New Sample Shape | (5, 8) |

The transformed dataset contains eight selected features and the target column.

## 9. Output Files

- `models/feature_pipeline.pkl`: Saved preprocessing pipeline.
- `output/transformed_data.csv`: Transformed dataset.

## 10. Conclusion

The project successfully implements a reusable Feature Engineering Pipeline using Scikit-learn. It combines data cleaning, missing value imputation, numerical scaling, categorical encoding, and feature selection into a single workflow.

The saved pipeline can be loaded and reused to preprocess new input data consistently, demonstrating an important practice in machine learning and MLOps.
