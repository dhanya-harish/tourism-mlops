import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from datasets import load_dataset
from huggingface_hub import login, HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO", "dhani10/tourism-app-dataset")
MODEL_REPO = os.getenv("MODEL_REPO", "dhani10/tourism-model")

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN is required")

print("[train] login + load datasets from HF")
login(HF_TOKEN)
dataset = load_dataset("csv", data_files={
    "train": f"hf://datasets/{DATASET_REPO}/data/train.csv",
    "test":  f"hf://datasets/{DATASET_REPO}/data/test.csv",
})

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

X_train, y_train = train_df.drop("ProdTaken", axis=1), train_df["ProdTaken"].astype(int)
X_test, y_test   = test_df.drop("ProdTaken", axis=1), test_df["ProdTaken"].astype(int)

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipe = Pipeline([("preprocessor", preproc),
                 ("classifier", DecisionTreeClassifier(random_state=42))])

param_grid = {
    "classifier__max_depth": [3, 5, 7, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__criterion": ["gini", "entropy"]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("[train] best params:", grid.best_params_)
print("[train] saving + uploading model")

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/best_model.joblib"
joblib.dump(best_model, model_path)

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model/best_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("[train] done")
