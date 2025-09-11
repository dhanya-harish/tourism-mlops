import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO", "dhani10/tourism-app-dataset")
CSV = os.getenv("CSV_PATH", "tourism.csv")

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN is required")

print(f"[data_prep] reading {CSV}")
df = pd.read_csv(CSV)

for c in ("CustomerID", "Unnamed: 0"):
    if c in df.columns:
        df.drop(columns=[c], inplace=True)
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

for col in ["Age","DurationOfPitch","NumberOfTrips","NumberOfFollowups","MonthlyIncome","PreferredPropertyStar"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
for col in ["TypeofContact","Occupation","Gender","ProductPitched","MaritalStatus","Designation"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"].astype(int)
train_df, test_df = train_test_split(pd.concat([X, y], axis=1), test_size=0.2, random_state=42, stratify=y)

out = Path("artifacts"); out.mkdir(exist_ok=True, parents=True)
train_df.to_csv(out/"train.csv", index=False)
test_df.to_csv(out/"test.csv", index=False)

print("[data_prep] login + upload to HF dataset hub")
login(HF_TOKEN)
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
api.upload_file(path_or_fileobj=str(out/"train.csv"), path_in_repo="data/train.csv",
                repo_id=DATASET_REPO, repo_type="dataset")
api.upload_file(path_or_fileobj=str(out/"test.csv"),  path_in_repo="data/test.csv",
                repo_id=DATASET_REPO, repo_type="dataset")
print("[data_prep] done")
