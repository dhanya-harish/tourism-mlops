#!/usr/bin/env python3
"""
Deploy a Streamlit Space that loads a scikit-learn pipeline from the HF Hub.

Env vars expected (aligned with mlops_pipeline.yml):
  HF_TOKEN    (required)   - HF access token (GitHub Secret)
  SPACE_ID    (required)   - e.g. "username/space-name"
  MODEL_REPO  (required)   - e.g. "username/model-repo"
  MODEL_FILE  (optional)   - path in model repo (default: "model/best_model.joblib")
  UPLOAD_DIR  (optional)   - folder to upload to Space (default: "space/" if exists else repo root)
"""

import os
import sys
from pathlib import Path
from textwrap import dedent
from string import Template
from huggingface_hub import login, HfApi, create_repo, upload_folder
from huggingface_hub.utils import HfHubHTTPError

# ---------------------------
# Read environment variables
# ---------------------------
HF_TOKEN   = os.getenv("HF_TOKEN")
SPACE_ID   = os.getenv("SPACE_ID")          # e.g., "dhani10/tourism-app"
MODEL_REPO = os.getenv("MODEL_REPO") or os.getenv("MODEL_REPOID")
MODEL_FILE = os.getenv("MODEL_FILE", "model/best_model.joblib")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")        # optional

if not HF_TOKEN:
    sys.exit("HF_TOKEN env var is required (GitHub Secret).")
if not SPACE_ID:
    sys.exit("SPACE_ID env var is required (e.g., 'username/space-name').")
if not MODEL_REPO:
    sys.exit("MODEL_REPO env var is required (e.g., 'username/model-repo').")

print(f"[deploy] SPACE_ID={SPACE_ID}")
print(f"[deploy] MODEL_REPO={MODEL_REPO}")
print(f"[deploy] MODEL_FILE={MODEL_FILE}")

# ---------------------------
# Login and API client
# ---------------------------
print("[deploy] Logging in to Hugging Face…")
login(HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# ---------------------------
# Pick upload directory
# ---------------------------
repo_root = Path(".").resolve()
default_space_dir = repo_root / "space"
if UPLOAD_DIR:
    upload_dir = Path(UPLOAD_DIR).resolve()
elif default_space_dir.exists():
    upload_dir = default_space_dir
else:
    upload_dir = repo_root

print(f"[deploy] Using upload folder: {upload_dir}")

# ---------------------------
# Ensure minimal app files exist (if not provided)
# ---------------------------
app_py = upload_dir / "app.py"
reqs   = upload_dir / "requirements.txt"

if not app_py.exists():
    print("[deploy] 'app.py' not found. Generating a minimal Streamlit app that loads the model from the Hub…")
    app_code = Template(dedent(r"""
        import os, joblib, pandas as pd, streamlit as st
        from huggingface_hub import hf_hub_download, login

        st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")
        st.title("🌍 Wellness Tourism Package Purchase Predictor")

        HF_TOKEN = os.getenv("HF_TOKEN")
        HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "$MODEL_REPO")
        MODEL_FILE = os.getenv("MODEL_FILE", "$MODEL_FILE")

        # Writable cache on Spaces
        os.environ.setdefault("HF_HOME", "/tmp/huggingface")
        os.environ.setdefault("HF_HUB_CACHE", "/tmp/huggingface/hub")
        os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

        if HF_TOKEN:
            try:
                login(HF_TOKEN)
            except Exception:
                pass

        @st.cache_resource
        def load_model():
            p = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=MODEL_FILE,
                repo_type="model",
                token=HF_TOKEN,
                cache_dir=os.environ["HF_HUB_CACHE"],
            )
            return joblib.load(p)

        model = load_model()

        Age = st.number_input("Age", 18, 100, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        CityTier = st.selectbox("City Tier", [1,2,3])
        Occupation = st.selectbox("Occupation", ["Salaried","Freelancer","Other"])
        Trips = st.slider("Trips / year", 0, 20, 2)
        Passport = st.selectbox("Has Passport?", [0,1], index=0)
        OwnCar = st.selectbox("Owns Car?", [0,1], index=0)
        Income = st.number_input("Monthly Income", 0, 10_000_000, 50_000, step=1_000)

        if st.button("Predict"):
            row = pd.DataFrame([{
                "Age": Age, "Gender": Gender, "CityTier": CityTier,
                "Occupation": Occupation, "NumberOfTrips": Trips,
                "Passport": Passport, "OwnCar": OwnCar, "MonthlyIncome": float(Income)
            }])
            pred = model.predict(row)[0]
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(row)[0,1])
            else:
                proba = None
            if pred == 1:
                st.success(f"✅ Likely to purchase (conf {proba:.2f})" if proba is not None else "✅ Likely to purchase")
            else:
                st.error(f"❌ Not likely (conf {1-proba:.2f})" if proba is not None else "❌ Not likely to purchase")
    """)).substitute(MODEL_REPO=MODEL_REPO, MODEL_FILE=MODEL_FILE)
    app_py.write_text(app_code)

if not reqs.exists():
    print("[deploy] 'requirements.txt' not found. Generating a minimal one…")
    reqs.write_text(dedent("""
        streamlit
        pandas
        numpy
        scikit-learn
        joblib
        huggingface_hub
    """).strip() + "\n")

# ---------------------------
# Create/ensure the Space
# ---------------------------
print("[deploy] Ensuring Space exists…")
try:
    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        private=True,
        exist_ok=True,
        token=HF_TOKEN,
        # do not pass space_sdk (older backends may reject it)
    )
except HfHubHTTPError as e:
    print(f"[deploy][warn] create_repo returned: {e}")

# Try to set runtime to Streamlit (best-effort)
try:
    api.update_space_runtime(
        repo_id=SPACE_ID,
        sdk="streamlit",
        hardware="cpu-basic",
        token=HF_TOKEN,
    )
    print("[deploy] Space runtime set to Streamlit (cpu-basic).")
except Exception as e:
    print(f"[deploy][info] Could not explicitly set runtime (continuing): {e}")

# ---------------------------
# Set Space variables / secrets (best-effort)
# ---------------------------
try:
    api.add_space_variable(repo_id=SPACE_ID, key="HF_MODEL_REPO", value=MODEL_REPO)
    api.add_space_variable(repo_id=SPACE_ID, key="MODEL_FILE", value=MODEL_FILE)
    print("[deploy] Set Space variables HF_MODEL_REPO and MODEL_FILE.")
except Exception as e:
    print(f"[deploy][info] Could not set Space variables (continuing): {e}")

try:
    api.add_space_secret(repo_id=SPACE_ID, key="HF_TOKEN", value=HF_TOKEN)
    print("[deploy] Set Space secret HF_TOKEN.")
except Exception as e:
    print(f"[deploy][info] Could not set Space secret HF_TOKEN (continuing): {e}")

# ---------------------------
# Upload files & restart
# ---------------------------
print(f"[deploy] Uploading folder to Space: {upload_dir}")
upload_folder(
    folder_path=str(upload_dir),
    repo_id=SPACE_ID,
    repo_type="space",
    token=HF_TOKEN,
)
print("[deploy] Upload complete.")

try:
    api.restart_space(repo_id=SPACE_ID, token=HF_TOKEN)
    print("[deploy] Space restart triggered.")
except Exception as e:
    print(f"[deploy][info] Could not restart Space (continuing): {e}")

print(f"[deploy] ✅ Deployed to https://huggingface.co/spaces/{SPACE_ID}")
