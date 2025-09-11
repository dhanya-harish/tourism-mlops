import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, login

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config MUST be the very first Streamlit call on the page
# (use a guard so it only runs once, even on reruns).
# ──────────────────────────────────────────────────────────────────────────────
if "_page_config_set" not in st.session_state:
    st.set_page_config(page_title="Tourism Wellness Package Predictor", layout="centered")
    st.session_state["_page_config_set"] = True

# ──────────────────────────────────────────────────────────────────────────────
# HF Hub config & auth
# ──────────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")  # optional if model repo is public
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "dhani10/tourism-model")
MODEL_FILE = os.getenv("MODEL_FILE", "model/best_model.joblib")

# Writable caches on Spaces
HF_CACHE_ROOT = os.getenv("HF_HOME", "/tmp/huggingface")
os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_ROOT, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_ROOT, "transformers")
for d in (HF_CACHE_ROOT, os.environ["HF_HUB_CACHE"], os.environ["TRANSFORMERS_CACHE"]):
    os.makedirs(d, exist_ok=True)

# Login if token present (private repos)
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Load model from the Hub (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILE,
        repo_type="model",
        token=HF_TOKEN,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    return joblib.load(local_path)

model = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Helper: get the raw input feature names the ColumnTransformer expects
# ──────────────────────────────────────────────────────────────────────────────
def get_expected_input_columns(clf):
    pre = clf.named_steps.get("preprocessor")
    cols = []
    if pre is None:
        return cols
    # Works both before and after fit
    transformers = getattr(pre, "transformers", None) or getattr(pre, "transformers_", [])
    for _, _, selected in transformers:
        if selected in (None, "drop"):
            continue
        if isinstance(selected, list):
            cols.extend(selected)
        elif isinstance(selected, (tuple, np.ndarray, pd.Index)):
            cols.extend(list(selected))
    # unique, preserve order
    return list(dict.fromkeys(cols))

EXPECTED_COLS = get_expected_input_columns(model)

# Known categorical feature names from your dataset
CAT_FEATURES = {
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
}
# Reasonable defaults for features we don't expose explicitly
CAT_DEFAULT = "Unknown"
NUM_DEFAULT = 0

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Tourism Wellness Package Predictor")
st.caption("Fill in customer details to predict purchase likelihood.")

# Categorical options (adjust if your dataset vocabulary differs)
TYPE_OF_CONTACT_OPTS = ["Company Invited", "Self Inquiry"]
OCCUPATION_OPTS      = ["Salaried", "Freelancer", "Other"]
GENDER_OPTS          = ["Male", "Female"]
PRODUCT_PITCHED_OPTS = ["Basic", "Deluxe", "King", "Standard", "Super Deluxe", "Elite"]
MARITAL_STATUS_OPTS  = ["Single", "Married", "Divorced"]
DESIGNATION_OPTS     = ["Executive", "Manager", "Senior Manager", "AVP", "VP"]

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30)
        TypeofContact = st.selectbox("Type of Contact", TYPE_OF_CONTACT_OPTS)
        CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
        DurationOfPitch = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=600, value=10)
        Occupation = st.selectbox("Occupation", OCCUPATION_OPTS)
        Gender = st.selectbox("Gender", GENDER_OPTS)
        NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=0, max_value=20, value=1)
        NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=50, value=2)

    with col2:
        ProductPitched = st.selectbox("Product Pitched", PRODUCT_PITCHED_OPTS)
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=3)
        MaritalStatus = st.selectbox("Marital Status", MARITAL_STATUS_OPTS)
        NumberOfTrips = st.number_input("Number Of Trips (year)", min_value=0, max_value=50, value=2)
        Passport = st.selectbox("Has Passport?", [0, 1], index=0)
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        OwnCar = st.selectbox("Owns Car?", [0, 1], index=0)
        NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (≤5y)", min_value=0, max_value=10, value=0)
        Designation = st.selectbox("Designation", DESIGNATION_OPTS)
        MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=10_000_000, value=50_000, step=1_000)

    submitted = st.form_submit_button("Predict")

if submitted:
    # User-provided features
    ui_row = {
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": int(CityTier),
        "DurationOfPitch": float(DurationOfPitch),
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": int(NumberOfPersonVisiting),
        "NumberOfFollowups": int(NumberOfFollowups),
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": int(PreferredPropertyStar),
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": int(NumberOfTrips),
        "Passport": int(Passport),
        "PitchSatisfactionScore": int(PitchSatisfactionScore),
        "OwnCar": int(OwnCar),
        "NumberOfChildrenVisiting": int(NumberOfChildrenVisiting),
        "Designation": Designation,
        "MonthlyIncome": float(MonthlyIncome),
    }

    # Build a 1-row frame with EXACTLY the expected columns:
    # 1) Start from defaults (avoid NaNs)
    defaults = {}
    for c in EXPECTED_COLS:
        if c in CAT_FEATURES:
            defaults[c] = CAT_DEFAULT
        else:
            defaults[c] = NUM_DEFAULT
    row = pd.DataFrame({k: [v] for k, v in defaults.items()})

    # 2) Overlay user inputs where available
    for k, v in ui_row.items():
        if k in row.columns:
            row.at[0, k] = v

    try:
        pred = model.predict(row)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(row)[0, 1])

        st.subheader("Result")
        if pred == 1:
            st.success(f"Likely to purchase (confidence: {proba:.2f})" if proba is not None else "Likely to purchase")
        else:
            st.error(f"Not likely to purchase (confidence: {1 - proba:.2f})" if proba is not None else "Not likely to purchase")

        with st.expander("Inputs sent to model"):
            st.write(row)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Debug: expected raw feature names"):
            st.write(EXPECTED_COLS)
