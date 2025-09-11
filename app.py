import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download, login

# MUST be the first Streamlit command
st.set_page_config(page_title="Tourism Wellness Package Predictor", layout="centered")

# ----------------------------
# HF auth & writable cache (/tmp)
# ----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # Space secret if needed
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "dhani10/tourism-model")
MODEL_FILE = os.getenv("MODEL_FILE", "model/best_model.joblib")

HF_CACHE_ROOT = os.getenv("HF_HOME", "/tmp/huggingface")
os.environ["HF_HOME"] = HF_CACHE_ROOT
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_ROOT, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_ROOT, "transformers")
for d in (HF_CACHE_ROOT, os.environ["HF_HUB_CACHE"], os.environ["TRANSFORMERS_CACHE"]):
    os.makedirs(d, exist_ok=True)

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

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

# ----------------------------
# Helper: read expected input columns from preprocessor
# ----------------------------
def get_expected_input_columns(clf):
    pre = clf.named_steps.get("preprocessor")
    cols = []
    if pre is None:
        return cols
    transformers = getattr(pre, "transformers", None) or getattr(pre, "transformers_", [])
    for _, _, selected in transformers:
        if selected in (None, "drop"):
            continue
        if isinstance(selected, list):
            cols.extend(selected)
        elif isinstance(selected, (tuple, np.ndarray, pd.Index)):
            cols.extend(list(selected))
    # preserve order + unique
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

EXPECTED_COLS = get_expected_input_columns(model)

# ----------------------------
# Streamlit UI (collect ALL training features)
# ----------------------------
st.set_page_config(page_title="Tourism Wellness Package Predictor", layout="centered")
st.title("Wellness Tourism Package Predictor")
st.caption("Fill in customer details to predict purchase likelihood.")

# Categorical options — match dataset spellings exactly
TYPE_OF_CONTACT_OPTS = ["Company Invited", "Self Enquiry"]  # <- dataset text
OCCUPATION_OPTS      = ["Salaried", "Small Business", "Freelancer", "Large Business", "Other"]
GENDER_OPTS          = ["Male", "Female"]
PRODUCT_PITCHED_OPTS = ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]  # dataset list
MARITAL_STATUS_OPTS  = ["Single", "Married", "Divorced", "Unmarried"]           # dataset used "Unmarried"
DESIGNATION_OPTS     = ["Executive", "Senior Executive", "Manager", "Senior Manager", "AVP", "VP", "Director", "Junior Executive"]

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
    # Build UI row (exact training column names)
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

    # Fallback if EXPECTED_COLS couldn't be read for some reason
    base_cols = EXPECTED_COLS if EXPECTED_COLS else list(ui_row.keys())

    # Start with expected columns set to NaN, then overlay UI values
    template = {c: [np.nan] for c in base_cols}
    row = pd.DataFrame(template)
    for k, v in ui_row.items():
        if k in row.columns:
            row.at[0, k] = v

    # Coerce numerics (safeguard)
    numeric_cols = [
        "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
        "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
        "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
    ]
    for c in numeric_cols:
        if c in row.columns:
            row[c] = pd.to_numeric(row[c], errors="coerce")

    # Optional: if your pipeline didn't add imputers, simple fill for numerics
    for c in numeric_cols:
        if c in row.columns and pd.isna(row.at[0, c]):
            row.at[0, c] = 0

    try:
        pred = model.predict(row)[0]
        proba = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None

        st.subheader("Result")
        if pred == 1:
            st.success(f"Likely to purchase (confidence: {proba:.2f})" if proba is not None else "Likely to purchase")
        else:
            st.error(f"Not likely to purchase (confidence: {1 - proba:.2f})" if proba is not None else "Not likely to purchase")

        with st.expander("Inputs sent to model"):
            st.dataframe(row)

        if not EXPECTED_COLS:
            st.info("Note: EXPECTED_COLS could not be read from the pipeline; used UI keys as fallback.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Debug"):
            st.write("Expected columns:", EXPECTED_COLS)
            st.dataframe(row)
