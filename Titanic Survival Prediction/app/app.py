# app/app.py - Fixed for Streamlit Cloud
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# === App Config===
st.set_page_config(page_title="Titanic Survival", page_icon="ship", layout="centered")

# === Relative Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_final.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# === Load Model & Preprocessor (AFTER set_page_config) ===
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        # st.success() حذف شد — فقط در صورت موفقیت نمایش داده میشه
        return model, preprocessor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Check if `models/` folder and `.pkl` files are in the repository.")
        st.stop()

final_model, preprocessor = load_models()

# === Feature Engineering ===
def add_features(df):
    df = df.copy()
    df['Cabin'] = df['Cabin'].replace('', np.nan).replace(' ', np.nan)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].fillna('Unknown')
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')
    df['Age_missing'] = df['Age'].isnull().astype(int)
    return df

# === UI ===
st.title("Will you survive the Titanic? ship:")

st.markdown("""
Predict survival based on passenger details.  
Uses **XGBoost + SMOTE + SHAP** from Kaggle model.  
**Kaggle Score: ~0.82**
""")

with st.form("titanic_form"):
    col1, col2 = st.columns(2)
   
