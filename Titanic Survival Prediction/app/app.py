import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# === App Config (MUST BE FIRST) ===
st.set_page_config(page_title="Titanic Survival", page_icon="ship", layout="centered")

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_final.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# === Load Models ===
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.info("Ensure `models/` folder is in repo root.")
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
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs').fillna('Unknown')
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')
    df['Age_missing'] = df['Age'].isnull().astype(int)
    return df

# === UI ===
st.title("Will you survive the Titanic? ship:")

st.markdown("""
Predict survival using **XGBoost + SMOTE + PyOD**.  
**Kaggle Score: ~0.82** | Interactive Web App
""")

# === Form (Submit Button INSIDE) ===
with st.form("titanic_form"):
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        name = st.text_input("Name", "John Doe")
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
    with col2:
        parch = st.slider("Parents/Children", 0, 6, 0)
        ticket = st.text_input("Ticket", "A/5 21171")
        fare = st.number_input("Fare (£)", 0.0, 1000.0, 32.0, step=0.1)
        cabin = st.text_input("Cabin", "", placeholder="e.g. C85")
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    
    # SUBMIT BUTTON INSIDE FORM
    submitted = st.form_submit_button("Predict Survival", use_container_width=True)

# === Prediction ===
if submitted:
    input_data = pd.DataFrame([{
        'Pclass': pclass, 'Name': name, 'Sex': sex, 'Age': age,
        'SibSp': sibsp, 'Parch': parch, 'Ticket': ticket,
        'Fare': fare, 'Cabin': cabin if cabin.strip() else np.nan, 'Embarked': embarked
    }])
    
    input_data = add_features(input_data)
    
    try:
        X_processed = preprocessor.transform(input_data)
        classifier = final_model.named_steps['classifier']
        prediction = classifier.predict(X_processed)[0]
        probability = classifier.predict_proba(X_processed)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success("**Survived!** thumbsup:")
            st.balloons()
        else:
            st.error("**Did not survive.** thumbsdown:")
        
        st.metric("Survival Probability", f"{probability:.1%}")
        st.caption("Key factors: Sex > Title > Pclass > Fare > Age")
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
