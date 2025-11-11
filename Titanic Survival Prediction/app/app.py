import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re

# === Absolute Paths (Local System Only) ===
MODEL_PATH = r"C:\Users\Asus\Desktop\Titanic\Jupyter Notebook\models\titanic_final.pkl"
PREPROCESSOR_PATH = r"C:\Users\Asus\Desktop\Titanic\Jupyter Notebook\models\preprocessor.pkl"

# === Load Model & Preprocessor ===
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        st.success("Model and preprocessor loaded successfully!")
        return model, preprocessor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Check file paths and model files.")
        st.stop()

final_model, preprocessor = load_models()

# === Feature Engineering (Same as Notebook) ===
def add_features(df):
    df = df.copy()
    
    # 1. Convert empty cabin to np.nan
    df['Cabin'] = df['Cabin'].replace('', np.nan).replace(' ', np.nan)
    
    # 2. FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 3. IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4. Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].fillna('Unknown')
    
    # 5. Deck from Cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')
    
    # 6. Age_missing indicator
    df['Age_missing'] = df['Age'].isnull().astype(int)
    
    return df

# === App Config ===
st.set_page_config(page_title="Titanic Survival", page_icon="ship", layout="centered")
st.title("Will you survive the Titanic? ship:")

st.markdown("""
Predict survival based on passenger details.  
Uses **XGBoost + SMOTE + SHAP** from Kaggle model.
""")

# === Input Form ===
with st.form("titanic_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1=First, 2=Second, 3=Third")
        name = st.text_input("Name", "John Doe")
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)

    with col2:
        parch = st.slider("Parents/Children Aboard", 0, 6, 0)
        ticket = st.text_input("Ticket Number", "A/5 21171")
        fare = st.number_input("Fare (£)", 0.0, 1000.0, 32.0, step=0.1)
        cabin = st.text_input("Cabin", "", placeholder="e.g. C85")
        embarked = st.selectbox("Embarked Port", ["S", "C", "Q"], help="S=Southampton, C=Cherbourg, Q=Queenstown")

    submitted = st.form_submit_button("Predict Survival", use_container_width=True)

# === Prediction ===
if submitted:
    # 1. Create raw input DataFrame
    input_data = pd.DataFrame([{
        'Pclass': pclass, 'Name': name, 'Sex': sex, 'Age': age,
        'SibSp': sibsp, 'Parch': parch, 'Ticket': ticket,
        'Fare': fare, 'Cabin': cabin if cabin.strip() else np.nan, 'Embarked': embarked
    }])

    # 2. Apply feature engineering
    input_data = add_features(input_data)

    try:
        # 3. Preprocess
        X_processed = preprocessor.transform(input_data)

        # 4. Predict
        classifier = final_model.named_steps['classifier']
        prediction = classifier.predict(X_processed)[0]
        probability = classifier.predict_proba(X_processed)[0][1]

        # 5. Display result
        st.divider()
        if prediction == 1:
            st.success("**Survived!** thumbsup:")
            st.balloons()
        else:
            st.error("**Did not survive.** thumbsdown:")

        st.metric("Survival Probability", f"{probability:.1%}")
        st.caption("Top influencing factors: Sex > Title > Pclass > Fare > Age")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check input values and model files.")