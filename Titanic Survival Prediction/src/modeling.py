# src/modeling.py
"""
Titanic Survival Prediction - Full Pipeline
--------------------------------------------
Matches exactly with Titanic.ipynb notebook.
Features: FamilySize, IsAlone, Title, Deck, Age_missing
Preprocessing: Imputation, Outlier Removal (PyOD), Scaling, Encoding
Model: XGBoost + SMOTE + GridSearchCV
Output: models/titanic_final.pkl, models/preprocessor.pkl
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from pyod.models.knn import KNN

# === Paths (Relative) ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# === Load Data ===
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_passenger_id = df_test['PassengerId'].copy()

print(f"Train: {df_train.shape} | Test: {df_test.shape}")

# === Feature Engineering ===
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: FamilySize, IsAlone, Title, Deck"""
    df = df.copy()
    
    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].fillna('Unknown')
    
    # Deck: First letter of Cabin, 'M' for missing
    df['Cabin'] = df['Cabin'].replace('', np.nan).replace(' ', np.nan)
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'M')
    
    return df

df_train = add_features(df_train)
df_test = add_features(df_test)

# === Impute Missing Values (Leak-Proof) ===
def impute_missing(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Impute Age (median), Embarked (mode) from train only"""
    age_median = train_df['Age'].median()
    embarked_mode = train_df['Embarked'].mode()[0]
    
    train = train_df.copy()
    test = test_df.copy()
    
    train['Age_missing'] = train['Age'].isnull().astype(int)
    test['Age_missing'] = test['Age'].isnull().astype(int)
    
    train['Age'] = train['Age'].fillna(age_median)
    test['Age'] = test['Age'].fillna(age_median)
    
    train['Embarked'] = train['Embarked'].fillna(embarked_mode)
    test['Embarked'] = test['Embarked'].fillna(embarked_mode)
    
    return train, test

df_train, df_test = impute_missing(df_train, df_test)

# === Outlier Removal (PyOD KNN) ===
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using KNN on Age and Fare"""
    X_num = df[['Age', 'Fare']]
    detector = KNN(contamination=0.1)
    outliers = detector.fit_predict(X_num)
    df_clean = df[outliers == 0].reset_index(drop=True)
    print(f"Outliers removed: {len(df) - len(df_clean)}")
    return df_clean

df_train = remove_outliers(df_train)

# === Prepare Features ===
X = df_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df_train['Survived']
X_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
num_cols = ['Age', 'Fare', 'FamilySize', 'IsAlone', 'Age_missing']

# === Preprocessor ===
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
], remainder='drop')

# === Train/Val Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Base Pipeline for GridSearch ===
base_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
])

# === Cross Validation ===
scores = cross_val_score(base_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# === Hyperparameter Tuning ===
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best params: {best_params}")

# === Final Model with SMOTE ===
final_model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        **{k.split('__')[1]: v for k, v in best_params.items()}
    ))
])

final_model.fit(X, y)

# === Save Model & Preprocessor ===
joblib.dump(final_model, os.path.join(MODELS_DIR, "titanic_final.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

print("Model and preprocessor saved successfully!")
print(f"   → {os.path.join(MODELS_DIR, 'titanic_final.pkl')}")
print(f"   → {os.path.join(MODELS_DIR, 'preprocessor.pkl')}")