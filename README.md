# Titanic Survival Prediction
**Will you survive the Titanic?**  
Predict survival with **XGBoost + SMOTE + PyOD** | **Kaggle Score: ~0.82**

---

## Features
### Feature Engineering
- `Title` from Name (Mr, Mrs, Rare...)
- `Deck` from Cabin (A–G, M)
- `FamilySize` & `IsAlone`
- `Age_missing` indicator

### Preprocessing
- Outlier removal with **PyOD (KNN)**
- **SMOTE** for class imbalance
- `StandardScaler` + `OneHotEncoder`

### Modeling
- **XGBoost** with `GridSearchCV`
- Full `Pipeline` + `ColumnTransformer`

### Deployment
- **Streamlit WebApp** (Interactive Prediction)
- Docker-ready
- Kaggle submission ready

---

## Project Phases

| # | Phase                     | Category       | Required | Status   | Tools                              |
|---|---------------------------|----------------|----------|----------|------------------------------------|
| 1 | Import Libs               | Preparation    | Yes      | Done     | `pandas, sklearn, plotly, shap, ...` |
| 2 | Import Dataset            | Preparation    | Yes      | Done     | `pd.read_csv()`                    |
| 3 | Data Leakage Check        | Preparation    | Yes      | Done     | `assert`                           |
| 4 | Pandas Profiling          | EDA            | No       | Done     | `ProfileReport()`                  |
| 5 | Correlation + Heatmap     | EDA            | Yes      | Done     | `df.corr(), plotly`                |
| 6 | Plotly Visualization      | EDA            | Yes      | Done     | `px.scatter, px.box`               |
| 7 | Feature Engineering       | EDA            | Yes      | Done     | `log, ratio, binning`              |
| 8 | Missing Value + Indicator | Preprocessing  | Yes      | Done     | `SimpleImputer(add_indicator=True)`|
| 9 | Outlier Handling (PyOD)   | Preprocessing  | Yes      | Done     | `KNN(), IsolationForest`           |
|10 | Encoding                  | Preprocessing  | Yes      | Done     | `OneHotEncoder(drop='first')`      |
|11 | Scaling                   | Preprocessing  | Yes      | Done     | `StandardScaler()`                 |
|12 | Sampling (SMOTE)          | Preprocessing  | No       | Done     | `SMOTE()`                          |
|13 | Dimensionality Reduction  | Preprocessing  | No       | Skipped  | `PCA(n_components=0.95)`           |
|14 | Pipeline + ColumnTransformer | Preprocessing | Yes      | Done     | `Pipeline, ColumnTransformer`       |
|15 | Train/Test Split + Stratify | Preprocessing | Yes      | Done     | `stratify=y`                       |
|16 | PyCaret (AutoML)          | Modeling       | No       | Skipped  | `setup(), compare_models()`        |

---

## WebApp (Live Demo)
[Try it now!](https://titanic-survival-prediction-dev-monib.streamlit.app)  
*(Interactive survival prediction)*

---

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app/app.py
