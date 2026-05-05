import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="HCV Fibrosis Predictor", layout="wide", page_icon="🧬")

# ---------------------------------------------------------
# DATA PREPROCESSING LOGIC
# ---------------------------------------------------------

class ExpertBinner(BaseEstimator, TransformerMixin):
    """
    Applies medical discretization/binning criteria.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_binned = pd.DataFrame(X, columns=self.feature_names).copy()
        
        for c in self.feature_names:
            if c == 'WBC':
                X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 4000, 11000, np.inf], labels=False)
            elif c == 'RBC':
                X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 3000000, 5000000, np.inf], labels=False)
            elif c == 'Plat':
                X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 100000, 255000, np.inf], labels=False)
            elif 'AST' in c or 'ALT' in c:
                X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 20, 40, np.inf], labels=False)
            elif 'RNA' in c:
                X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 5, np.inf], labels=False)
            elif c == 'HGB':
                X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 12.3, 15.3, np.inf], labels=False)
                
        return X_binned.fillna(0).values

@st.cache_resource(show_spinner=True)
def load_data_and_train_models():
    # Load
    df = pd.read_csv('hepatitis+c+virus+hcv+for+egyptian+patients/HCV-Egy-Data.csv')
    df.columns = df.columns.str.strip()
    
    X = df.drop(columns=['Baselinehistological staging'])
    y = df['Baselinehistological staging']
    y = y - y.min() # Align labels 0-3 for XGBoost
    
    # Feature grouping
    numeric_cols = ['Age', 'BMI']
    binary_cols = ['Gender', 'Fever', 'Nausea/Vomting', 'Headache', 'Diarrhea', 
                   'Fatigue & generalized bone ache', 'Jaundice', 'Epigastric pain']
    pass_through_cols = ['Baseline histological Grading']
    binned_cols = [c for c in X.columns if c not in numeric_cols + binary_cols + pass_through_cols]
    
    feature_names = numeric_cols + binary_cols + binned_cols + pass_through_cols
    
    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('bin_cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), binary_cols),
            ('expert_bin', ExpertBinner(feature_names=binned_cols), binned_cols),
            ('pass', 'passthrough', pass_through_cols)
        ],
        remainder='drop'
    )
    
    X_prep = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
    
    # SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # XGBoost with RandomizedSearchCV
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    search = RandomizedSearchCV(xgb, param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    search.fit(X_train_sm, y_train_sm)
    best_model = search.best_estimator_
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42).fit(X_prep)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(best_model)
    
    # PCA for clustering visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_prep)
    
    return preprocessor, best_model, kmeans, explainer, feature_names, X_pca, kmeans.labels_, pca

# ---------------------------------------------------------
# UI SETUP & SIDEBAR
# ---------------------------------------------------------

st.title("🧬 Predictive Modeling of Liver Fibrosis in Hepatitis C Patients")
st.markdown("""
Welcome to the clinical inference dashboard. This tool integrates **XGBoost** with **SMOTE** balancing to predict fibrosis staging based on patient biomarkers, providing real-time SHAP explainability and cluster segmentation.
""")

with st.spinner("Loading models and preprocessing pipeline..."):
    preprocessor, best_model, kmeans, explainer, feature_names, X_pca, cluster_labels, pca_model = load_data_and_train_models()

st.sidebar.header("Patient Clinical Inputs")

def get_binary_val(label):
    val = st.sidebar.radio(label, ["Absent", "Present"], index=0)
    return 1 if val == "Absent" else 2

# Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.number_input("Age", 1, 100, 45)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 2
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)

# Clinical Symptoms
st.sidebar.subheader("Clinical Symptoms")
fever = get_binary_val("Fever")
nausea = get_binary_val("Nausea/Vomiting")
headache = get_binary_val("Headache")
diarrhea = get_binary_val("Diarrhea")
fatigue = get_binary_val("Fatigue & generalized bone ache")
jaundice = get_binary_val("Jaundice")
epigastric = get_binary_val("Epigastric pain")

# Blood & Liver Panels
st.sidebar.subheader("Blood & Liver Panels")
wbc = st.sidebar.number_input("WBC", value=6000)
rbc = st.sidebar.number_input("RBC", value=4000000)
hgb = st.sidebar.number_input("HGB", value=13.0)
plat = st.sidebar.number_input("Platelets", value=150000)

st.sidebar.subheader("ALT / AST Panels")
ast1 = st.sidebar.number_input("AST 1", value=25)
alt1 = st.sidebar.number_input("ALT 1", value=25)
alt4 = st.sidebar.number_input("ALT 4", value=25)
alt12 = st.sidebar.number_input("ALT 12", value=25)
alt24 = st.sidebar.number_input("ALT 24", value=25)
alt36 = st.sidebar.number_input("ALT 36", value=25)
alt48 = st.sidebar.number_input("ALT 48", value=25)
alt24w = st.sidebar.number_input("ALT after 24 w", value=25)

st.sidebar.subheader("Viral RNA Load")
rna_base = st.sidebar.number_input("RNA Base", value=500000)
rna_4 = st.sidebar.number_input("RNA 4", value=500000)
rna_12 = st.sidebar.number_input("RNA 12", value=500000)
rna_eot = st.sidebar.number_input("RNA EOT", value=500000)
rna_ef = st.sidebar.number_input("RNA EF", value=500000)

grading = st.sidebar.slider("Baseline histological Grading", 1, 16, 5)

# Assemble DataFrame
input_dict = {
    'Age': age, 'Gender': gender_val, 'BMI': bmi, 'Fever': fever, 
    'Nausea/Vomting': nausea, 'Headache': headache, 'Diarrhea': diarrhea, 
    'Fatigue & generalized bone ache': fatigue, 'Jaundice': jaundice, 
    'Epigastric pain': epigastric, 'WBC': wbc, 'RBC': rbc, 'HGB': hgb, 
    'Plat': plat, 'AST 1': ast1, 'ALT 1': alt1, 'ALT4': alt4, 'ALT 12': alt12, 
    'ALT 24': alt24, 'ALT 36': alt36, 'ALT 48': alt48, 'ALT after 24 w': alt24w, 
    'RNA Base': rna_base, 'RNA 4': rna_4, 'RNA 12': rna_12, 
    'RNA EOT': rna_eot, 'RNA EF': rna_ef, 'Baseline histological Grading': grading
}
user_df = pd.DataFrame([input_dict])

# ---------------------------------------------------------
# TABS AND INFERENCE
# ---------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔍 Analysis (SHAP)", "🔬 Research (Clustering)"])

# Preprocess user input
user_prep = preprocessor.transform(user_df)
pred = best_model.predict(user_prep)[0]
proba = best_model.predict_proba(user_prep)[0]
confidence = proba[pred] * 100

with tab1:
    st.subheader(f"Predicted Fibrosis Stage: **F{pred + 1}**")
    st.progress(float(confidence / 100.0))
    st.write(f"**Confidence Score:** {confidence:.2f}%")
    
    st.markdown("### Multi-Class Probability Distribution")
    prob_df = pd.DataFrame({
        "Stage": ["F1", "F2", "F3", "F4"],
        "Probability": proba
    })
    st.bar_chart(prob_df.set_index("Stage"))

with tab2:
    st.subheader("SHAP Local Explainability")
    st.write(f"Visualizing the feature impacts for the predicted stage **F{pred + 1}**.")
    
    shap_values = explainer(user_prep)
    
    try:
        # Extract SHAP values for the specific prediction class
        shap_val_pred = shap_values[0, :, pred]
        shap_val_pred.feature_names = feature_names
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_val_pred, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating SHAP waterfall plot: {e}")

with tab3:
    st.subheader("Patient Segmentation & Clustering")
    st.write("Visualizing K-Means clusters to show patient segmentation.")
    
    user_cluster = kmeans.predict(user_prep)[0]
    user_pca = pca_model.transform(user_prep)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5, label='Historical Patients')
    
    # Overlay current patient
    ax2.scatter(user_pca[:, 0], user_pca[:, 1], color='red', marker='X', s=250, edgecolor='white', label=f'Current Patient (Cluster {user_cluster})')
    
    plt.colorbar(scatter, label='Cluster Segment')
    plt.legend()
    plt.title("Patient Segments (PCA Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(fig2)
