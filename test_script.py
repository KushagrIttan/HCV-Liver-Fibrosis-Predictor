import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/home/allectron/Documents/ml project/hepatitis+c+virus+hcv+for+egyptian+patients/HCV-Egy-Data.csv')
df.columns = df.columns.str.strip()

# Target and features
X = df.drop(columns=['Baselinehistological staging'])
y = df['Baselinehistological staging']
# Map 1-4 to 0-3 for xgboost and standard classification
y = y - y.min() 

# Define column groups
numeric_cols = ['Age', 'BMI']
binary_cols = ['Gender', 'Fever', 'Nausea/Vomting', 'Headache', 'Diarrhea', 'Fatigue & generalized bone ache', 'Jaundice', 'Epigastric pain']

# For custom binning we will use a Custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin

class ExpertBinner(BaseEstimator, TransformerMixin):
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

binned_cols = [c for c in X.columns if c not in numeric_cols and c not in binary_cols and c != 'Baseline histological Grading']
pass_through_cols = ['Baseline histological Grading']

# Task 1: Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('bin_cat', OrdinalEncoder(), binary_cols),
        ('expert_bin', ExpertBinner(feature_names=binned_cols), binned_cols),
        ('pass', 'passthrough', pass_through_cols)
    ],
    remainder='drop'
)

# Task 2 & 3: Models and Evaluation
# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit transform the preprocessing
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Dictionary to store results
class_results = []

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

print("--- Classification Results ---")
for name, clf in classifiers.items():
    clf.fit(X_train_prep, y_train)
    y_pred = clf.predict(X_test_prep)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    
class_df = pd.DataFrame(class_results)
print(class_df.to_string(index=False))

print("\n--- Clustering Results ---")
# For clustering, we use the whole processed dataset without target
X_all_prep = preprocessor.fit_transform(X)

clusterers = {
    'K-Means': KMeans(n_clusters=4, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=4),
    'DBSCAN': DBSCAN(eps=3.0, min_samples=5)
}

cluster_results = []
for name, clu in clusterers.items():
    labels = clu.fit_predict(X_all_prep)
    # DBSCAN might assign all to noise (-1)
    if len(set(labels)) > 1:
        sil = silhouette_score(X_all_prep, labels)
        db = davies_bouldin_score(X_all_prep, labels)
    else:
        sil, db = None, None
    cluster_results.append({'Model': name, 'Silhouette Score': sil, 'Davies-Bouldin Index': db})

cluster_df = pd.DataFrame(cluster_results)
print(cluster_df.to_string(index=False))

print("\n--- Plotting Dendrogram for Agglomerative Hierarchical ---")
linked = linkage(X_all_prep[:100], 'ward') # Plotting subset for visibility
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title("Hierarchical Clustering Dendrogram (Subset)")
plt.savefig('/home/allectron/Documents/ml project/dendrogram.png')
print("Saved dendrogram.png")

print("\n--- Feature Importance (Random Forest) ---")
# Getting feature names after transformation
# numeric_cols + binary_cols + binned_cols + pass_through_cols
all_features = numeric_cols + binary_cols + binned_cols + pass_through_cols
rf_model = classifiers['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

top_5_indices = indices[:5]
top_5_features = [all_features[i] for i in top_5_indices]
top_5_importances = importances[top_5_indices]

plt.figure(figsize=(8, 5))
plt.barh(top_5_features[::-1], top_5_importances[::-1], color='skyblue')
plt.title("Top 5 Clinical Features - Random Forest")
plt.xlabel("Feature Importance")
plt.savefig('/home/allectron/Documents/ml project/rf_importance.png')
print("Saved rf_importance.png")
