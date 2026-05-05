import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/home/allectron/Documents/ml project/hepatitis+c+virus+hcv+for+egyptian+patients/HCV-Egy-Data.csv')
df.columns = df.columns.str.strip()

X = df.drop(columns=['Baselinehistological staging'])
y = df['Baselinehistological staging']
y = y - y.min()

numeric_cols = ['Age', 'BMI']
binary_cols = ['Gender', 'Fever', 'Nausea/Vomting', 'Headache', 'Diarrhea', 'Fatigue & generalized bone ache', 'Jaundice', 'Epigastric pain']
pass_through_cols = ['Baseline histological Grading']
binned_cols = [c for c in X.columns if c not in numeric_cols + binary_cols + pass_through_cols]

class ExpertBinner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_binned = pd.DataFrame(X, columns=self.feature_names).copy()
        for c in self.feature_names:
            X_binned[c] = pd.cut(X_binned[c], bins=[-np.inf, 20, 40, np.inf], labels=False)
        return X_binned.fillna(0).values

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('bin_cat', OrdinalEncoder(), binary_cols),
        ('expert_bin', ExpertBinner(feature_names=binned_cols), binned_cols),
        ('pass', 'passthrough', pass_through_cols)
    ]
)

X_prep = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
sample = X_test[0:1]
pred = model.predict(sample)[0]

shap_values = explainer(sample)

try:
    print(shap_values.shape)
    # in newer shap versions for multi-class XGBoost:
    shap_val_pred = shap_values[0, :, pred]
    shap_val_pred.feature_names = numeric_cols + binary_cols + binned_cols + pass_through_cols
    shap.plots.waterfall(shap_val_pred, show=False)
    plt.savefig('/home/allectron/Documents/ml project/test_shap.png')
    print("Waterfall successful")
except Exception as e:
    print("Error:", e)
