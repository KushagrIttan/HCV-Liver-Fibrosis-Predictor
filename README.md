# Predictive Modeling of Liver Fibrosis in Hepatitis C Patients

**Student Name:** Kushagr Ittan[cite: 1]  
**Enrollment No:** 07119051624[cite: 1]  
**Specialization:** B.Tech AIML (USAR, GGSIPU)[cite: 1]

## 🏥 Overview
This project implements a non-invasive methodology for predicting liver fibrosis severity in patients with the Hepatitis C Virus (HCV)[cite: 1]. Utilizing a dataset of 1,385 Egyptian patients, the system applies multiple machine learning paradigms to assist in clinical decision-making[cite: 1].

## 🚀 Key Features
*   **Multi-Model Analysis:** Implements 5 Classification models (Logistic Regression, Random Forest, SVM, KNN, XGBoost) and 3 Clustering models (K-Means, Hierarchical, DBSCAN)[cite: 1].
*   **Explainable AI (XAI):** Integrated **SHAP** values to provide transparency on why the model predicts specific fibrosis stages[cite: 1].
*   **Class Imbalance Handling:** Utilizes **SMOTE** to ensure robust performance across minority disease stages.
*   **Interactive Dashboard:** A **Streamlit**-based web interface for real-time patient risk assessment[cite: 1].

## 📊 Dataset Metadata
*   **Source:** UCI Machine Learning Repository (HCV for Egyptian patients)[cite: 1].
*   **Instances:** 1,385 records[cite: 1].
*   **Features:** 28 clinical and demographic variables including Age, BMI, and 20+ symptoms (Fever, Nausea, Jaundice, etc.)[cite: 1].

## 🛠️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/HCV-Liver-Fibrosis-XAI.git](https://github.com/YOUR_USERNAME/HCV-Liver-Fibrosis-XAI.git)
   cd HCV-Liver-Fibrosis-XAI
   
```
2. **Create a virtual environment (Fedora/Linux):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   streamlit run app.py
   