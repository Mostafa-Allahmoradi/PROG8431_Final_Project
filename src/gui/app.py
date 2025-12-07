import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score
)

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Obesity Risk ML App",
    layout="wide",
    initial_sidebar_state="expanded", 
    menu_items={
        'About': "This app performs data analysis and machine learning classification to predict obesity risk based on meal macros and lifestyle data."
        "\n\n\nGroup members: "
        "\n\nMostafa Allahmoradi - 9087818 "
        "\n\nCemil Caglar Yapici - 9081058 "
        "\n\nJarius Bedward - 8841640"
    }
)
# -------------------------------------------------------------------------
# Sidebar Navigation & Configuration
# -------------------------------------------------------------------------
st.sidebar.title("Navigation")

# Main Mode Selection
app_mode = st.sidebar.radio("Go to:", ["Data Analysis", "Machine Learning"])

# -------------------------------------------------------------------------
# Data Loading & Engineering Logic
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    try:
        # Tries to look in a data folder, falls back to root
        try:
            df = pd.read_csv('./data/raw/detailed_meals_macros_.csv')
        except FileNotFoundError:
            df = pd.read_csv('detailed_meals_macros_.csv')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # --- Data Cleaning ---
    df = df.interpolate()
    df.columns = [c.strip() for c in df.columns]
    
    if 'Disease' in df.columns:
        df['Disease'] = df['Disease'].str.strip()

    # --- Feature Engineering ---
    if 'Height' in df.columns:
        df['Height_m'] = df['Height'] / 100
        
    if 'Weight' in df.columns and 'Height_m' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
        
    # Target: Obesity
    df['Obesity'] = np.where(df['BMI'] >= 30, 1, 0)
    
    # Activity Numeric Map
    activity_map = {
        "Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2, 
        "Very Active": 3, "Extremely Active": 4
    }
    
    if df['Activity Level'].dtype == 'O':
        df['activity_level_numeric'] = df['Activity Level'].map(activity_map)
    else:
        df['activity_level_numeric'] = df['Activity Level']

    # Weight Gain Risk
    df["Weight_Gain_Risk"] = np.where(
       (df['Calories'] > 2500) & (df['activity_level_numeric'] < 2), 1, 0
    )
    
    return df

df = load_and_prep_data()

# -------------------------------------------------------------------------
# APP MODE: DATA ANALYSIS
# -------------------------------------------------------------------------
if app_mode == "Data Analysis":
    st.title("Exploratory Data Analysis & Statistics")
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Descriptive Stats
        with st.expander("Descriptive Statistics"):
            st.write(df.describe())

        st.markdown("---")
        
        # Hypothesis Testing
        st.header("1. Hypothesis Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fat Intake vs. Obesity")
            group_sick = df[df['Obesity'] == 1]['Fat']
            group_healthy = df[df['Obesity'] == 0]['Fat']
            t_stat, p_val = stats.ttest_ind(group_sick, group_healthy, equal_var=False)
            
            st.write(f"**T-statistic:** {t_stat:.4f}")
            st.write(f"**P-value:** {p_val:.4f}")
            
            fig_fat, ax_fat = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Obesity', y='Fat', data=df, palette='Set2', hue='Obesity', legend=False, ax=ax_fat)
            st.pyplot(fig_fat)

        with col2:
            st.subheader("Calorie Intake vs. Obesity")
            group_sick_cal = df[df['Obesity'] == 1]['Calories']
            group_healthy_cal = df[df['Obesity'] == 0]['Calories']
            t_stat_cal, p_val_cal = stats.ttest_ind(group_sick_cal, group_healthy_cal, equal_var=False)
            
            st.write(f"**T-statistic:** {t_stat_cal:.4f}")
            st.write(f"**P-value:** {p_val_cal:.4f}")
            
            fig_cal, ax_cal = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Obesity', y='Calories', data=df, palette='Set2', hue='Obesity', legend=False, ax=ax_cal)
            st.pyplot(fig_cal)

        st.markdown("---")
        
        # Correlation
        st.header("2. Correlation Analysis")
        numerical_df = df.select_dtypes(include=[np.number])
        corr_matrix = numerical_df.corr(method='pearson')
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax_corr)
        st.pyplot(fig_corr)

# -------------------------------------------------------------------------
# APP MODE: MACHINE LEARNING
# -------------------------------------------------------------------------
elif app_mode == "Machine Learning":
    st.title("Machine Learning Classification")
    
    # --- Sidebar Model Selection ---
    st.sidebar.markdown("### Choose Algorithm")
    model_selection = st.sidebar.radio(
        "Select Classifier:",
        [
            "Logistic",
            "K-NN",
            "Support Vector Machine",
            "Decision Trees",
            "Naive Bayes",
            "Random Forest"
        ]
    )

    if df is not None:
        # Data Preparation
        target_col = 'Obesity'
        X = df.drop([target_col], axis=1)
        y = df[target_col]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocessing Pipelines
        num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) 
        ])

        # Note: sparse_threshold=0 ensures dense output for Naive Bayes compatibility
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols)
            ],
            sparse_threshold=0 
        )

        # --- Model Instantiation Logic ---
        if model_selection == "Logistic":
            clf = LogisticRegression(max_iter=1000)
            st.subheader("Logistic Regression")
            
        elif model_selection == "K-NN":
            k_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
            clf = KNeighborsClassifier(n_neighbors=k_neighbors)
            st.subheader(f"K-Nearest Neighbors (K={k_neighbors})")
            
        elif model_selection == "Support Vector Machine":
            # probability=True is needed for ROC Curve
            clf = SVC(probability=True, kernel='linear') 
            st.subheader("Support Vector Machine (Linear Kernel)")
            
        elif model_selection == "Decision Trees":
            max_d = st.sidebar.slider("Max Depth", 1, 20, 5)
            clf = DecisionTreeClassifier(max_depth=max_d, random_state=42)
            st.subheader(f"Decision Tree (Max Depth={max_d})")
            
        elif model_selection == "Naive Bayes":
            clf = GaussianNB()
            st.subheader("Gaussian Naive Bayes")
            
        elif model_selection == "Random Forest":
            n_est = st.sidebar.slider("Number of Trees", 10, 200, 100)
            clf = RandomForestClassifier(n_estimators=n_est, random_state=42)
            st.subheader(f"Random Forest (Trees={n_est})")

        # --- Training and Evaluation Pipeline ---
        full_pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", clf)
        ])

        # Train
        with st.spinner(f"Training {model_selection}..."):
            full_pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = full_pipeline.predict(X_test)
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]

        # Results Display
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc:.2%}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap='Blues')
            st.pyplot(fig_cm)

        with col2:
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='darkorange', lw=2)
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color='navy')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        st.markdown("#### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

else:
    st.warning("Please verify dataset availability.")