import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

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

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Data_Preprocessing.EDA import NutritionEDA

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
app_mode = st.sidebar.radio("Go to:", ["Overview", "Problem Statement", "Data Analysis", "Machine Learning"])

# -------------------------------------------------------------------------
# Data Loading & Engineering Logic
# -------------------------------------------------------------------------
def load_and_prep_data():
    try:
        # --- Data Cleaning ---
        nutrition_eda.clean_data()

        # --- Feature Engineering ---
        nutrition_eda.perform_feature_engineering()
        df = nutrition_eda.df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
    
    return df

nutrition_eda = NutritionEDA('./data/raw/detailed_meals_macros_.csv')
df = load_and_prep_data()

# -------------------------------------------------------------------------
# APP MODE: Overview
# -------------------------------------------------------------------------
if app_mode == "Overview":
    st.header("Final Project - Obesity Risk Prediction Using Machine Learning")

    if df is not None:  
        st.markdown("""
        **Course:** PROG8431 - Data Analysis Mathematics, Algorithms and Modeling

        **Fall 2025 - Section 1**
        
        **Instructor:** Professor David Espinosa Carrillo and Professor Yun Qian Miao
        
        **Group members:**
        - Mostafa Allahmoradi - 9087818
        - Cemil Caglar Yapici - 9081058
        - Jarius Bedward - 8841640
                    
        #### This application performs data analysis and machine learning classification to predict obesity risk based on daily nutrition and lifestyle data.
        """)

# -------------------------------------------------------------------------
# APP MODE: Problem Statement
# -------------------------------------------------------------------------
if app_mode == "Problem Statement":
    st.header("Problem Statement: Investigating the Relationship Between High Calorie and Fat Intake and Obesity Risk")

    if df is not None:  
        st.markdown("""
        **Area of Focus:** Obesity is one of the most critical global health challenges. Diet quality—specifically the daily consumption of dietary fat and total calorie intake—is widely recognized as a primary determinant of this condition. While higher consumption of fat and calories contributes to obesity risk, the precise relationship between individual intake levels and the likelihood of obesity requires further clarification. Understanding this relationship is essential for designing targeted dietary interventions and improving nutritional guidelines.

        #### Research Question: 
        Is there a statistically significant difference in daily calorie and fat intake between individuals diagnosed with obesity and those without the condition?

        #### Research Hypothesis
        * **Null Hypothesis ($H_0$):** There is no significant difference in the mean daily caloric and fat intake between individuals diagnosed with obesity and those without the condition. 
            $$H_0: \mu_{obesity} \leq \mu_{healthy}$$

        * **Alternative Hypothesis ($H_1$):** Individuals diagnosed with obesity have a significantly higher mean daily caloric and fat intake compared to individuals without the condition. 
            $$H_1: \mu_{obesity} > \mu_{healthy}$$

        #### Research Summary: Diet Quality as a Determinant of Obesity Risk
        Introduction The escalating prevalence of obesity represents one of the most pressing public health challenges of the modern era. While the etiology of obesity is multifactorial—involving genetics, environment, and metabolism—diet quality remains the most modifiable and significant determinant. This research project focuses specifically on the quantitative relationship between dietary habits and obesity. By isolating daily caloric intake and dietary fat consumption as primary variables, this study aims to clarify the extent to which these specific nutritional factors distinguish individuals diagnosed with obesity from those who maintain a healthy weight.
                    
        The Problem Space Despite the general consensus that “overeating” contributes to weight gain, the precise statistical relationship between specific intake thresholds and clinical obesity requires rigorous validation. Public health guidelines often provide generalized advice, but understanding the statistical significance of intake differences is crucial for creating evidence-based interventions. The problem statement highlights a need to move beyond anecdotal assumptions and toward a data-driven understanding of how high consumption of fat and total calories correlates with the likelihood of an obesity diagnosis. This clarification is essential for the development of targeted dietary guidelines that are not only theoretically sound but statistically validated against real-world population data.
        
        Methodological Framework The core of this research is a comparative analysis designed to answer a specific question: Is there a statistically significant difference in daily calorie and fat intake between individuals diagnosed with obesity and those without the condition?

        To answer this, the study employs a classical hypothesis testing framework. This statistical approach allows the research to move beyond simple observation and determine if the differences in diet between the two groups are substantial enough to be considered non-random.            
        
        Hypothesis Analysis The study is grounded in two opposing hypotheses that will be tested against the dataset:
            
        * The Null Hypothesis ($H_0$): This hypothesis assumes the skeptical position. It posits that there is no significant statistical difference in the mean daily caloric and fat intake between obese and non-obese individuals. In statistical terms ($H_0: \mu_{obesity} \leq \mu_{healthy}$), this suggests that the mean intake of the obese population is less than or equal to that of the healthy population. If the data fails to reject this hypothesis, it would suggest that factors other than total calorie and fat volume (such as genetics, metabolic adaptation, or activity levels) may play a larger role in the disease's pathology than simply the amount of food consumed.
            
        * The Alternative Hypothesis ($H_1$): This hypothesis represents the anticipated outcome of the study. It proposes that individuals diagnosed with obesity exhibit a significantly higher mean daily caloric and fat intake compared to their non-obese counterparts ($H_1: \mu_{obesity} > \mu_{healthy}$). Confirming this hypothesis provides the statistical evidence needed to assert that higher intake levels are consistent, predictable markers of obesity in the target population.
        
        Implications and Conclusion Validating the Alternative Hypothesis is critical for the future of nutritional intervention. If this study confirms a statistically significant difference in intake levels, it reinforces the validity of caloric and fat restriction as a primary treatment modality. Furthermore, quantifying how significant this difference is can help healthcare providers design more precise nutritional plans. By validating the link between specific dietary metrics and obesity risk, this research contributes to a foundation for improved public health strategies, more effective weight management programs, and a deeper understanding of the nutritional drivers of the global obesity crisis.            
        
        """)    

# -------------------------------------------------------------------------
# APP MODE: DATA ANALYSIS
# -------------------------------------------------------------------------
if app_mode == "Data Analysis":
    st.title("Exploratory Data Analysis & Statistics")
    
    if df is not None:

        # Descriptive Stats
        nutrition_eda.overview()
        st.markdown("---")
        nutrition_eda.variable_types()
        st.markdown("---")
        nutrition_eda.detect_outliers()
        st.markdown("---")
        
        # Hypothesis Testing
        st.header("1. Hypothesis Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fat Intake vs. Obesity")
            group_sick = df[df['obesity'] == 1]['fat']
            group_healthy = df[df['obesity'] == 0]['fat']
            t_stat, p_val = stats.ttest_ind(group_sick, group_healthy, equal_var=False)
            
            st.write(f"**T-statistic:** {t_stat:.4f}")
            st.write(f"**P-value:** {p_val:.4f}")
            
            fig_fat, ax_fat = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='obesity', y='fat', data=df, palette='Set2', hue='obesity', legend=False, ax=ax_fat)
            st.pyplot(fig_fat)

        with col2:
            st.subheader("Calorie Intake vs. Obesity")
            group_sick_cal = df[df['obesity'] == 1]['calories']
            group_healthy_cal = df[df['obesity'] == 0]['calories']
            t_stat_cal, p_val_cal = stats.ttest_ind(group_sick_cal, group_healthy_cal, equal_var=False)
            
            st.write(f"**T-statistic:** {t_stat_cal:.4f}")
            st.write(f"**P-value:** {p_val_cal:.4f}")
            
            fig_cal, ax_cal = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='obesity', y='calories', data=df, palette='Set2', hue='obesity', legend=False, ax=ax_cal)
            st.pyplot(fig_cal)

        st.markdown("---")
        
        # Correlation
        # st.header("2. Correlation Analysis")
        # numerical_df = df.select_dtypes(include=[np.number])
        # corr_matrix = numerical_df.corr(method='pearson')
        
        # fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax_corr)
        # st.pyplot(fig_corr)
        nutrition_eda.correlation_heatmap()
        st.markdown("---")
        # Histograms
        nutrition_eda.plot_histograms(feature_list=['calories', 'fat'])
        st.markdown("---")
        # Boxplots
        nutrition_eda.boxplots(feature_list=['calories', 'fat'])
        st.markdown("---")
        # Obesity Intake Comparison
        nutrition_eda.obesity_intake_comparison()

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
        target_col = 'obesity'
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