import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Data_Preprocessing.EDA import NutritionEDA
#created model imports
from models.LogisticRegression_Model import LogisticRegressionModel
from models.SupportVectorMachine_Model import SupportVectorMachineModel
from models.KNN_Model import KNNModel
from models.RandomForest_Model import RandomForestModel
from models.NaiveBayes_Model import NaiveBayesModel
from models.DecisionTrees_Model import DecisionTreeModel

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

        x = nutrition_eda.x #use ml read x and y
        y = nutrition_eda.y
        df =  nutrition_eda.df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, None

    return x, y, df

nutrition_eda = NutritionEDA('./data/raw/detailed_meals_macros_.csv')
x, y, df = load_and_prep_data()

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
        st.header("Hypothesis Testing")

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

        nutrition_eda.correlation_heatmap()
        st.markdown("---")

        # PCA Variance Plot
        nutrition_eda.pca_variance_plot()
        st.markdown("---")

        # Histograms'
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

    if x is not None and y is not None:
        # --- Instantiate & Train the Selected Model ---
        if model_selection == "Logistic":
            # Sidebar for options for hyperparameters
            c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            max_iter_val = st.sidebar.slider("Max. Iterations", 100, 5000, 1000)
            use_prob = st.sidebar.checkbox("Show Probability", value=True) #checkbox
            # Initalize
            model = LogisticRegressionModel(x, y)
            model.train(c=c_val, max_iter=max_iter_val)

            st.subheader(f"Logistic Regression on BMI (C={c_val}), max_iter={max_iter_val})")
            model.plot_logistic_curve()
            model.evaluate()
            # Show probabilistic reasoning
            if use_prob:
                model.predict_probabilistic()

        elif model_selection == "K-NN":
            k_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
            model = KNNModel(x, y, n_neighbors=k_neighbors)
            st.subheader(f"K-NN (K={k_neighbors})")

            with st.spinner(f"Training {model_selection}..."):
                model.train()
                model.evaluate()
            model.plot_knn_curve(max_k=20)

        elif model_selection == "Support Vector Machine":
            kernel = st.sidebar.selectbox("Kernel",  [ "linear", "rbf", "poly", "sigmoid"], index=0)
            c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
            gamma_val = st.sidebar.selectbox("Gamma", ["scale", "auto"])
            prob = st.sidebar.checkbox("Enable probability estimates", value=False)

            # Use only selected features for svm
            svm_features = [ "calories", "fat",]
            target_col = "obesity"
            svm_df = x[svm_features].copy()
            svm_df[target_col] = y.values
            model = SupportVectorMachineModel(svm_df, target_col=target_col, features=svm_features)

            # Train the model
            model.train(kernel=kernel, c=c_val, gamma=gamma_val, probability=prob)

            # Plot (first two features)
            model.plot_svm_boundary()
            # Evaluate
            model.evaluate()

            st.subheader(f"Support Vector Machine (kernel={kernel} C={c_val}, gamma={gamma_val})")

        elif model_selection == "Decision Trees":
            max_d = st.sidebar.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeModel(x, y, max_depth=max_d)

            st.subheader(f"Decision Tree (Max Depth={max_d})")
            # Train and Evaluate
            with st.spinner(f"Training {model_selection}..."):
                model.train()
                model.evaluate()

            model.plot_tree(feature_names=nutrition_eda.x.columns, class_names=["Non-Obese", "Obese"])


        elif model_selection == "Naive Bayes":
            st.sidebar.subheader("Naive Bayes Options")
            nb_features = ["bmi", "calories", "fat"]

            # Initalize Gaussian Naive Bayes
            model = NaiveBayesModel(df=x.join(y), target_col="obesity", features = nb_features)

            model.train()
            st.subheader(f"Gaussian Naive Bayes (Features: {','.join(nb_features)})")

            # Evaluate performance
            model.evaluate()

            # Plot decision boundary using first 2 features
            if len(nb_features) >= 2:
                model.plot_decision_boundary()



        elif model_selection == "Random Forest":
            n_est = st.sidebar.slider("Number of Trees", 2, 10, 5)
            max_d = st.sidebar.slider("Max Depth", 1, 10, 5)
            n_plot = st.sidebar.slider("Number of Trees to visualize", 1)
            model = RandomForestModel(df=x.join(y), target_col="obesity", n_estimators=n_est, max_depth=max_d)
            st.subheader(f"Random Forest (Trees={n_est}), Max Depth={max_d}")

            # Train model
            with st.spinner(f"Training {model_selection}..."):
                model.train()
                model.evaluate()

            model.plot_tree(n_trees=n_plot)

else:
    if app_mode != "Overview" and app_mode != "Problem Statement":
        st.warning("Please verify dataset availability.")