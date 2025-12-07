import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_curve, auc
)

# -------------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Obesity Risk Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# Introduction & Problem Statement
# -------------------------------------------------------------------------
st.title("Group Presentation #6")

st.sidebar.header("Team Members")
st.sidebar.markdown("""
1. Mostafa Allahmoradi - 9087818
2. Cemil Caglar Yapici – 9081058
3. Jarius Bedward - 8841640
""")

st.header("Problem Statement: Investigating the Relationship Between High Calorie and Fat Intake and Obesity Risk")

st.markdown("""
**Area of Focus:** Obesity is one of the most critical global health challenges. Diet quality—specifically the daily consumption of dietary fat and total calorie intake—is widely recognized as a primary determinant of this condition. While higher consumption of fat and calories contributes to obesity risk, the precise relationship between individual intake levels and the likelihood of obesity requires further clarification. Understanding this relationship is essential for designing targeted dietary interventions and improving nutritional guidelines.

#### Research Question: 
Is there a statistically significant difference in daily calorie and fat intake between individuals diagnosed with obesity and those without the condition?

#### Research Hypothesis
* **Null Hypothesis ($H_0$):** There is no significant difference in the mean daily caloric and fat intake between individuals diagnosed with obesity and those without the condition. 
    $$H_0: \mu_{obesity} \leq \mu_{healthy}$$

* **Alternative Hypothesis ($H_1$):** Individuals diagnosed with obesity have a significantly higher mean daily caloric and fat intake compared to individuals without the condition. 
    $$H_1: \mu_{obesity} > \mu_{healthy}$$

#### Summary
Obesity remains a pervasive global health crisis, closely linked to diet quality. This study investigates the correlation between dietary habits—specifically daily caloric and fat consumption—and the prevalence of obesity. The primary objective is to determine if individuals diagnosed with obesity exhibit statistically higher mean intake levels compared to non-obese individuals. By analyzing these variables, the project aims to validate whether higher intake is a consistent predictor of obesity in the target population. Findings from this research will help clarify the impact of diet on weight management and inform the development of more effective, evidence-based nutritional interventions.
""")

# -------------------------------------------------------------------------
# Data Loading & Engineering Logic
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    # Note: Adjust path if necessary. Using relative path based on notebook context.
    try:
        df = pd.read_csv('./data/raw/detailed_meals_macros_.csv')
    except FileNotFoundError:
        # Fallback for demonstration if file structure differs
        try:
            df = pd.read_csv('detailed_meals_macros_.csv')
        except:
            st.error("File 'detailed_meals_macros_.csv' not found. Please ensure the data file is present.")
            return None

    # --- Data Cleaning (Replicating DataPrepairer logic) ---
    # Interpolate missing values
    df = df.interpolate()
    
    # Normalize headers (lowercase, strip) - simplistic approximation of normalization
    df.columns = [c.strip() for c in df.columns]
    
    if 'Disease' in df.columns:
        df['Disease'] = df['Disease'].str.strip()

    # --- Feature Engineering ---
    # Add Height in meters
    if 'Height' in df.columns:
        df['Height_m'] = df['Height'] / 100
        
    # Add BMI
    if 'Weight' in df.columns and 'Height_m' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
        
    # Add Obesity Target (1 if BMI >= 30)
    df['Obesity'] = np.where(df['BMI'] >= 30, 1, 0)
    
    # Add High Calorie Intake
    if 'Calories' in df.columns and 'Daily Calorie Target' in df.columns:
        df['High_Calorie_Intake'] = np.where(df['Calories'] > df['Daily Calorie Target'], 1, 0)
    
    # Map Activity Level to Numeric for Specific Risk Calc
    # NOTE: The notebook output implies specific mapping was done
    activity_map = {
        "Sedentary": 0,
        "Lightly Active": 1,
        "Moderately Active": 2,
        "Very Active": 3,
        "Extremely Active": 4
    }
    
    # Check if Activity Level is string, then map
    if df['Activity Level'].dtype == 'O':
        # Create a numeric column for the calculation
        df['activity_level_numeric'] = df['Activity Level'].map(activity_map)
        # Keep original for display or one-hot encoding later, but update the column used for logic
    else:
        df['activity_level_numeric'] = df['Activity Level']

    # Weight Gain Risk Logic
    df["Weight_Gain_Risk"] = np.where(
       (df['Calories'] > 2500) & (df['activity_level_numeric'] < 2),
        1, # High Risk
        0 # Low risk
    )
    
    return df

df = load_and_prep_data()

if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Data Analysis Report")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
    with col2:
        st.write("**Target Distribution (Obesity):**")
        st.write(df['Obesity'].value_counts())

    with st.expander("View Descriptive Statistics"):
        st.write(df.describe())

    with st.expander("View Outlier Analysis (IQR Method)"):
        # Simple implementation of the outlier logic from notebook
        outlier_report = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].values
            if len(outliers) > 0:
                outlier_report.append(f"✅ {len(outliers)} Outliers detected for column '{col}'")
        
        for line in outlier_report:
            st.write(line)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Hypothesis Testing 1
    # -------------------------------------------------------------------------
    st.header("Hypothesis Testing: Fat Intake vs. Obesity")
    
    group_sick_fat = df[df['Obesity'] == 1]['Fat']
    group_healthy_fat = df[df['Obesity'] == 0]['Fat']
    
    mean_fat_sick = group_sick_fat.mean()
    mean_fat_healthy = group_healthy_fat.mean()
    
    st.write(f"**Mean fat intake (Obese group):** {mean_fat_sick:.2f} g")
    st.write(f"**Mean fat intake (Non-Obese group):** {mean_fat_healthy:.2f} g")
    
    t_stat_fat, p_value_fat = stats.ttest_ind(group_sick_fat, group_healthy_fat, equal_var=False)
    
    st.write(f"**T-statistic:** {t_stat_fat:.4f}, **P-value:** {p_value_fat:.4f}")
    
    if p_value_fat < 0.05:
        st.success("Result: Reject the null hypothesis. There is a significant difference in fat intake between the two groups.")
    else:
        st.warning("Result: Fail to reject the null hypothesis.")
        
    # Plotting
    fig_fat, ax_fat = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Obesity', y='Fat', data=df, palette='Set2', hue='Obesity', legend=False, ax=ax_fat)
    ax_fat.set_title('Distribution of Fat Intake vs Obesity Status')
    ax_fat.set_xlabel('Is Obese (0=No, 1=Yes)')
    ax_fat.set_ylabel('Daily Fat Intake (g)')
    st.pyplot(fig_fat)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Hypothesis Testing 2
    # -------------------------------------------------------------------------
    st.header("Hypothesis Testing: Calorie Intake vs. Obesity")
    
    group_sick_cal = df[df['Obesity'] == 1]['Calories']
    group_healthy_cal = df[df['Obesity'] == 0]['Calories']
    
    mean_cal_sick = group_sick_cal.mean()
    mean_cal_healthy = group_healthy_cal.mean()
    
    st.write(f"**Mean calorie intake (Obese group):** {mean_cal_sick:.2f} kcal")
    st.write(f"**Mean calorie intake (Non-Obese group):** {mean_cal_healthy:.2f} kcal")
    
    t_stat_cal, p_value_cal = stats.ttest_ind(group_sick_cal, group_healthy_cal, equal_var=False)
    
    st.write(f"**T-statistic:** {t_stat_cal:.4f}, **P-value:** {p_value_cal:.4f}")
    
    if p_value_cal < 0.05:
        st.success("Result: Reject the null hypothesis. There is a significant difference in calorie intake between the two groups.")
    else:
        st.warning("Result: Fail to reject the null hypothesis.")
        
    # Plotting
    fig_cal, ax_cal = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Obesity', y='Calories', data=df, palette='Set2', hue='Obesity', legend=False, ax=ax_cal)
    ax_cal.set_title('Distribution of Calorie Intake vs Obesity Status')
    ax_cal.set_xlabel('Is Obese (0=No, 1=Yes)')
    ax_cal.set_ylabel('Daily Calories (kcal)')
    st.pyplot(fig_cal)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Correlation Analysis
    # -------------------------------------------------------------------------
    st.header("Pearson's Correlation Analysis")
    
    st.markdown("""
    **Relevance:** Pearson correlation is used to measure the association between daily caloric intake and dietary fat consumption to determine if high-calorie diets are driven by fat content (multicollinearity) and how intake tracks with body mass.
    """)
    
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr(method='pearson')
    
    # Display heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)
    
    st.subheader("Strong Correlations (|r| > 0.5)")
    strong_corrs = []
    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if col != row and abs(corr_matrix.loc[row, col]) > 0.5:
                if row < col: # Avoid duplicates
                    strong_corrs.append(f"{row} vs {col}: {corr_matrix.loc[row, col]:.2f}")
    
    st.write(strong_corrs)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Logistic Classification
    # -------------------------------------------------------------------------
    st.header("Machine Learning: Logistic Classification")
    st.markdown("""
    We use Logistic Regression to predict Obesity.
    1. **Split Data:** 80% Train, 20% Test.
    2. **Preprocessing:** Imputation, Scaling, One-Hot Encoding.
    3. **Training:** Logistic Regression.
    """)

    # Setup X and y
    # Note: 'Obesity' is target. We also need to drop derived columns that directly calculate Obesity to avoid data leakage 
    # (Like BMI, Height_m, weight if we treat it as a pure nutritional predictor, but notebook kept weight/height).
    # Following Notebook logic: X drops "Obesity", y is "Obesity"
    
    # Important: In the notebook, X was created from macros_dataset.drop("obesity", axis=1).
    # Since headers were normalized differently in my load function, adapting:
    target_col = 'Obesity'
    
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Detect columns types
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # Build Pipeline
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    st.subheader("Model Evaluation")
    
    # Classification Report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report_dict).transpose())

    col_conf, col_roc = st.columns(2)
    
    with col_conf:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
        st.pyplot(fig_cm)

    with col_roc:
        st.write("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Conditional Probability
    # -------------------------------------------------------------------------
    st.header("Probabilistic Reasoning")
    st.markdown("Quantifying the likelihood of being at **High Risk of Weight Gain** given specific conditions.")

    def calculate_conditional_probability(dataframe, condition_col, target_col):
        target_val = 1
        condition_values = dataframe[condition_col].dropna().unique()
        results = []

        for value in condition_values:
            condition_count = dataframe[dataframe[condition_col] == value].shape[0]
            both_count = dataframe[
                (dataframe[condition_col] == value) &
                (dataframe[target_col] == target_val)
            ].shape[0]

            probability = both_count / condition_count if condition_count > 0 else 0.0

            results.append({
                condition_col: value,
                "P(Target | Condition)": probability,
            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df["Percentage"] = (results_df["P(Target | Condition)"] * 100).round(2).astype(str) + "%"
            results_df = results_df.sort_values(by="P(Target | Condition)", ascending=False)
        
        return results_df

    # 1. Activity Level
    st.subheader("1. Activity Level Risk Analysis")
    st.markdown("""
    * **0:** Sedentary
    * **1:** Lightly Active
    * **2:** Moderately Active
    * **3:** Very Active
    * **4:** Extremely Active
    """)
    
    # We use the numeric column created during engineering
    risk_col = "Weight_Gain_Risk"
    activity_col = "activity_level_numeric" # using the engineered column
    
    if risk_col in df.columns and activity_col in df.columns:
        act_risk = calculate_conditional_probability(df, activity_col, risk_col)
        st.dataframe(act_risk)
        st.info("Interpretation: Sedentary groups (0) show the highest risk, while moderate to high activity groups show significantly lower or zero risk in this dataset.")
    else:
        st.error("Required columns for Activity Risk analysis not found.")

    # 2. Dietary Preference
    st.subheader("2. Dietary Preference Risk Analysis")
    diet_col = "Dietary Preference"
    
    if risk_col in df.columns and diet_col in df.columns:
        diet_risk = calculate_conditional_probability(df, diet_col, risk_col)
        st.dataframe(diet_risk)
        st.info("Interpretation: Shows the percentage of individuals within each diet group deemed High Risk.")
    else:
        st.error("Required columns for Diet Risk analysis not found.")

else:
    st.warning("Please ensure the dataset is available to run the analysis.")