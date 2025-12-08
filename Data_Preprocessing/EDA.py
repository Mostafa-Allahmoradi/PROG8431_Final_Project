import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Data_Preprocessing.clean_data import DataCleaner
from Data_Preprocessing.feature_engineering import DataPreprocessor

class NutritionEDA:
    def __init__(self, file_path: str):
        self.preprocessor = None
        self.x = None
        self.y = None
        self.df = pd.DataFrame()
        try:
            self.df = pd.read_csv(file_path)
            st.toast("Dataset loaded successfully!")
        except Exception as e:
            st.toast(f"Error loading dataset: {e}")
            self.df = pd.DataFrame()

    def overview(self):
        st.subheader("Dataset Overview")
        st.dataframe(self.df.head())

        st.subheader("Descriptive Statistics")
        st.dataframe(self.df.describe(include='all').T)

        return self.df.head()
    
    def clean_data(self):
        cleaner = DataCleaner(self.df)
        cleaner.clean_pipe()
        self.df = cleaner.get_clean_data()
        st.toast("Data cleaning completed!")


    def perform_feature_engineering(self):
        self.preprocessor = DataPreprocessor(self.df)
        self.preprocessor.convert_height_meters()
        self.preprocessor.calculate_bmi()
        self.preprocessor.add_obesity_features()
        self.preprocessor.encode_categorical_features()
        numeric_cols = ['calories', 'protein', 'fat', 'sugar', 'sodium',
                        'carbohydrates', 'fiber', 'height_m', 'weight', 'bmi']
        categorical_cols = ['activity_level', 'dietary_preference']
        self.preprocessor.build_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,

        )
        self.x, self.y = self.preprocessor.prepare_features_and_target(
            feature_cols=[],  # not needed for ColumnTransformer
            target_col='obesity'
        )
        self.df = self.preprocessor.df
        st.toast("Feature engineering completed!")
    
    def variable_types(self):
        numeric = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical = self.df.select_dtypes(exclude=np.number).columns.tolist()

        st.subheader("Variable Types")
        st.write("### Numeric Columns:")
        st.write(numeric)

        st.write("### Categorical Columns:")
        st.write(categorical)

        return numeric, categorical
    
    def detect_outliers(self):
        st.subheader("Outlier Summary (IQR Method)")

        summary = {}

        for col in self.df.select_dtypes(include=np.number).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            summary[col] = outliers

        outlier_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Outlier Count'])
        st.dataframe(outlier_df)

        return outlier_df

    
    def plot_histograms(self, feature_list=None):
        st.subheader("Histograms for Numerical Variables")

        numeric_cols = self.df.select_dtypes(include=np.number).columns

        if feature_list is not None:
            numeric_cols = [col for col in numeric_cols if col in feature_list]

        for col in numeric_cols:
            fig, ax = plt.subplots()
            ax.hist(self.df[col].dropna(), bins=20)
            ax.set_title(f"Histogram: {col}")
            st.pyplot(fig)

        return list(numeric_cols)
    
    def boxplots(self, feature_list=None):
        st.subheader("Boxplots for Outlier Detection")

        numeric_cols = self.df.select_dtypes(include=np.number).columns

        if feature_list is not None:
            numeric_cols = [col for col in numeric_cols if col in feature_list]

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=self.df[col], ax=ax)
            ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)

        return list(numeric_cols)

    def obesity_intake_comparison(self):
        st.subheader("Calorie & Fat Intake Comparison by Disease Status")

        if "disease" not in self.df.columns:
            st.error("Disease column not found in dataset.")
            return None

        calories = ["calories", "daily_calorie_target"]
        fats = ["fat"]

        results = {}

        for col in calories + fats:
            if col in self.df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=self.df["disease"], y=self.df[col], ax=ax)
                ax.tick_params(axis='x', labelrotation=45)
                ax.set_title(f"{col} by Disease Status")
                st.pyplot(fig)
                results[col] = self.df.groupby("disease")[col].mean()

        return results

    def correlation_heatmap(self):

        st.subheader("Correlation Heatmap")

        numeric_cols = self.df.select_dtypes(include=np.number)
        corr_matrix = numeric_cols.corr(method='pearson')

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        return numeric_cols.corr()

    def pca_variance_plot(self, n_components=5):
        if self.preprocessor is None:
            st.warning("Please run feature engineering first!")
            return None

        numeric_cols = [
            "calories", "protein", "fat", "sugar", "sodium",
            "carbohydrates", "fiber", "height_m", "weight", "bmi"
        ]
        self.preprocessor.scale_features(numeric_cols)
        _, pca = self.preprocessor.apply_pca(n_components=n_components, plot_variance=False)


        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7)
        ax.step(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        return pca.explained_variance_ratio_

    


