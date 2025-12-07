import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Data_Preprocessing.clean_data import DataCleaner
from Data_Preprocessing.feature_engineering import DataPreprocessor

class NutritionEDA:
    def __init__(self, file_path: str):

        self.df = pd.read_csv(file_path)
        st.success("Dataset loaded successfully!")

    def overview(self):
        st.subheader("Dataset Overview")
        st.dataframe(self.df.head())

        st.subheader("Data Info")
        buffer = []
        self.df.info(buf=buffer.append)
        st.text("\n".join(buffer))

        st.subheader("Descriptive Statistics")
        st.dataframe(self.df.describe(include='all').T)

        return self.df.head()
    
    def clean_data(self):
        st.subheader("Data Cleaning")
        cleaner = DataCleaner(self.df)
        cleaner.clean_pipe()
        self.df = cleaner.get_clean_data()
        st.success("Data cleaning completed!")
        st.dataframe(self.df.head())
        return self.df
    
    def perform_feature_engineering(self):
        st.subheader("Feature Engineering")
        preprocessor = DataPreprocessor(self.df)
        preprocessor.convert_height_meters()
        preprocessor.calculate_bmi()
        preprocessor.add_obesity_features()
        preprocessor.encode_categorical_features()
        self.df = preprocessor.df
        st.success("Feature engineering completed!")
        st.dataframe(self.df.head())
        return self.df
    
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
    
    def correlation_heatmap(self):
        st.subheader("Correlation Heatmap")

        numeric_cols = self.df.select_dtypes(include=np.number)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_cols.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        return numeric_cols.corr()
    
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

        if "Disease" not in self.df.columns:
            st.error("Disease column not found in dataset.")
            return None

        calories = ["Calories", "Daily Calorie Target"]
        fats = ["Fat"]

        results = {}

        for col in calories + fats:
            if col in self.df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=self.df["Disease"], y=self.df[col], ax=ax)
                ax.set_title(f"{col} by Disease Status")
                st.pyplot(fig)
                results[col] = self.df.groupby("Disease")[col].mean()

        return results
    


