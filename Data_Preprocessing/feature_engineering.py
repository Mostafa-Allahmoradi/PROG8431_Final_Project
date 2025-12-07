import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    #Class to handle feature engineering where we
    # Convert height from cm to m
    # calculate bmi
    # create obesity label (gender-specific)
    #Encode categorical features
    #Scale numeric features
    #optional features for experimentation
    #Dimrndionslity Reduction Techniques(Correlation filter + pca)

    def __init__(self, df: pd.DataFrame):

        self.df= df.copy()
        self.x_scaled = None #Store scaled feature matrix
        self.y = None #Store target vector
        self.label_encoders = {} #Store encoders for categorical vairables

        #Feature Creation
    def convert_height_meters(self, height_col: str = "height"):
        #Converts height from cm to m
        self.df["height_m"] = self.df[height_col]/100

    def calculate_bmi(self, weight_col: str = "weight"):
        #calculate BMI using weight and height in meters
        self.df["bmi"] = self.df[weight_col] / (self.df["height_m"] ** 2)

    def add_obesity_features(self, gender_col: str = "gender"):
        # Add obesity feature based on gender-specific BMI thresholds:
        # Male: BMI >= 30 > Obese
        # Female: BMI >= 25 Obese
        gender_cat = {"Male": 1, "Female": 0}
        self.df["gender"] = self.df[gender_col].map(gender_cat)
        self.df["obesity"] = np.where(
            ((self.df["gender"]== 1) & (self.df["bmi"] >= 30)) |
            ((self.df["gender"]== 0) & (self.df["bmi"] >= 25)),
            1,
            0
        )
    #Categorical Encoding
    def encode_categorical_features(self, cat_cols=None):
        #Encode categorical features to numeric using labelcncoder
        if cat_cols is None:
            cat_cols = ["activity_level", "dietary_preference"]
        le = LabelEncoder()
        for col in cat_cols:
            self.df[col] = self.df[col].astype("category")
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le # Save encoder for future decoding

    # def impute_missing(self):
    #     numeric_cols = self.df.select_dtypes(include="number").columns
    #     cat_cols = self.df.select_dtypes(include="number").columns
    #
    #     #Numeric > Median
    #     self.df[numeric_cols] = SimpleImputer(strategy="median").fit_transform(self.df[numeric_cols])
    #
    #     #Categoricals > Most frequent
    #     if len(cat_cols) > 0:
    #         self.df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(self.df[cat_cols])
    #Optional additions\
    def weight_risk(self):
        #High Caloire intake and weight gain risk
        self.df["high_calorie_intake"] = np.where(
        self.df["calories"] > self.df["daily_calorie_target"], 1, 0
        )
        self.df["weight_gain_risk"] = np.where(
            (self.df["calories"] > 2500 )& (self.df["activity_level"] < 2), 1, 0
        )

    #Scaling

    def scale_features(self, features_col: list):
        scaler = StandardScaler()
        self.x_scaled = pd.DataFrame(
            scaler.fit_transform(self.df[features_col]), columns=features_col
        )


    #Feature and Target preparation
    def prepare_features_and_target(self, feature_cols: list, target_col: str = "obesity"):
        #Finalize x_scaled and target vector y
        self.scale_features(feature_cols)
        self.y = self.df[target_col]

    def get_processed_data(self):
        # Return processed features and target
        return self.x_scaled, self.y

    #Dimension Reduction techniques (High correlation_Filter & PCA)
    def high_correlation_filter(self, threshold: float = 0.9, plot_heatmap: bool =  True):
        #Remove features that are highly correlated with each other
        if self.x_scaled is None:
            raise ValueError("x_scaled must be computed before applying filter")

        corr_matrix = self.x_scaled.corr().abs()
        #Plot for filter
        if plot_heatmap:
            plt.figure(figsize = (12,10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Feature Correlation Heatmap")
            plt.show()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        #Find features with correlation above threshold
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        #Drop highly correlated features
        self.x_scaled = self.x_scaled.drop(columns=to_drop)
        print(f"High correlation filter applied. Dropped Columns: {to_drop}")
        return to_drop

    def apply_pca(self, n_components: int =5, plot_variance: bool = True):
        #Apply PCA to reduce dimensionality of the feature matrix
        if self.x_scaled is None:
            raise ValueError("x_scaled must be computed before applying PCA")
        pca = PCA(n_components=n_components)
        x_pca =  pca.fit_transform(self.x_scaled)
        x_pca_df = pd.DataFrame(
            x_pca, columns=[f"PCA_{i+1}" for i in range(n_components)]
        )
        print(f"PCA applied: {n_components} components generated")

        #Explained Variance Plot
        if plot_variance:
            plt.figure(figsize=(8, 5))
            plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
            plt.step(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red')
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance Ratio")
            plt.title("PCA Explained Variance")
            plt.xticks(range(1, n_components + 1))
            plt.grid(alpha=0.3)
            plt.show()
        return x_pca_df, pca