import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

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
        self.df["height_m"] =