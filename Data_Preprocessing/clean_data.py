import pandas as pd

class DataCleaner:
    #Class to handle basic data cleaning and steps:
    # Remove duplicates, drop missing vales and noramlize column names
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() #Work on a copy to avoid modifying original


    def drop_duplicates(self):
        #Removes duplicate rows from the dataset
        self.df = self.df.drop_duplicates()

    def drop_missing_values(self):
        #remove rows with any missing values
        self.df = self.df.dropna()

    def normalize_col(self):
        #Normalize column names:
        # strip whitespace, replace spaces with underscores and convert to lowercase
        self.df.columns = self.df.columns.str.strip()
        self.df.columns = [c.replace(" ", "_").lower() for c in self.df.columns]

    def clean_pipe(self):
        #run all cleaning steps
        self.drop_duplicates()
        self.drop_missing_values()
        self.normalize_col()

    def get_clean_data(self):
        #Return cleaned dataframe
        return self.df