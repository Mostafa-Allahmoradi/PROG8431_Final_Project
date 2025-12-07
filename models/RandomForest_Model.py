import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class RandomForestModel:
    def __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the Random Forest model class.

        Parameters:
        - df : preprocessed DataFrame
        - target_column : name of the target variable
        - test_size : proportion of test data
        - random_state : reproducibility for consistent results
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        # Prepare features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Initialized model and predictions
        self.model = None
        self.predictions = None


    def build_model(self, n_estimators=100, max_depth=None, criterion="gini"):
        """
        Build and train the Random Forest classifier.

        Parameters:
        - n_estimators : number of trees
        - max_depth : limit the depth of trees
        - criterion : 'gini' or 'entropy'
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=self.random_state
        )

        self.model.fit(self.X_train, self.y_train)
        return self.model


    def evaluate_model(self):
        """
        Evaluate the trained Random Forest model.

        Returns:
        A dictionary containing accuracy, confusion matrix, and classification report.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call build_model() first.")

        # Predict
        self.predictions = self.model.predict(self.X_test)

        # Metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        report = classification_report(self.y_test, self.predictions, output_dict=False)
        cm = confusion_matrix(self.y_test, self.predictions)

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
        }

        return results


    def get_feature_importance(self):
        """
        Returns feature importance in a sorted DataFrame.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call build_model() first.")

        importance_df = pd.DataFrame({
            "feature": self.X.columns,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        return importance_df
