import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NaiveBayesModel:
    def __init__(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the Naive Bayes model class.

        Parameters:
        - df : preprocessed DataFrame
        - target_column : name of the target variable
        - test_size : proportion of test data
        - random_state : reproducibility
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Model + predictions
        self.model = None
        self.predictions = None


    def build_model(self, model_type="gaussian"):
        """
        Build and train a Naive Bayes model.

        Parameters:
        - model_type : "gaussian", "multinomial", or "bernoulli"

        Returns:
        - trained model
        """
        if model_type.lower() == "gaussian":
            self.model = GaussianNB()

        elif model_type.lower() == "multinomial":
            self.model = MultinomialNB()

        elif model_type.lower() == "bernoulli":
            self.model = BernoulliNB()

        else:
            raise ValueError("model_type must be: 'gaussian', 'multinomial', or 'bernoulli'")

        self.model.fit(self.X_train, self.y_train)
        return self.model


    def evaluate_model(self):
        """
        Evaluate the Naive Bayes model.

        Returns:
        A dictionary containing accuracy, confusion matrix, and classification report.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call build_model() first.")

        # Predictions
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
