import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class KNNModel:
    def __init__(self, df, target_column, n_neighbors=5, test_size=0.2, random_state=42):
        """
        Object-Oriented KNN model class for Streamlit applications.
        Assumes dataframe is already preprocessed before being passed in.

        Parameters:
        - df: pandas DataFrame (preprocessed)
        - target_column: name of target column
        - n_neighbors: number of neighbors for KNN
        - test_size: train/test split ratio
        """
        self.df = df
        self.target_column = target_column
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    # ------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------
    def train(self):
        """Train the KNN model."""
        st.subheader("Training KNN Model")

        if self.target_column not in self.df.columns:
            st.error(f"Target column '{self.target_column}' not found in dataframe.")
            return None

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Model initialization & training
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(self.X_train, self.y_train)

        st.success(f"KNN model trained with K={self.n_neighbors}")

        return self.model

    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------
    def evaluate(self):
        """Evaluate the trained model and display metrics."""
        if self.model is None:
            st.error("Model is not trained yet. Please train the model first.")
            return None

        st.subheader("Model Evaluation")

        # Predictions
        self.y_pred = self.model.predict(self.X_test)

        # Accuracy
        acc = accuracy_score(self.y_test, self.y_pred)
        st.write(f"### Accuracy: `{acc:.4f}`")

        # Classification report
        report = classification_report(
            self.y_test,
            self.y_pred,
            output_dict=True
        )
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).T)

        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm
        }

    # ------------------------------------------------------
    # PREDICTION ON NEW DATA
    # ------------------------------------------------------
    def predict(self, new_data: pd.DataFrame):
        """
        Predict new instances using the trained model.
        new_data must match the training feature format.
        """
        if self.model is None:
            st.error("Model must be trained before calling predict().")
            return None

        st.subheader("Prediction on New Data")

        preds = self.model.predict(new_data)

        result_df = pd.DataFrame({
            "Input": new_data.index,
            "Prediction": preds
        })

        st.dataframe(result_df)
        return preds
