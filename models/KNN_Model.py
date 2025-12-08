import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class KNNModel:
    def __init__(self, x, y, n_neighbors=5, test_size=0.25, random_state=42):
        """
        Object-Oriented KNN model class for Streamlit applications.
        Assumes dataframe is already preprocessed before being passed in.

        Parameters:
        - x: feature matrix (NumPy array or sparse matrix)
        - y: target vector (NumPy array or sparse matrix)
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.y_pred = None

    # ------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------
    def train(self):
        return self.model.fit(self.x_train, self.y_train)
    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------
    def evaluate(self):
        """Evaluate the trained model and display metrics."""
        if self.model is None:
            st.error("Model is not trained yet. Please train the model first.")
            return None

        st.subheader("Model Evaluation:")

        # Predictions
        self.y_pred = self.model.predict(self.x_test)

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

        st.dataframe(pd.DataFrame({"Predictions": preds}))
        return preds

