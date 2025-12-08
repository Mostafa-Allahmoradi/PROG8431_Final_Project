import streamlit as st
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DecisionTreeModel:
    def __init__(self, x, y, test_size=0.25, random_state=42, max_depth=None, criterion="gini"):
        """
     but
        Parameters:
        - x : feature matrix (NumPy array or sparse matrix)
        - y : target vector
        - test_size : test split
        - random_state : reproducibility
        - max_depth : max depth of tree
        - criterion : "gini" or "entropy"
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
        self.y_pred = None

    def train(self):
        return self.model.fit(self.x_train, self.y_train)


    def evaluate(self):
        """
        Evaluates the trained Decision Tree model.

        Returns:
        A dictionary of Streamlit-friendly outputs.
        """
        if self.model is None:
            st.error("Model has not been trained yet. Call build_model() first.")
            return None

        st.subheader("Decision Tree Model Evaluation:")
        # Predictions
        self.y_pred = self.model.predict(self.x_test)

        acc =  accuracy_score(self.y_test, self.y_pred)
        st.write(f"### Accuracy: `{acc:.4f}`")
        # Classification Report
        report = classification_report(self.y_test, self.y_pred, output_dict=True)
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

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": range(self.x_train.shape[1]),  # since x is a matrix
                "Importance": self.model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
        st.write("### Feature Importance")
        st.dataframe(importance_df)

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "feature_importance": importance_df if hasattr(self.model, "feature_importances_") else None
        }

        # ------------------------------------------------------
        # PREDICTION ON NEW DATA
        # ------------------------------------------------------
    def predict(self, new_data):
        if self.model is None:
            st.error("Model must be trained before calling predict().")
            return None

        st.subheader("Prediction on New Data")
        preds = self.model.predict(new_data)
        st.dataframe(pd.DataFrame({"Predictions": preds}))
        return preds
