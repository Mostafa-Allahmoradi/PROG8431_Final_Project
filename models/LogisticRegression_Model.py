import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LogisticRegressionModel:
    def __init__(self, x, y, test_size=0.25, random_state=42):
        # Convert sparse to dense if needed
        if isinstance(x, pd.DataFrame):
            x = x[["bmi"]].values
        elif hasattr(x, "toarray"):
            x = x.toarray()
            x = x[:, [0]]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.model = None
        self.y_pred = None

    def train(self, c=1.0, max_iter=1000):

        self.model = LogisticRegression(C=c, max_iter=max_iter)
        self.model.fit(self.x_train, self.y_train)

    def plot_logistic_curve(self):
        if self.model is None:
            st.error("Train the model first.")
            return
        x_sorted = np.linspace(self.x_train.min(), self.x_train.max(), 300).reshape(-1, 1)
        prob = self.model.predict_proba(x_sorted)[:, 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.x_train, self.y_train, color="blue", alpha=0.5, label="Actual")
        ax.plot(x_sorted, prob, color="red", linewidth=2, label="Fitted Curve")
        ax.set_xlabel("Scaled BMI")
        ax.set_ylabel("Probability of Obese")
        ax.set_title("Logistic Regression: Obese vs BMI")
        ax.legend()
        st.pyplot(fig)

    def evaluate(self):
        if self.model is None:
            st.error("Model has not been trained yet. Call train() first.")
            return None

        self.y_pred = self.model.predict(self.x_test)
        acc = accuracy_score(self.y_test, self.y_pred)

        st.subheader("Logistic Regression Evaluation")
        st.write(f"### Accuracy: `{acc:.4f}`")

        report = classification_report(self.y_test, self.y_pred, output_dict=True)
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report).T)

        cm = confusion_matrix(self.y_test, self.y_pred)
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Feature coefficients
        if hasattr(self.model, "coef_"):
            coef_df = pd.DataFrame({
                "Feature": [f"Feature_{i}" for i in range(self.x_train.shape[1])],
                "Coefficient": self.model.coef_.ravel()
            }).sort_values(by="Coefficient", ascending=False)
            st.write("### Feature Coefficients")
            st.dataframe(coef_df)
        else:
            coef_df = None

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "coefficients": coef_df
        }



    def predict_probabilistic(self, new_data=None):
        if self.model is None:
            st.error("Model must be trained before calling predict().")
            return None

        if new_data is None:
            data =self.x_test
        else:
            if isinstance(new_data, pd.DataFrame):
                data = new_data[["bmi"]].values
            else:
                data = np.array(new_data).reshape(-1, 1)
        probs = self.model.predict_proba(data)[:,1]

        st.subheader("Probabilistic Reasoning of Being Obese")
        df_probs = pd.DataFrame({"bmi": data.flatten(), "Probability_Obese": probs})
        st.dataframe(df_probs.reset_index(drop=True))
        return df_probs