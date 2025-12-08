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
        if hasattr(x, "toarray"):
            x = x.toarray()
 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.model = None
        self.scaler = StandardScaler()
        self.y_pred = None
 
    def train(self, C=1.0, max_iter=1000):
        # Scale features
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
 
        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.model.fit(self.x_train, self.y_train)
 
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
 
    def predict(self, new_data):
        if self.model is None:
            st.error("Model must be trained before calling predict().")
            return None
 
        if hasattr(new_data, "toarray"):
            new_data = new_data.toarray()
 
        new_data_scaled = self.scaler.transform(new_data)
        preds = self.model.predict(new_data_scaled)
 
        st.subheader("Prediction on New Data")
        st.dataframe(pd.DataFrame({"Predictions": preds}))
 
        return preds