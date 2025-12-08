import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


class SupportVectorMachineModel:
    def __init__(self, x, y, test_size=0.25, random_state=42):
    #Uses Support Vector Machine classifier for obesity classification
    #Uses selected features: age, activity level & dietary preference

    # Convert sparse to desne if neede
        if hasattr(x, "toarray"):
         x = x.toarray()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.model = None
        self.y_pred = None


    def train(self, kernel="rbf", c=1.0, gamma= "scale", probability=False): #rbf sets the type of SVM kernel which determines how the SVM transforms the input to find a decsion boundary rbf is radial basis function whcih is non-linear and mostly used
        #C=1.0 controls the trade-off between correctly classifying points and keeping the decision boundary smooth for avoiding overfitting
        #Gamma=scale is the kernal coefficient which controls how far influence ofa  single training point
        #Train SVM model
        self.model = SVC(kernel=kernel, C=c, gamma=gamma, probability=probability)
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        if self.model is None:
            st.error("Model has not been trained yet. Call build_model() first.")
            return None

        st.subheader("Support Vector Machine Evaluation")
        # Predictions
        self.y_pred = self.model.predict(self.x_test)

        # Accuracy
        acc = accuracy_score(self.y_test, self.y_pred)
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


        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,

        }

    #Prediction on New Data
    def predict(self, new_data):
        if self.model is None:
            st.error("Model must be trained before calling predict().")
            return None

        if hasattr(new_data, "toarray"):
            new_data = new_data.toarray()

        st.subheader("Prediction on New Data")
        preds = self.model.predict(new_data)
        st.dataframe(pd.DataFrame({"Predictions": preds}))
        return preds
    def predict_proba(self, new_data):
        if self.model is None:
            st.error("Model must be trained before calling predict_proba().")
            return None
        if hasattr(new_data, "toarray"):
            new_data = new_data.toarray()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(new_data)
        else:
            st.error("SVM was not trained with probability=True")
            return None
