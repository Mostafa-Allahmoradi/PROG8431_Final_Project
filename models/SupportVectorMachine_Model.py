import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


class SupportVectorMachineModel:
    def __init__(self, df, target_col="obesity", features=["calories", "fat" ],test_size=0.25, random_state=42):
    #Uses Support Vector Machine classifier for obesity classification
    #Uses selected features: age, activity level & dietary preference

        for f in features:
            if f not in df.columns:
                raise ValueError(f"Feature {f} is not present in the dataframe")
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} is not present in the dataframe")

        self.features = features
        self.target_col = target_col
        self.x = df[features].values
        self.y = df[target_col].values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        self.model = None
        self.y_pred = None


    def train(self, kernel="rbf", c=1.0, gamma= "scale", probability=False): #rbf sets the type of SVM kernel which determines how the SVM transforms the input to find a decsion boundary rbf is radial basis function whcih is non-linear and mostly used
        #C=1.0 controls the trade-off between correctly classifying points and keeping the decision boundary smooth for avoiding overfitting
        #Gamma=scale is the kernal coefficient which controls how far influence ofa  single training point
        #Train SVM model
        self.model = SVC(kernel=kernel, C=c, gamma=gamma, probability=probability)
        self.model.fit(self.x_train, self.y_train)

    def plot_svm_boundary(self):
        if self.model is None:
            st.error("Train the model first")
            return

        if len(self.features) != 2:
            st.error("Features must have 2 values")
            return
        x_plot = self.x_train[:, :2]
        y_plot = self.y_train


        #fit meshgrid
        x_min, x_max = self.x_train[:, 0].min() - 1, self.x_train[:, 0].max() + 1
        y_min, y_max = self.x_train[:, 1].min() - 1, self.x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(xx, yy, z, alpha=0.2, cmap="coolwarm")  # decision regions

        #Plot training points
        scatter = ax.scatter(x_plot[:, 0], x_plot[:, 1], c=y_plot, cmap="coolwarm",
                             s=50, edgecolor="k",)

        #Highlight support vectors
        if hasattr(self.model, "decision_function"):
            zf = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            zf = zf.reshape(xx.shape)
            #Decsion boundary
            ax.contour(xx, yy, zf, levels=[0], colors='k', linewidths=2)
            ax.contour(xx, yy, zf, levels=[-1, 1], colors='k', linestyles='--', linewidths=1)

            # Highlight support vectors
            sv = self.model.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5,
                       label="Support Vectors")

        ax.set_xlabel(self.features[0].capitalize())
        ax.set_ylabel(self.features[1].capitalize())
        ax.set_title("SVM Decision Boundary with Margins")
        ax.legend()
        st.pyplot(fig)


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
