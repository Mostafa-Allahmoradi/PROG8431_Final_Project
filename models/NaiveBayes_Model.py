import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NaiveBayesModel:
    def __init__(self, df, target_col="obesity", features=["bmi", "calories","fat"],  test_size=0.25, random_state=42):
        """

        Parameters:
        - x: feature matrix (NumPy array or sparse matrix)
        - y: target vector (NumPy array or sparse matrix)
        - test_size: proportion of test data
        - random_state : reproducibility seed
        """
        #if x is sparse it'll convert to dense
        if len(features) > 3:
            raise ValueError("Use max of 3 Features for Naive Bayes")

        for f in features:
            if f not in df.columns:
                raise ValueError(f"Feature {f} not in dataframe")
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not in dataframe")
        self.features = features
        self.target_col = target_col
        self.x = df[features].values
        self.y = df[target_col].values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        #Intialize model
        self.model = GaussianNB()
        self.y_pred = None



    def train(self):
        return self.model.fit(self.x_train, self.y_train)

    def plot_decision_boundary(self):
        #only use first two features for fplotting
        x_min, x_max = self.x_train[:, 0].min() - 1, self.x_train[:, 0].max() + 1
        y_min, y_max = self.x_train[:, 1].min() - 1, self.x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        #if 3rd feature exists fix it at mean
        if len(self.features) == 3:
            z_feature = np.full((grid.shape[0], 1), self.x_train[:, 2].mean())
            grid = np.hstack([grid, z_feature])
        probs = self.model.predict(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.contourf(xx, yy, probs, alpha=0.3, cmap="coolwarm")
        scatter = ax.scatter(self.x_train[:, 0], self.x_train[:, 1],
                             c=self.y_train, cmap="coolwarm", edgecolor="k", s=60)
        ax.set_xlabel(self.features[0].capitalize())
        ax.set_ylabel(self.features[1].capitalize())
        ax.set_title("Naive Bayes Decision Boundary (Train set)")

        #legend
        handles, _ = scatter.legend_elements()
        labels = [str(c) for c in np.unique(self.y_train)]
        ax.legend(handles, labels, title=self.target_col)
        st.pyplot(fig)


    def evaluate(self):

        if self.model is None:
            st.error("Model has not been trained yet. Call build_model() first.")
            return None

        st.subheader("Naive Bayes Model Evaluation")
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





    def predict(self, new_data):
        if self.model is None:
            st.error("Model must be trained before calling predict().")
            return None

        st.subheader("Prediction on New Data")
        preds = self.model.predict(new_data)
        st.dataframe(pd.DataFrame({"Predictions": preds}))
        return preds