
import streamlit as st
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NaiveBayesModel:
    def __init__(self, x, y, model_type="gaussian", test_size=0.25, random_state=42):
        """

        Parameters:
        - x: feature matrix (NumPy array or sparse matrix)
        - y: target vector (NumPy array or sparse matrix)
        - test_size: proportion of test data
        - random_state : reproducibility seed
        """
        #if x is sparse it'll convert to dense
        if hasattr(x, "toarray"):
            x = x.toarray()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.model_type = model_type.lower()
        self.model = None
        self.y_pred = None
        self.random_state = random_state

        #Intialize model
        if self.model_type == "gaussian":
            self.model = GaussianNB()
        elif self.model_type == "multinomial":
            self.model = MultinomialNB()
        elif self.model_type == "bernoulli":
            self.model = BernoulliNB()
        else:
            raise ValueError("model_type must be 'gaussian', 'multinomial', or 'bernoulli'")


    def train(self):
        return self.model.fit(self.x_train, self.y_train)

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