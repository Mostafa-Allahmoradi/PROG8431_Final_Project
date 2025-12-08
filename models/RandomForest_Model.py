import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree


class RandomForestModel:
    def __init__(self, df, target_col="obesity", features=None, n_estimators=7, max_depth=None , criterion="gini", test_size= 0.25, random_state = 42):
        """
        Initialize the Random Forest model class.

       Parameters:
        - x : feature matrix (NumPy array or sparse matrix)
        - y : target vector
        - n_estimators : number of trees
        - max_depth : max depth of tree
        - criterion : "gini" or "entropy
        - test_size : test split
        - random_state : reproducibility
        - max_depth : max depth of tree
        - criterion : "gini" or "entropy"
        """
        # Convert sparse to desne if neede
        if features is None:
            #Default feature set
            features = ["calories", "fat", "protein", "carbohydrates", "sugar", "fiber", "sodium"]

        for f in features:
            if f not in df.columns:
                raise ValueError(f"Feature{f} not found")
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        self.features = features
        self.target_col = target_col
        self.x = df[features].values
        self.y = df[target_col].values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        self.model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=random_state)
        self.y_pred = None

    def train(self):
        self.model.fit(self.x_train, self.y_train)

        return self.model

    def plot_tree(self, n_trees=7):
        if self.model is None:
            st.error("Train the model first to plot")
            return
        n_trees = min(n_trees, len(self.model.estimators_))
        st.subheader(f"Random Forest: Visualizing {n_trees} Trees")

        for i in range(n_trees):
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                self.model.estimators_[i],
                feature_names=self.features,
                class_names=[str(c) for c in self.model.classes_],
                filled=True,
                rounded=True,
                proportion=True,
                fontsize=10,
                ax=ax
            )
            ax.set_title(f"Tree {i + 1} of {len(self.model.estimators_)}")
            st.pyplot(fig)


    def evaluate(self):

        if self.model is None:
            st.error("Model has not been trained yet. Call build_model() first.")
            return None

        st.subheader("Random Forest Model Evaluation")
        # Predictions
        self.y_pred = self.model.predict(self.x_test)

        #Accuracy
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

        """
        Returns feature importance in a sorted DataFrame.
        """
        importance_df = pd.DataFrame({
            "Feature": range(self.x_train.shape[1]),
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        st.write("### Feature Importance")
        st.dataframe(importance_df)

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "feature_importance": importance_df

        }

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

