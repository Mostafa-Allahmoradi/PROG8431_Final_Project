import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class SVMClassifier:
    #Uses Support Vector Machine classifier for obesity classification
    #Uses selected features: age, activity level & dietary preference

    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):

        #Initlaize classifier with features and split data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None


    def train(self, kernel="rbf", c=1.0, gamma= "scale"): #rbf sets the type of SVM kernel which determines how the SVM transforms the input to find a decsion boundary rbf is radial basis function whcih is non-linear and mostly used
        #C=1.0 controls the trade-off between correctly classifying points and keeping the decision boundary smooth for avoiding overfitting
        #Gamma=scale is the kernal coefficient which controls how far influence ofa  single training point
        #Train SVM model
        self.model = SVC(kernel=kernel, C=c, gamma=gamma)

        self.model.fit(self.x_train, self.y_train)
        print(f"Support Vector Machine trained with kernal={kernel}, C={c}, gamma={gamma}")

    def evaluate(self):
        #evaluate the model with classification report and confusion matrix
        y_pred = self.model.predict(self.x_test)
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")
        plt.title("SVM Confusion Matrix")
        plt.show()

    def predict(self, x_new: pd.DataFrame):
        #Make predictions on new data
        return self.model.predict(x_new)

    def predict_proba(self, x_new: pd.DataFrame):
        #Return class probability for new data
        if hasattr(self.model, "predict_proba"): #hasattr checks if an object has a specific attribute or method
            return self.model.predict_proba(x_new)
        else:
            raise AttributeError("SVM model was not trained with probability=true")




#Example usage
# if __name__ == "__main__":
#     # Assume X_scaled and y are already loaded from your preprocessing pipeline
#     feature_cols = ["age", "activity_level", "dietary_preference"]
#     X_selected = x_scaled[feature_cols]
#
#     # Initialize SVM classifier
#     svm_clf = SVMClassifier(X_selected, y)
#
#     # Train and evaluate
#     svm_clf.train(kernel='rbf', C=1.0, gamma='scale')
#     svm_clf.evaluate()
# Optional: predict new data
# y_new_pred = svm_clf.predict(X_new_df)
