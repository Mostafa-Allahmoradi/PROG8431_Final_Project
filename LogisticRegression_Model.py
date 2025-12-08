import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class LogisticRegressionModel:
    """
    Logistic Regression model for Obese vs Non-Obese classification.

    Designed to work cleanly with your current repo structure without requiring:
    from Data_Loader... or from Data_Preprocessing...

    You can pass a preprocessed df OR let this class load the CSV.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        data_path: str = "data/raw/detailed_meals_macros_.csv",
        target_column: str = "obese",
        test_size: float = 0.2,
        random_state: int = 42,
        bmi_threshold: float = 30.0,
        drop_height_weight_from_X: bool = True
    ):
        self.df = df
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.bmi_threshold = bmi_threshold
        self.drop_height_weight_from_X = drop_height_weight_from_X

        self.model = None
        self.scaler = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.feature_columns_ = None

    # -------------------------
    # Core pipeline steps
    # -------------------------

    def load_data(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df.copy()

        return pd.read_csv(self.data_path)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Lowercase + replace spaces with underscores
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Handle duplicate-like names such as dinner_protein.1
        # Make columns unique if pandas auto-added suffixes
        # (safe no-op if already unique)
        new_cols = []
        seen = {}
        for c in df.columns:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
        df.columns = new_cols

        return df

    def _create_obesity_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates BMI + obese label if not already present.
        Assumes height in cm and weight in kg if available.
        """
        df = df.copy()

        if self.target_column in df.columns:
            return df

        # Try common column names after normalization
        height_col = "height"
        weight_col = "weight"

        if height_col in df.columns and weight_col in df.columns:
            height_m = df[height_col] / 100.0
            bmi = df[weight_col] / (height_m ** 2)

            df["bmi"] = bmi
            df[self.target_column] = (df["bmi"] >= self.bmi_threshold).astype(int)
        else:
            raise ValueError(
                f"Cannot create target '{self.target_column}'. "
                f"Expected columns '{height_col}' and '{weight_col}' to compute BMI."
            )

        return df

    def _drop_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop any column that looks like suggestion text
        drop_cols = [c for c in df.columns if "suggestion" in c]
        # Your dataset also has a "disease" column that looks like goal text
        if "disease" in df.columns:
            drop_cols.append("disease")

        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df.drop(columns=existing, inplace=True)

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Identify object columns except target
        cat_cols = [
            c for c in df.columns
            if df[c].dtype == "object" and c != self.target_column
        ]

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df

    def preprocess(self) -> pd.DataFrame:
        df = self.load_data()
        df = self._normalize_columns(df)
        df = df.drop_duplicates()

        # Basic missing-value handling
        # (We’ll do final numeric fill later)
        df = df.dropna(subset=[c for c in df.columns if c in ["height", "weight"]], how="any")

        df = self._create_obesity_label(df)
        df = self._drop_text_columns(df)
        df = self._encode_categoricals(df)

        # Optionally remove direct BMI inputs from features to reduce leakage
        if self.drop_height_weight_from_X:
            for col in ["height", "weight", "bmi"]:
                if col in df.columns:
                    # keep them in df for reference if you want,
                    # but we'll drop from X later in build_features
                    pass

        # Fill remaining missing numeric with median
        for c in df.columns:
            if c != self.target_column and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())

        self.df = df
        return df

    def build_features(self):
        if self.df is None:
            self.preprocess()

        df = self.df.copy()

        y = df[self.target_column].astype(int)

        X = df.drop(columns=[self.target_column])

        if self.drop_height_weight_from_X:
            for col in ["height", "weight", "bmi"]:
                if col in X.columns:
                    X = X.drop(columns=[col])

        # Keep feature names for later reference
        self.feature_columns_ = X.columns.tolist()

        return X, y

    def split_data(self):
        X, y = self.build_features()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

    def build_model(self, C: float = 1.0, max_iter: int = 1000):
        if self.X_train is None:
            self.split_data()

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        self.model = LogisticRegression(C=C, max_iter=max_iter)
        self.model.fit(X_train_scaled, self.y_train)

        # Store scaled test for evaluation convenience
        self._X_test_scaled = X_test_scaled

    def evaluate_model(self):
        if self.model is None:
            self.build_model()

        y_pred = self.model.predict(self._X_test_scaled)

        acc = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=["Non-Obese", "Obese"])

        return {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def get_coefficients(self) -> pd.DataFrame:
        if self.model is None or self.feature_columns_ is None:
            raise ValueError("Train the model before requesting coefficients.")

        coef = self.model.coef_.ravel()
        return pd.DataFrame({
            "feature": self.feature_columns_,
            "coefficient": coef
        }).sort_values(by="coefficient", ascending=False)

    def predict(self, new_df: pd.DataFrame):
        if self.model is None or self.scaler is None:
            raise ValueError("Train the model before calling predict().")

        tmp = new_df.copy()
        tmp = self._normalize_columns(tmp)
        tmp = self._drop_text_columns(tmp)
        tmp = self._encode_categoricals(tmp)

        # Align columns to training features
        for col in self.feature_columns_:
            if col not in tmp.columns:
                tmp[col] = 0

        tmp = tmp[self.feature_columns_]

        X_scaled = self.scaler.transform(tmp)
        return self.model.predict(X_scaled)
