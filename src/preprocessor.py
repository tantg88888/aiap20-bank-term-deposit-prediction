import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """Handles preprocessing of the bank marketing data, including feature engineering."""

    def __init__(self, numerical_features=None, categorical_features=None, target_column='Subscription Status', project_root=None):
        """
        Initializes the DataPreprocessor.

        Args:
            numerical_features (list): List of numerical column names.
            categorical_features (list): List of categorical column names.
            target_column (str): Name of the target variable column.
            project_root (str, optional): Absolute path to the project root. If None, os.getcwd() is assumed.
        """
        self.project_root = project_root if project_root else os.getcwd()
        # Default features, can be overridden by config
        self.numerical_features = numerical_features if numerical_features is not None else ["Age", "Campaign Calls", "DaysSinceLastContact"]
        self.categorical_features = categorical_features if categorical_features is not None else [
            "Occupation", "Marital Status", "Education Level", 
            "Credit Default", "Housing Loan", "Personal Loan", "Contact Method",
            "PreviouslyContacted" # Engineered feature
        ]
        self.target_column = target_column
        self.preprocessor_pipeline = None
        self.cleaned_columns_before_split = [] # Columns after cleaning, before X,y split
        self.feature_names_out_ = None # To store feature names after transformation

    def _engineer_features(self, df):
        """Performs feature engineering, specifically for 'Previous Contact Days'."""
        # 1. Feature Engineering for 'Previous Contact Days'
        if "Previous Contact Days" in df.columns:
            # Create 'PreviouslyContacted' (binary: 1 if contacted, 0 if 999)
            # Stored as a string category ("0"/"1") so the categorical branch sees uniform types
            df["PreviouslyContacted"] = df["Previous Contact Days"].apply(lambda x: "0" if x == 999 else "1")
            
            # Create 'DaysSinceLastContact' for actual contact days, 999 becomes NaN to be imputed
            # This new column will be treated as numerical. 
            # If not contacted (original was 999), this will be NaN.
            df["DaysSinceLastContact"] = df["Previous Contact Days"].replace(999, np.nan)
            
            # Original 'Previous Contact Days' can now be dropped if desired, as its info is split
            # df = df.drop(columns=["Previous Contact Days"])
            # For now, let's assume it's dropped and we use the new features.
            # If we drop it, ensure it's not in self.numerical_features if it was there by default.
            if "Previous Contact Days" in self.numerical_features:
                self.numerical_features.remove("Previous Contact Days")
            if "DaysSinceLastContact" not in self.numerical_features:
                 self.numerical_features.append("DaysSinceLastContact")
            if "PreviouslyContacted" not in self.categorical_features:
                self.categorical_features.append("PreviouslyContacted")
        else:
            # If 'Previous Contact Days' is missing, create dummy engineered features to avoid pipeline errors
            df["PreviouslyContacted"] = "0" # Assuming no contact if column is missing
            df["DaysSinceLastContact"] = np.nan # Will be imputed
            if "DaysSinceLastContact" not in self.numerical_features:
                 self.numerical_features.append("DaysSinceLastContact")
            if "PreviouslyContacted" not in self.categorical_features:
                self.categorical_features.append("PreviouslyContacted")

        return df

    def clean_data(self, df_input):
        """Performs initial cleaning steps based on EDA findings and new feature engineering."""
        df = df_input.copy()

        if "Client ID" in df.columns:
            df = df.drop(columns=["Client ID"])

        if "Age" in df.columns:
            # expand=False returns a Series (not a one-column DataFrame)
            df["Age"] = df["Age"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            df.loc[df["Age"] > 100, "Age"] = np.nan
            if df["Age"].isnull().any():
                df["Age"] = df["Age"].fillna(df["Age"].mean())
        
        if "Campaign Calls" in df.columns:
            df["Campaign Calls"] = df["Campaign Calls"].abs()

        # Feature Engineering for 'Previous Contact Days'
        df = self._engineer_features(df)

        for col in ["Housing Loan", "Personal Loan", "Credit Default", "Education Level", "Occupation", "Marital Status"]:
            if col in df.columns:
                # fillna also covers Python None values coming from SQLite NULLs
                df[col] = df[col].astype(object).fillna("unknown")

        if "Contact Method" in df.columns:
            df["Contact Method"] = df["Contact Method"].replace({"Cell": "cellular", "Telephone": "telephone"})
            df["Contact Method"] = df["Contact Method"].fillna("unknown") # Also impute Nones here
        
        if self.target_column in df.columns:
            df[self.target_column] = df[self.target_column].map({"yes": 1, "no": 0})
            if df[self.target_column].isnull().any():
                print(f"Warning: Null values found in target column '{self.target_column}' after mapping. Dropping these rows.")
                df.dropna(subset=[self.target_column], inplace=True)
            df[self.target_column] = df[self.target_column].astype(int)

        self.cleaned_columns_before_split = df.columns.tolist()
        return df

    def build_pipeline(self, X_df):
        """
        Builds the scikit-learn preprocessing pipeline based on columns present in X_df.
        """
        # Filter features based on actual columns in X_df after cleaning and X,y split
        actual_numerical_features = [f for f in self.numerical_features if f in X_df.columns]
        actual_categorical_features = [f for f in self.categorical_features if f in X_df.columns]

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")), 
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.preprocessor_pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, actual_numerical_features),
                ("cat", categorical_transformer, actual_categorical_features)
            ],
            remainder="drop" 
        )
        return self.preprocessor_pipeline

    def process(self, df_input, test_size=0.2, random_state=42):
        """
        Applies cleaning and preprocessing pipeline to the data, then splits into train/test sets.
        """
        df_cleaned = self.clean_data(df_input)
        
        if self.target_column not in df_cleaned.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in cleaned data.")

        X = df_cleaned.drop(columns=[self.target_column])
        y = df_cleaned[self.target_column]

        self.build_pipeline(X) # Build pipeline based on the columns in X

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        X_train_processed = self.preprocessor_pipeline.fit_transform(X_train)
        X_test_processed = self.preprocessor_pipeline.transform(X_test)
        
        try:
            self.feature_names_out_ = self.preprocessor_pipeline.get_feature_names_out()
        except Exception as e:
            print(f"Could not get feature names from ColumnTransformer: {e}")
            # Fallback: construct manually if possible (can be complex)
            num_feature_names = [f for f in self.numerical_features if f in X.columns]
            cat_feature_names_onehot = []
            if hasattr(self.preprocessor_pipeline.named_transformers_["cat"], "named_steps") and \
               hasattr(self.preprocessor_pipeline.named_transformers_["cat"].named_steps["onehot"], "get_feature_names_out"):
                cat_features_used_in_transformer = [f for f in self.categorical_features if f in X.columns]
                if cat_features_used_in_transformer:
                    cat_feature_names_onehot = list(self.preprocessor_pipeline.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_features_used_in_transformer))
            self.feature_names_out_ = num_feature_names + cat_feature_names_onehot
            if not self.feature_names_out_:
                 print("Warning: Feature names could not be constructed.")

        return X_train_processed, X_test_processed, y_train, y_test, self.feature_names_out_

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.dirname(current_script_dir)
    print(f"Testing Preprocessor with project_root: {project_root_for_test}")

    data = {
        "Client ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Age": ["30 years", "150 years", "25 years", "30 years", None, "40 years", "50 years", "60 years", "33 years", "23 years"],
        "Occupation": ["technician", "admin", "services", "unknown", "blue-collar", "retired", "management", "student", "self-employed", "unemployed"],
        "Marital Status": ["married", "single", "married", "divorced", "single", "married", "single", "married", "divorced", "single"],
        "Education Level": ["high.school", "university.degree", "basic.9y", "professional.course", "unknown", "illiterate", "basic.4y", "basic.6y", "university.degree", "high.school"],
        "Credit Default": ["no", "unknown", "no", "no", "yes", "no", "unknown", "no", "no", "unknown"],
        "Housing Loan": ["yes", "no", None, "yes", "no", "unknown", "yes", "no", "yes", "no"],
        "Personal Loan": ["no", "yes", "no", "unknown", "no", "yes", None, "no", "yes", "no"],
        "Contact Method": ["cellular", "telephone", "Cell", "Telephone", "cellular", "telephone", "cellular", "telephone", "Cell", "Telephone"],
        "Campaign Calls": [1, -2, 3, 1, 5, -1, 2, 3, 4, 1],
        "Previous Contact Days": [999, 10, 999, 5, 999, 3, 999, 20, 999, 999],
        "Subscription Status": ["no", "yes", "no", "no", "yes", "no", "yes", "yes", "no", "yes"]
    }
    dummy_df = pd.DataFrame(data)

    print("\nOriginal Dummy DataFrame:")
    print(dummy_df[["Previous Contact Days"]])

    preprocessor = DataPreprocessor(project_root=project_root_for_test)
    
    cleaned_df = preprocessor.clean_data(dummy_df.copy())
    print("\nCleaned DataFrame with Engineered Features:")
    print(cleaned_df[["Previous Contact Days", "PreviouslyContacted", "DaysSinceLastContact", "Age", "Campaign Calls"]].head())
    cleaned_df.info()

    X_train, X_test, y_train, y_test, feature_names = preprocessor.process(dummy_df.copy())
    print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Feature names after processing ({len(feature_names)}): {feature_names}")
    print(f"\nProcessed X_train sample (first 2 rows):\n{X_train[:2]}")

