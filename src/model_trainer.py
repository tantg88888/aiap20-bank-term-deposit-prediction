import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:
    """Handles training of different classification models and extracting feature importances."""

    def __init__(self, models_config=None, model_save_dir_relative="models", project_root=None, results_save_dir_relative="results"):
        """
        Initializes the ModelTrainer.

        Args:
            models_config (dict): Configuration for models to train. 
            model_save_dir_relative (str): Relative directory to save trained models from project root.
            project_root (str, optional): Absolute path to the project root. If None, os.getcwd() is assumed.
            results_save_dir_relative (str): Relative directory to save feature importance plots.
        """
        self.project_root = project_root if project_root else os.getcwd()
        self.models_config = models_config if models_config is not None else self.get_default_models_config()
        self.trained_models = {}
        
        self.model_save_dir = os.path.join(self.project_root, model_save_dir_relative)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            print(f"Created directory: {self.model_save_dir}")

        self.results_save_dir = os.path.join(self.project_root, results_save_dir_relative)
        if not os.path.exists(self.results_save_dir):
            os.makedirs(self.results_save_dir) # For feature importance plots
            print(f"Created directory: {self.results_save_dir}")

        self.feature_importances = {}

    def get_default_models_config(self):
        """Returns a default configuration for models if none is provided."""
        return {
            "LogisticRegression": {"C": 1.0, "solver": "liblinear", "random_state": 42, "class_weight": "balanced"},
            "RandomForestClassifier": {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"},
            "GradientBoostingClassifier": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        }

    def _plot_feature_importance(self, importances, feature_names, model_name, top_n=15):
        """Plots and saves feature importances."""
        if feature_names is None or len(feature_names) != len(importances):
            print(f"Warning: Feature names not available or mismatch for {model_name}. Skipping feature importance plot.")
            return

        indices = np.argsort(importances)[::-1]
        top_indices = indices[:top_n]

        plt.figure(figsize=(10, max(6, top_n // 2)))
        plt.title(f"Feature Importances for {model_name} (Top {top_n})")
        plt.barh(range(len(top_indices)), importances[top_indices][::-1], align="center") # Plot in descending order
        plt.yticks(range(len(top_indices)), np.array(feature_names)[top_indices][::-1])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_save_dir, f"{model_name}_feature_importance.png")
        try:
            plt.savefig(plot_path)
            print(f"Feature importance plot for {model_name} saved to {plot_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving feature importance plot for {model_name}: {e}")

    def train_model(self, X_train, y_train, model_name, feature_names=None, params=None):
        """
        Trains a single specified model and extracts feature importances.
        """
        if params is None:
            params = self.models_config.get(model_name, {})

        model = None
        print(f"\nTraining {model_name} with parameters: {params}")

        if model_name == "LogisticRegression":
            model = LogisticRegression(**params)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(**params)
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(**params)
        elif model_name == "SVC":
            # SVC with kernel=\'linear\' has coef_, otherwise no direct feature_importances_
            # For non-linear SVC, permutation importance is an option but more complex here.
            kernel = params.get("kernel", "rbf") # Default to rbf if not specified
            if kernel == "linear":
                 model = SVC(**params, probability=True) # probability for roc_auc
            else:
                 model = SVC(**params, probability=True)
                 print(f"Note: Direct feature importances not available for SVC with kernel {kernel}.")
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier(**params)
            print("Note: Direct feature importances not available for KNeighborsClassifier.")
        else:
            print(f"Error: Model {model_name} not supported.")
            return None
        
        try:
            model.fit(X_train, y_train)
            print(f"{model_name} trained successfully.")
            self.trained_models[model_name] = model
            self.save_model(model, model_name)

            # Get feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                self.feature_importances[model_name] = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                print(f"Feature importances for {model_name}:\n{self.feature_importances[model_name].head()}")
                self._plot_feature_importance(importances, feature_names, model_name)
            elif hasattr(model, "coef_") and model_name == "LogisticRegression": # Also for LinearSVC
                # For linear models, coef_ can be used. For multiple classes, coef_ shape is (n_classes, n_features)
                # For binary, it is (1, n_features)
                importances = model.coef_[0]
                self.feature_importances[model_name] = pd.Series(np.abs(importances), index=feature_names).sort_values(ascending=False)
                print(f"Absolute feature coefficients (importances) for {model_name}:\n{self.feature_importances[model_name].head()}")
                self._plot_feature_importance(np.abs(importances), feature_names, model_name)
            elif hasattr(model, "coef_") and model_name == "SVC" and model.kernel == "linear":
                importances = model.coef_[0]
                self.feature_importances[model_name] = pd.Series(np.abs(importances), index=feature_names).sort_values(ascending=False)
                print(f"Absolute feature coefficients (importances) for {model_name} (LinearSVC):\n{self.feature_importances[model_name].head()}")
                self._plot_feature_importance(np.abs(importances), feature_names, model_name)
            else:
                self.feature_importances[model_name] = "Not available"
                print(f"Feature importances not directly available for {model_name}.")

            return model
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            self.feature_importances[model_name] = f"Error during training: {e}"
            return None

    def train_all_models(self, X_train, y_train, feature_names=None):
        """
        Trains all models specified in the models_config.
        """
        for model_name, params in self.models_config.items():
            self.train_model(X_train, y_train, model_name, feature_names=feature_names, params=params)
        return self.trained_models, self.feature_importances

    def save_model(self, model, model_name):
        """Saves the trained model to a file in model_save_dir."""
        model_path = os.path.join(self.model_save_dir, f"{model_name}.joblib")
        try:
            joblib.dump(model, model_path)
            print(f"Model {model_name} saved to {model_path}")
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")

    def load_model(self, model_name):
        """Loads a trained model from a file."""
        model_path = os.path.join(self.model_save_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"Model {model_name} loaded from {model_path}")
                self.trained_models[model_name] = model
                return model
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
        else:
            print(f"Model file {model_path} not found.")
            return None

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.dirname(current_script_dir)
    print(f"Testing ModelTrainer with project_root: {project_root_for_test}")

    X_train_dummy, y_train_dummy = make_classification(n_samples=100, n_features=5, random_state=42)
    dummy_feature_names = [f"feature_{i}" for i in range(X_train_dummy.shape[1])]

    trainer = ModelTrainer(project_root=project_root_for_test)
    
    print("\n--- Training Single Model (LogisticRegression) ---")
    lr_model = trainer.train_model(X_train_dummy, y_train_dummy, "LogisticRegression", feature_names=dummy_feature_names)

    print("\n--- Training All Configured Models ---")
    trained_models_dict, f_importances = trainer.train_all_models(X_train_dummy, y_train_dummy, feature_names=dummy_feature_names)
    print(f"\nTrained models dictionary: {trained_models_dict}")
    print(f"\nFeature Importances:")
    for name, imp in f_importances.items():
        if isinstance(imp, pd.Series):
            print(f"\n{name}:\n{imp.head()}")
        else:
            print(f"\n{name}: {imp}")

    print(f"\nModels saved in: {os.path.abspath(trainer.model_save_dir)}")
    print(f"Feature importance plots saved in: {os.path.abspath(trainer.results_save_dir)}")

