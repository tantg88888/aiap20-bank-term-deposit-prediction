from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelEvaluator:
    """Handles evaluation of trained classification models."""

    def __init__(self, results_save_dir_relative="results", project_root=None):
        """
        Initializes the ModelEvaluator.

        Args:
            results_save_dir_relative (str): Relative directory from project root to save evaluation results.
            project_root (str, optional): Absolute path to the project root. If None, os.getcwd() is assumed.
        """
        self.project_root = project_root if project_root else os.getcwd()
        self.results_save_dir = os.path.join(self.project_root, results_save_dir_relative)
        
        if not os.path.exists(self.results_save_dir):
            os.makedirs(self.results_save_dir)
            print(f"Created directory: {self.results_save_dir}")

    def evaluate_model(self, model, model_name, X_test, y_test):
        """
        Evaluates a single trained model.
        """
        print(f"\n--- Evaluating {model_name} ---")
        y_pred = model.predict(X_test)
        
        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            # Specify pos_label=1 if your positive class is 1 and it's not inferred correctly
            metrics["precision"] = precision_score(y_test, y_pred, zero_division=0, pos_label=1)
            metrics["recall"] = recall_score(y_test, y_pred, zero_division=0, pos_label=1)
            metrics["f1_score"] = f1_score(y_test, y_pred, zero_division=0, pos_label=1)
            
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            else:
                metrics["roc_auc"] = "N/A (predict_proba not available)"

            # Single quotes inside the f-strings: nested double quotes are a SyntaxError before Python 3.12
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision (for class 1): {metrics['precision']:.4f}")
            print(f"Recall (for class 1): {metrics['recall']:.4f}")
            print(f"F1-score (for class 1): {metrics['f1_score']:.4f}")
            roc_auc = metrics["roc_auc"]
            print(f"ROC AUC: {roc_auc if isinstance(roc_auc, str) else format(roc_auc, '.4f')}")

            print("\nClassification Report:")
            # Ensure target_names are appropriate if y_test is 0 and 1
            report = classification_report(y_test, y_pred, zero_division=0, target_names=["No (0)", "Yes (1)"])
            print(report)
            metrics["classification_report"] = report

            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm
            self.plot_confusion_matrix(cm, model_name, class_names=["No (0)", "Yes (1)"])

        except Exception as e:
            print(f"Error during evaluation of {model_name}: {e}")
            for mkey in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "classification_report", "confusion_matrix"]:
                if mkey not in metrics:
                    metrics[mkey] = f"Error: {e}"
        
        return metrics

    def plot_confusion_matrix(self, cm, model_name, class_names):
        """
        Plots and saves the confusion matrix.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix for {model_name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plot_path = os.path.join(self.results_save_dir, f"{model_name}_confusion_matrix.png")
        try:
            plt.savefig(plot_path)
            print(f"Confusion matrix plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving confusion matrix plot for {model_name}: {e}")

    def evaluate_all_models(self, trained_models, X_test, y_test):
        """
        Evaluates all trained models.
        """
        all_metrics_summary = []
        for model_name, model in trained_models.items():
            if model is not None:
                metrics = self.evaluate_model(model, model_name, X_test, y_test)
                metrics_summary_row = {
                    "Model": model_name,
                    "Accuracy": metrics.get("accuracy", "N/A"),
                    "Precision (Class 1)": metrics.get("precision", "N/A"),
                    "Recall (Class 1)": metrics.get("recall", "N/A"),
                    "F1-score (Class 1)": metrics.get("f1_score", "N/A"),
                    "ROC AUC": metrics.get("roc_auc", "N/A")
                }
                all_metrics_summary.append(metrics_summary_row)
            else:
                print(f"Skipping evaluation for {model_name} as it was not trained successfully.")
        
        results_df = pd.DataFrame(all_metrics_summary)
        print("\n--- Overall Model Evaluation Summary ---")
        print(results_df)
        
        summary_path = os.path.join(self.results_save_dir, "evaluation_summary.csv")
        try:
            results_df.to_csv(summary_path, index=False)
            print(f"Evaluation summary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving evaluation summary: {e}")
            
        return results_df

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.dirname(current_script_dir)
    print(f"Testing ModelEvaluator with project_root: {project_root_for_test}")

    X_dummy, y_dummy = make_classification(n_samples=200, n_features=5, random_state=42, n_classes=2)
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dummy, y_dummy, test_size=0.25, random_state=42)

    lr = LogisticRegression(solver="liblinear", random_state=42)
    lr.fit(X_train_d, y_train_d)
    dummy_trained_models = {"LogisticRegression": lr}

    evaluator = ModelEvaluator(project_root=project_root_for_test)
    summary_df = evaluator.evaluate_all_models(dummy_trained_models, X_test_d, y_test_d)

    print(f"\nEvaluation plots and summary CSV saved in: {os.path.abspath(evaluator.results_save_dir)}")

