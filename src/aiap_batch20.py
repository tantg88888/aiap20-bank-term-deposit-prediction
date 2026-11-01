"""Entry point for the end-to-end ML pipeline (AIAP Batch 20 technical assessment).

Run from the project root with `bash run.sh` (or `python src/aiap_batch20.py`).
Exits with a non-zero status code if any pipeline step fails, so CI fails loudly.
"""
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")  # headless backend: plots are saved to files, never shown

import pandas as pd

# Project root = parent of the src/ directory that contains this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make the sibling modules in src/ importable regardless of the current working directory
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config_manager import ConfigManager  # noqa: E402
from data_loader import DataLoader  # noqa: E402
from preprocessor import DataPreprocessor  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402
from evaluator import ModelEvaluator  # noqa: E402


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


class PipelineRunner:
    """Orchestrates the end-to-end machine learning pipeline."""

    def __init__(self, config_file_name="config.json"):
        """
        Args:
            config_file_name (str): Name of the configuration file, expected in the project root.
        """
        self.project_root = PROJECT_ROOT
        print(f"PipelineRunner initialised. Project root: {self.project_root}")

        self.config_manager = ConfigManager(config_file_name=config_file_name, project_root=self.project_root)

        dl_config = self.config_manager.get("data_loader", {})
        self.data_loader = DataLoader(
            db_path=dl_config.get("db_path_relative", "data/bmarket.db"),
            db_url=dl_config.get("db_url"),
            project_root=self.project_root,
        )

        pp_config = self.config_manager.get("preprocessor", {})
        self.preprocessor = DataPreprocessor(
            numerical_features=pp_config.get("numerical_features"),
            categorical_features=pp_config.get("categorical_features"),
            target_column=pp_config.get("target_column", "Subscription Status"),
            project_root=self.project_root,
        )

        mt_config = self.config_manager.get("model_trainer", {})
        self.model_trainer = ModelTrainer(
            models_config=mt_config.get("models_config"),
            model_save_dir_relative=mt_config.get("model_save_dir_relative", "models"),
            results_save_dir_relative=self.config_manager.get("evaluator.results_save_dir_relative", "results"),
            project_root=self.project_root,
        )

        ev_config = self.config_manager.get("evaluator", {})
        self.evaluator = ModelEvaluator(
            results_save_dir_relative=ev_config.get("results_save_dir_relative", "results"),
            project_root=self.project_root,
        )

    def run_pipeline(self):
        """Executes the full ML pipeline. Raises PipelineError on failure."""
        print("Starting ML pipeline...")

        # 1. Load data
        print("\n--- Step 1: Loading Data ---")
        dl_config = self.config_manager.get("data_loader", {})
        raw_data_df = self.data_loader.load_data(table_name=dl_config.get("table_name", "bank_marketing"))
        if raw_data_df is None or raw_data_df.empty:
            raise PipelineError("Data loading failed (no data returned).")
        print(f"Successfully loaded {len(raw_data_df)} rows.")

        # 2. Preprocess data
        print("\n--- Step 2: Preprocessing Data ---")
        pp_config = self.config_manager.get("preprocessor", {})
        X_train, X_test, y_train, y_test, feature_names_out = self.preprocessor.process(
            raw_data_df,
            test_size=pp_config.get("test_size", 0.2),
            random_state=pp_config.get("random_state", 42),
        )
        print(f"Data preprocessed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        if feature_names_out is not None and len(feature_names_out) > 0:
            print(f"Processed feature names ({len(feature_names_out)}): {list(feature_names_out[:10])}...")
        else:
            print("Warning: processed feature names not available.")

        # 3. Train models
        print("\n--- Step 3: Training Models ---")
        trained_models, _feature_importances = self.model_trainer.train_all_models(
            X_train, y_train, feature_names=feature_names_out
        )
        expected = list(self.model_trainer.models_config.keys())
        failed = [name for name in expected if name not in trained_models]
        if not trained_models or failed:
            raise PipelineError(f"Model training failed for: {failed or expected}")
        print(f"Successfully trained {len(trained_models)} model(s): {list(trained_models.keys())}")

        # 4. Evaluate models
        print("\n--- Step 4: Evaluating Models ---")
        evaluation_summary_df = self.evaluator.evaluate_all_models(trained_models, X_test, y_test)
        if evaluation_summary_df.empty:
            raise PipelineError("Evaluation produced no results.")

        metric_cols = [c for c in evaluation_summary_df.columns if c != "Model"]
        numeric = evaluation_summary_df[metric_cols].apply(pd.to_numeric, errors="coerce")
        if numeric.isnull().any().any():
            print(evaluation_summary_df)
            raise PipelineError("One or more evaluation metrics could not be computed.")

        # Compact summary table (also saved as results/summary.csv)
        summary = pd.concat([evaluation_summary_df[["Model"]], numeric.round(4)], axis=1)
        summary.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        summary_path = os.path.join(self.evaluator.results_save_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)

        print("\n" + "=" * 72)
        print("FINAL EVALUATION SUMMARY (test set, positive class = subscribed)")
        print("=" * 72)
        print(summary.to_string(index=False))
        print("=" * 72)
        print(f"Summary saved to {summary_path}")

        print("\nML pipeline finished successfully.")
        return summary


def main():
    try:
        PipelineRunner(config_file_name="config.json").run_pipeline()
    except Exception as e:  # noqa: BLE001 - report any failure and exit non-zero
        print(f"\nPIPELINE FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
