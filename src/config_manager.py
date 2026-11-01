import json
import os

class ConfigManager:
    """Handles loading and accessing pipeline configuration from a JSON file."""

    def __init__(self, config_file_name="config.json", project_root=None):
        """
        Initializes the ConfigManager.

        Args:
            config_file_name (str): Name of the JSON configuration file (e.g., "config.json").
            project_root (str, optional): Absolute path to the project root. 
                                        If None, os.getcwd() is assumed.
        """
        self.project_root = project_root if project_root else os.getcwd()
        self.config_file_name = config_file_name
        self.config_full_path = os.path.join(self.project_root, self.config_file_name)
        self.config = None
        self.load_config()

    def load_config(self):
        """Loads the configuration from the JSON file located at project_root/config_file_name."""
        try:
            if not os.path.exists(self.config_full_path):
                print(f"Warning: Config file {self.config_full_path} not found. Creating a default config.json at project root.")
                self.config = self._get_default_config()
                self.save_config(self.config, self.config_file_name) # Saves to project_root/config_file_name
            else:
                with open(self.config_full_path, "r") as f:
                    self.config = json.load(f)
                print(f"Configuration loaded successfully from {self.config_full_path}")
        except FileNotFoundError: # Should be caught by the exists check, but as a safeguard
            print(f"Error: Configuration file {self.config_full_path} not found. Using default configuration.")
            self.config = self._get_default_config()
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.config_full_path}. Using default configuration.")
            self.config = self._get_default_config()
        except Exception as e:
            print(f"An error occurred loading config: {e}. Using default configuration.")
            self.config = self._get_default_config()

    def _get_default_config(self):
        """Provides a default configuration structure with paths relative to project root."""
        return {
            "data_loader": {
                "db_path_relative": "data/bmarket.db", # Path relative to project root
                "db_url": "https://techassessment.blob.core.windows.net/aiap20-assessment-data/bmarket.db",
                "table_name": "bank_marketing"
            },
            "preprocessor": {
                "numerical_features": ["Age", "Campaign Calls"], # Previous Contact Days will be handled differently
                "categorical_features": [
                    "Occupation", "Marital Status", "Education Level", 
                    "Credit Default", "Housing Loan", "Personal Loan", "Contact Method",
                    "PreviouslyContacted" # New engineered feature
                ],
                "target_column": "Subscription Status",
                "test_size": 0.2,
                "random_state": 42
            },
            "model_trainer": {
                "models_config": {
                    "LogisticRegression": {"C": 1.0, "solver": "liblinear", "random_state": 42, "class_weight": "balanced"},
                    "RandomForestClassifier": {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"},
                    "GradientBoostingClassifier": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
                },
                "model_save_dir_relative": "models" # Path relative to project root
            },
            "evaluator": {
                "results_save_dir_relative": "results" # Path relative to project root
            }
        }

    def get(self, key, default=None):
        """Retrieves a configuration value for a given key from the loaded config."""
        # Navigate nested keys if key is like "data_loader.db_path"
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def save_config(self, config_data, file_name_in_root):
        """Saves configuration data to a JSON file in the project root."""
        save_path = os.path.join(self.project_root, file_name_in_root)
        try:
            with open(save_path, "w") as f:
                json.dump(config_data, f, indent=4)
            print(f"Configuration saved to {save_path}")
        except Exception as e:
            print(f"Error saving configuration to {save_path}: {e}")

if __name__ == "__main__":
    # Test ConfigManager
    # Assumes this script is in src/ and project_root is one level up.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.dirname(current_script_dir)
    print(f"Testing ConfigManager with project_root: {project_root_for_test}")

    # Create a dummy config.json for testing if it doesn_t exist in project root
    test_config_name = "config_test.json"
    test_config_full_path = os.path.join(project_root_for_test, test_config_name)
    
    if os.path.exists(test_config_full_path):
        os.remove(test_config_full_path) # Clean up before test
        print(f"Removed existing {test_config_full_path} for fresh test.")

    config_manager = ConfigManager(config_file_name=test_config_name, project_root=project_root_for_test)
    
    print("\nLoaded Configuration (should be default first time):")
    print(json.dumps(config_manager.config, indent=4))

    print(f"\nData loader DB relative path: {config_manager.get('data_loader.db_path_relative')}")
    print(f"Random Forest params: {config_manager.get('model_trainer.models_config.RandomForestClassifier')}")
    print(f"A non-existent key: {config_manager.get('non_existent_key	', 'default_value_for_non_existent')}")

    # Modify and save
    config_manager.config["data_loader"]["table_name"] = "new_bank_table"
    config_manager.save_config(config_manager.config, test_config_name)

    # Load again to see if changes persisted
    config_manager_reloaded = ConfigManager(config_file_name=test_config_name, project_root=project_root_for_test)
    print("\nReloaded Configuration:")
    print(json.dumps(config_manager_reloaded.config, indent=4))
    print(f"Reloaded table name: {config_manager_reloaded.get('data_loader.table_name')}")

    # Clean up dummy test config
    if os.path.exists(test_config_full_path):
        os.remove(test_config_full_path)
        print(f"Cleaned up {test_config_full_path}")

