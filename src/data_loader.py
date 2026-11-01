import sqlite3
import pandas as pd
import os
import requests  # For downloading the dataset

DEFAULT_DB_URL = "https://techassessment.blob.core.windows.net/aiap20-assessment-data/bmarket.db"


class DataLoader:
    """Handles loading data from the SQLite database, with auto-download capability."""

    def __init__(self, db_path="data/bmarket.db", db_url=None, project_root=None):
        """
        Initializes the DataLoader.

        Args:
            db_path (str): Path to the SQLite database file, relative to the project root (or absolute).
            db_url (str, optional): URL to download the database from if not found.
            project_root (str, optional): Absolute path to the project root. If None, os.getcwd() is assumed.
        """
        self.project_root = project_root if project_root else os.getcwd()
        self.db_path_relative = db_path
        # Resolve against the project root so the pipeline works from any working directory
        self.db_full_path = db_path if os.path.isabs(db_path) else os.path.join(self.project_root, db_path)
        self.db_url = db_url if db_url else DEFAULT_DB_URL

    def _ensure_db_exists(self):
        """Ensures the database file exists, downloading it if necessary."""
        if os.path.exists(self.db_full_path):
            return True

        print(f"Database file not found at {self.db_full_path}.")
        if not self.db_url:
            print("No download URL provided. Please place the database manually.")
            return False

        print(f"Attempting to download from {self.db_url}...")
        tmp_path = self.db_full_path + ".part"
        try:
            data_dir = os.path.dirname(self.db_full_path)
            if data_dir:
                os.makedirs(data_dir, exist_ok=True)

            with requests.get(self.db_url, stream=True, timeout=120) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, self.db_full_path)
            print(f"Database downloaded successfully to {self.db_full_path}.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading database: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    def load_data(self, table_name="bank_marketing"):
        """
        Loads data from the specified table in the SQLite database.

        Args:
            table_name (str): The name of the table to load data from.

        Returns:
            pandas.DataFrame: The loaded data as a DataFrame, or None if an error occurs.
        """
        if not self._ensure_db_exists():
            return None

        try:
            conn = sqlite3.connect(self.db_full_path)
            try:
                query = f'SELECT * FROM "{table_name}"'
                df = pd.read_sql_query(query, conn)
            finally:
                conn.close()
            print(f"Data loaded successfully from {table_name} in {self.db_full_path}.")
            return df
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return None


if __name__ == "__main__":
    # Quick manual check: download (if needed) and load the dataset.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_test = os.path.dirname(current_script_dir)
    print(f"Testing DataLoader with project_root: {project_root_for_test}")

    loader = DataLoader(project_root=project_root_for_test)
    data_df = loader.load_data()
    if data_df is not None:
        print(data_df.head())
        data_df.info()
    else:
        print("Data loading failed.")
