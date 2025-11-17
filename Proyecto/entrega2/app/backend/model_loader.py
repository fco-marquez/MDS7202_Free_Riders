"""
Module for loading the trained model from MLflow or local storage.
Implements fallback strategy: MLflow first, then local pickle file.
"""
import os
import logging
import mlflow
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages the prediction model with fallback strategy."""

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "sodai_drinks_prediction",
        local_model_path: str = "/models/best_model.pkl"
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.local_model_path = local_model_path
        self.model = None
        self.model_info = {}

    def load_model(self):
        """
        Load model with fallback strategy:
        1. Try MLflow (best model by val_recall)
        2. If fails, try local pickle file
        3. If both fail, raise exception
        """
        # Try MLflow first
        try:
            logger.info(f"Attempting to load model from MLflow at {self.mlflow_tracking_uri}")
            self.model = self._load_from_mlflow()
            self.model_info['source'] = 'mlflow'
            logger.info("✓ Model loaded successfully from MLflow")
            return self.model
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}")

        # Fallback to local file
        try:
            logger.info(f"Attempting to load model from local file: {self.local_model_path}")
            self.model = self._load_from_local()
            self.model_info['source'] = 'local'
            logger.info("✓ Model loaded successfully from local file")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load from local file: {e}")
            raise RuntimeError(
                "Failed to load model from both MLflow and local storage. "
                "Please ensure the model has been trained and saved."
            )

    def _load_from_mlflow(self):
        """Load best model from MLflow experiment."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found in MLflow")

        # Search for best run by val_recall
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName LIKE 'final_model%'",
            order_by=["metrics.val_recall DESC"],
            max_results=1
        )

        if runs.empty:
            raise ValueError(f"No runs found in experiment '{self.experiment_name}'")

        best_run = runs.iloc[0]
        run_id = best_run.run_id

        # Store model metadata
        self.model_info.update({
            'run_id': run_id,
            'val_recall': best_run.get('metrics.val_recall'),
            'val_precision': best_run.get('metrics.val_precision'),
            'val_f1': best_run.get('metrics.val_f1'),
            'val_auc_pr': best_run.get('metrics.val_auc_pr'),
        })

        # Load model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"Loaded model from run {run_id} with val_recall={best_run.get('metrics.val_recall'):.4f}")

        return model

    def _load_from_local(self):
        """Load model from local pickle file."""
        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(f"Local model file not found: {self.local_model_path}")

        model = joblib.load(self.local_model_path)

        # Store basic info
        self.model_info.update({
            'path': self.local_model_path,
        })

        return model

    def get_model(self):
        """Get the loaded model. Load if not already loaded."""
        if self.model is None:
            self.load_model()
        return self.model

    def get_model_info(self):
        """Get information about the loaded model."""
        return self.model_info

    def reload_model(self):
        """Force reload the model (useful after retraining)."""
        logger.info("Reloading model...")
        self.model = None
        self.model_info = {}
        return self.load_model()


# Singleton instance
_model_loader = None

def get_model_loader():
    """Get singleton ModelLoader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "sodai_drinks_prediction"),
            local_model_path=os.getenv("MODEL_PATH", "/models/best_model.pkl")
        )
    return _model_loader
