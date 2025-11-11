"""
MLflow Configuration Module
============================
Centralizes MLflow configuration and helper functions for experiment tracking.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME_BASE = "sodai_drinks_prediction"


def setup_mlflow():
    """
    Initialize MLflow tracking URI.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment

    Returns
    -------
    str
        Experiment ID
    """
    setup_mlflow()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id


def log_params_from_dict(params: Dict[str, Any]):
    """
    Log multiple parameters from dictionary.

    Parameters
    ----------
    params : dict
        Dictionary of parameters to log
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics_from_dict(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log multiple metrics from dictionary.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics to log
    step : int, optional
        Step number for metric versioning
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def log_figure(fig: plt.Figure, filename: str, close: bool = True):
    """
    Log matplotlib figure to MLflow.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to log
    filename : str
        Filename for the artifact
    close : bool
        Whether to close the figure after logging
    """
    mlflow.log_figure(fig, filename)
    if close:
        plt.close(fig)


def log_dict_as_artifact(data: Dict[str, Any], filename: str):
    """
    Log dictionary as JSON artifact.

    Parameters
    ----------
    data : dict
        Dictionary to log
    filename : str
        Filename for the artifact (should end in .json)
    """
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2)
        temp_path = f.name

    mlflow.log_artifact(temp_path, artifact_path=os.path.dirname(filename))
    os.unlink(temp_path)


def get_best_run_from_experiment(
    experiment_name: str, metric: str = "val_recall"
) -> Optional[mlflow.entities.Run]:
    """
    Get the best run from an experiment based on a metric.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    metric : str
        Metric to optimize (default: val_recall)

    Returns
    -------
    mlflow.entities.Run or None
        Best run or None if no runs found
    """
    setup_mlflow()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if len(runs) == 0:
        print(f"No runs found in experiment {experiment_name}")
        return None

    run_id = runs.iloc[0]["run_id"]
    return mlflow.get_run(run_id)


def load_model_from_mlflow(experiment_name: str, model_name: str = "model") -> Any:
    """
    Load the best model from MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    model_name : str
        Name of the model artifact (default: "model")

    Returns
    -------
    Any
        Loaded model
    """
    best_run = get_best_run_from_experiment(experiment_name)

    if best_run is None:
        raise ValueError(f"No runs found in experiment {experiment_name}")

    model_uri = f"runs:/{best_run.info.run_id}/{model_name}"
    print(f"Loading model from: {model_uri}")

    return mlflow.sklearn.load_model(model_uri)


def get_experiment_name(suffix: str = "") -> str:
    """
    Generate experiment name with optional suffix.

    Parameters
    ----------
    suffix : str
        Suffix to add to experiment name

    Returns
    -------
    str
        Full experiment name
    """
    if suffix:
        return f"{EXPERIMENT_NAME_BASE}_{suffix}"
    return EXPERIMENT_NAME_BASE
