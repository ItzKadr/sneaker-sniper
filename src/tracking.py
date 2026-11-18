"""Fail-safe MLflow wrapper: a tracking error is logged, never raised.

The store is local to the repo: ``sqlite:///<repo>/mlflow.db`` for runs/metrics and
``<repo>/mlruns/`` for artifacts. Browse with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Context manager around one MLflow run. Every method swallows its own errors."""

    def __init__(self, repo_dir: str | Path, experiment: str = "sneaker-sniper", run_name: str | None = None, enabled: bool = True) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.experiment = experiment
        self.run_name = run_name
        self.enabled = enabled
        self._mlflow: Any = None
        self.run_id: str | None = None

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> "MLflowTracker":
        if not self.enabled:
            return self
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            db = (self.repo_dir / "mlflow.db").as_posix()
            mlflow.set_tracking_uri(f"sqlite:///{db}")
            exp = mlflow.get_experiment_by_name(self.experiment)
            if exp is None:
                mlflow.create_experiment(self.experiment, artifact_location=(self.repo_dir / "mlruns").as_uri())
            mlflow.set_experiment(self.experiment)
            run = mlflow.start_run(run_name=self.run_name)
            self.run_id = run.info.run_id
            self._mlflow = mlflow
            logger.info("MLflow run %s started (sqlite:///%s)", self.run_id, db)
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("MLflow disabled (start failed: %s)", exc)
            self._mlflow = None
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        except Exception as err:  # pragma: no cover
            logger.warning("MLflow end_run failed: %s", err)

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    # --------------------------------------------------------------- logging
    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.active:
            return
        try:
            clean = {k: (str(v)[:500] if not isinstance(v, (int, float, bool)) else v) for k, v in params.items()}
            self._mlflow.log_params(clean)
        except Exception as exc:
            logger.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if not self.active:
            return
        try:
            safe = {self._safe_key(k): float(v) for k, v in metrics.items() if v is not None and v == v}
            self._mlflow.log_metrics(safe, step=step)
        except Exception as exc:
            logger.warning("MLflow log_metrics failed: %s", exc)

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        if not self.active:
            return
        try:
            self._mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("MLflow log_artifacts failed: %s", exc)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.active:
            return
        try:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("MLflow log_artifact failed: %s", exc)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.active:
            return
        try:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
        except Exception as exc:
            logger.warning("MLflow set_tags failed: %s", exc)

    @staticmethod
    def _safe_key(key: str) -> str:
        """MLflow metric names allow alphanumerics, _ - . / and spaces; map '@' -> '_at_'."""
        return key.replace("@", "_at_")
