import os
import hashlib
import mlflow
import logging
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("observability")


def init_mlflow(experiment_name: str = "book-recommender"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT", experiment_name)
    mlflow.set_experiment(experiment)


def _prompt_version(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def log_interaction(question: str,
                    prompt: str,
                    response: str,
                    model: str,
                    provider: str,
                    retrieved_books: str | None = None,
                    metrics: dict | None = None,
                    tags: dict | None = None) -> str:

    prompt_ver = _prompt_version(prompt)
    content_hash = hashlib.sha256((prompt + (response or "")).encode("utf-8")).hexdigest()[:8]

    with mlflow.start_run(nested=True, run_name=f"interaction_{content_hash}") as run:
        mlflow.set_tag("model", model)
        mlflow.set_tag("provider", provider)
        mlflow.set_tag("prompt_version", prompt_ver)

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))

        mlflow.log_param("user_question", question)
        mlflow.log_param("prompt_length", len(prompt))
        mlflow.log_param("response_length", len(response) if response else 0)

        # Instancia DENTRO da função — depois do init_mlflow já ter configurado a URI
        client = MlflowClient()
        try:
            client.log_text(run.info.run_id, prompt, f"prompt_{prompt_ver}.txt")
            client.log_text(run.info.run_id, response or "", f"response_{prompt_ver}.txt")
            if retrieved_books:
                client.log_text(run.info.run_id, retrieved_books, f"retrieved_books_{prompt_ver}.txt")
        except Exception as e:
            logger.info(f"Failed to log artifacts to MLflow: {e}")

        if metrics:
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    mlflow.log_param(k, str(v))

        return run.info.run_id