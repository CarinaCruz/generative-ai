### 📚 Book Recommendation Agent with RAG

An intelligent book recommendation system powered by Retrieval-Augmented Generation (RAG) and LLMs. This system provides personalized book suggestions by combining vector search with generative AI, offering detailed recommendations based on actual book data.

#### ✨ Features

🔍 **Intelligent Search**: Semantic search across book titles, authors, and metadata using vector embeddings

🤖 **AI-Powered Recommendations**: LLM-generated personalized recommendations with reasoning

📊 **Data-Driven**: Based on real book ratings, genres, and descriptions

⚡ **Fast & Scalable**: Built with FastAPI and optimized vector databases

🐳 **Docker Ready**: Containerized for easy deployment (see Docker section)


#### 🚀 Quick Start (Local development)

Prerequisites:

- Python 3.9+
- (Optional) A virtual environment

Local dev steps:

- python3 -m venv .venv
- source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt

Run the API locally:

- uvicorn app:app --reload


#### Environment

Create a `.env` file in project root and provide your sensitive keys. At minimum set:

- GEMINI_API_KEY=<your_google_gemini_api_key>

The project reads environment variables from `.env` when using Docker Compose.


#### Observability (MLflow)

This project now includes optional observability using MLflow. The application logs interactions and retriever traces to an MLflow tracking server when the following environment variables are provided:

- `MLFLOW_TRACKING_URI` — URL for the MLflow tracking server (e.g. `http ://mlflow:5000` when using Docker Compose)
- `MLFLOW_EXPERIMENT` — name of the MLflow experiment (defaults to `book-recommender`)
- `MLFLOW_ARTIFACT_ROOT` — optional artifact root (container-compose maps `./mlflow` to `/mlflow`)

When enabled, the app creates nested runs for each interaction and stores prompt/response artifacts and basic metrics (latency, lengths, etc.).

You can run MLflow locally using the included `docker-compose.yml` (the compose file defines a `mlflow` service), which exposes the MLflow UI at `http ://localhost:5000` and persists artifacts to `./mlflow` on the host.

To run with MLflow via Docker Compose:

- docker compose up --build -d
- open http ://localhost:8000 for the API UI
- open http ://localhost:5000 for the MLflow UI

Alternatively you can start a local MLflow server without Docker:

- pip install mlflow sqlalchemy
- mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5000
- export MLFLOW_TRACKING_URI=http\://localhost:5000
- export MLFLOW_EXPERIMENT=book-recommender

Notes:
- The Dockerfile contains a multi-stage build that includes a `mlflow-server` target used by the Compose file.
- Artifacts and the MLflow SQLite DB are stored under `./mlflow` when using the provided compose setup.


#### Docker (recommended for deployment)

The repository includes a `Dockerfile` and `docker-compose.yml` to build and run the service.

Build and run with Docker Compose (recommended):

- docker compose up --build -d
- docker compose logs web
- docker compose down 
- web browser: http ://localhost:8000/

Build and run the image manually:

- docker build -t book-recommender:latest .
- docker run --env-file .env -p 8000:8000 -v "$(pwd)/data:/app/data" -v "$(pwd)/vdb:/app/vdb" --rm book-recommender:latest

Notes:

- The Compose file maps host port `8000` to container port `8000`.
- Volumes: `./data` and `./vdb` are mounted into `/app/data` and `/app/vdb` so the vectorstore and dataset persist outside the container.
- A healthcheck is configured for the service and probes the root endpoint.
- Use `.dockerignore` to keep large or sensitive files out of the image (the repo includes one).


#### Requirements changes

Observability and the MLflow server require additional packages. The `requirements.txt` was updated to include `mlflow` and `sqlalchemy` for the MLflow server support. Ensure these are installed when running MLflow locally.


#### Dataset

- zygmunt/goodbooks-10k from Kaggle — cleaned and stored at `data/books.csv`

#### API Key

- Store your Google Gemini API Key in `.env` as `GEMINI_API_KEY`.


![alt text](image.png)
