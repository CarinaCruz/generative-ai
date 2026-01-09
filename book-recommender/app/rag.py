import os
import shutil
import logging
import pandas as pd

from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


class RAGSystem:
    """
    Core RAG system responsible for:
    - Loading data
    - Creating documents
    - Building or loading a vectorstore
    - Performing similarity search
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.embedding_model = settings.EMBEDDING_MODEL

        self.embeddings = self._load_embeddings()
        self.vectorstore: Chroma | None = None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        self.logger.info("Loading embedding model...")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64}
        )

    # ------------------------------------------------------------------
    # Data loading & document creation
    # ------------------------------------------------------------------
    def load_data(self, data_path: str) -> pd.DataFrame:
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        df = (
            df
            .dropna(subset=[settings.TITLE_COLUMN_NAME])
            .loc[:, settings.MAIN_COLUMNS]
        )

        return df

    def create_documents(self, df: pd.DataFrame) -> list[Document]:
        self.logger.info("Creating LangChain documents...")

        documents = []
        for _, row in df.iterrows():
            content = f"Book title: {row[settings.TITLE_COLUMN_NAME]}"

            metadata = {
                "title": row[settings.TITLE_COLUMN_NAME],
                "authors": row.get("authors", "unknown"),
                "rating": float(row.get("average_rating", -1)),
                "language": row.get("language_code", "unknown"),
            }

            documents.append(
                Document(page_content=content, metadata=metadata)
            )

        self.logger.info(f"Created {len(documents)} documents")
        return documents

    # ------------------------------------------------------------------
    # Vectorstore lifecycle
    # ------------------------------------------------------------------
    def vectorstore_exists(self) -> bool:
        return os.path.exists(self.persist_dir) and os.listdir(self.persist_dir)

    def initialize_vectorstore(
        self,
        documents: list[Document] = None,
        force_recreate: bool = False
    ) -> Chroma:
        """
        Initialize the vectorstore.

        - If it exists and force_recreate=False → load from disk
        - If force_recreate=True → rebuild from documents
        """

        if force_recreate:
            self._delete_vectorstore()

        if self.vectorstore_exists():
            self.logger.info("Loading existing vectorstore from disk...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            return self.vectorstore

        if documents is None:
            raise ValueError(
                "Documents must be provided when creating a new vectorstore."
            )

        self.logger.info("Creating new vectorstore (this may take a while)...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        self.vectorstore.persist()
        self.logger.info("Vectorstore created and persisted.")

        return self.vectorstore

    def _delete_vectorstore(self):
        if os.path.exists(self.persist_dir):
            self.logger.warning("Deleting existing vectorstore...")
            shutil.rmtree(self.persist_dir)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search_books(self, query: str, k: int = 5):
        if not self.vectorstore:
            raise RuntimeError(
                "Vectorstore not initialized. Call initialize_vectorstore first."
            )

        return self.vectorstore.similarity_search(query, k=k)
