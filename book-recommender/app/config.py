import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    KAGGLE_BOOKS_DATASET_PATH="zygmunt/goodbooks-10k" #datasource
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "../rag")
    DATA_PATH = os.getenv("DATA_PATH", "../data/raw/books.csv")
    MAIN_COLUMNS = ["authors", "original_title", "average_rating", "language_code"]
    TITLE_COLUMN_NAME = "original_title"

settings = Settings()

keywords = {
            # Blocks
            "credentials": ["senha", "password", "token", "api key", "chave de api", "api",
                            "hack", "hacking", "exploit"], 
            "personal_data": ["cpf", "rg", "dados pessoais"],
            "database_extraction": [
                "all data", "all database", "show me all", "full database",
                "entire database", "dump database"
            ],
            "danger": ["kill", "murder", "suicide", "cocaine", "drugs"],

            # Allowed
            "book_query": ["book", "livro", "novel", "romance", "story", "história"],
            "author_query": ["author", "autor", "written by", "escrito por"],
            "catacters_names": ["give me one name", "autor", "important name", "one import name", 
                                "character", "characters", "character name", "character names"],
            "book_recommendation": ["recommend", "recommendation", "sugira", "indique"],
            "isbn_lookup": ["isbn"],
            "book_info": ["who wrote"]
        }


pii_patterns = {
            "cpf": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "google_api_key": r"\bAIza[0-9A-Za-z\-_]{35}\b",
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        }


invalid_intents = ["credentials", "personal_data", "database_extraction", "danger"]
valid_intents = ["book_recommendation", "livro", "author", "autor", "isbn",
                 "catacters_names", "author_query",
                  "isbn_lookup", "book_info"]