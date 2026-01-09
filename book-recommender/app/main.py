import sys
import logging
import os
from dotenv import load_dotenv

from rag import RAGSystem
from llm import LLMProvider
from agent import BookRecommendationAgent
from guardrails import SecurityGuardrails

guardrails = SecurityGuardrails()

# --------------------------------------------------
# Environment & logging
# --------------------------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Initialization (runs once)
# --------------------------------------------------
def get_rag():
    rag = RAGSystem()

    if not rag.vectorstore_exists():
        df = rag.load_data("data/raw/books.csv")
        docs = rag.create_documents(df)
        rag.initialize_vectorstore(docs)
    else:
        rag.initialize_vectorstore()
    return rag 


def initialize_app():
    logger.info("Initializing RAG system...")

    rag_system = get_rag()
    llm_provider = LLMProvider()
    agent = BookRecommendationAgent(rag_system, llm_provider)

    try:
        logger.info("Loading data and initializing vectorstore...")
        df = rag_system.load_data("../data/raw/books.csv")
        documents = rag_system.create_documents(df)
        rag_system.initialize_vectorstore(documents, force_recreate=False)
        logger.info("Vectorstore initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {e}")
        sys.exit(1)

    return agent

def run_cli(agent: BookRecommendationAgent):
    print("📚 Book Recommendation CLI")
    question = input(">>> ")
    result = guardrails.check_user_input(question)
    if result.allowed:
        answer = agent.ask(question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    agent = initialize_app()
    run_cli(agent)
