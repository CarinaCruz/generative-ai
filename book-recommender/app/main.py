import sys
import logging
import os
from dotenv import load_dotenv

from rag import RAGSystem
from llm import LLMProvider
from agent import BookRecommendationAgent
from guardrails import SecurityGuardrails

guardrails = SecurityGuardrails()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Environment & logging
# --------------------------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# --------------------------------------------------
# Initialization (runs once)
# --------------------------------------------------
def get_rag():
    rag = RAGSystem()
    rag.initialize_vectorstore()
    return rag 

def initialize_app():    
    rag_system = get_rag()
    logger.info("RAG system initialized successfully.")
    llm_provider = LLMProvider()
    agent = BookRecommendationAgent(rag_system, llm_provider)
    logger.info("Book Recommendation Agent initialized successfully.")
    return agent

def run_cli(agent: BookRecommendationAgent):
    question = input(">>> ")
    result = guardrails.check_user_input(question)
    if result.allowed:
        answer = agent.ask(question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    logger.info("📚 Book Recommendation CLI")
    agent = initialize_app()
    run_cli(agent)
