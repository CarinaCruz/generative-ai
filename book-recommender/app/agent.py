from rag import RAGSystem
from llm import LLMProvider
from observability import init_mlflow, log_interaction
import time
import logging
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# Initialize MLflow once at module import
try:
    init_mlflow()
except Exception as e:
    logger.info(f"Failed to initialize MLflow: {e}")
    pass

class BookRecommendationAgent:
    def __init__(self, rag_system: RAGSystem, llm_provider: LLMProvider):
        self.rag_system = rag_system
        self.llm_provider = llm_provider
        self.prompt_template = llm_provider.create_prompt_template()

    @mlflow.trace(name="agent.ask", span_type="LANGUAGE_MODEL")
    def ask(self, question: str, k: int = 5):
        # Search for similar books
        similar_books = self.rag_system.get_similar_books(question, k=k)

        if not similar_books:
            return "I couldn't find any books matching your query. Please try with different keywords."

        # Generate response
        prompt = self.prompt_template.format(books_list=similar_books, question=question)

        # Time the model invocation
        start = time.time()
        try:
            response = self.llm_provider.llm.invoke(prompt)
            content = response.content
        except Exception as e:
            # Log failure to MLflow and re-raise
            try:
                log_interaction(
                    question=question,
                    prompt=prompt,
                    response=str(e),
                    model=self.llm_provider.llm.model,
                    provider="google-genai",
                    retrieved_books=similar_books,
                    metrics={"error": 1}
                )
            except Exception as e:
                logger.info(f"Failed to log error interaction to MLflow: {e}")
                pass
            raise
        finally:
            latency = time.time() - start

        # Best-effort observability logging
        try:
            log_interaction(
                question=question,
                prompt=prompt,
                response=content,
                model=getattr(self.llm_provider.llm, "model", "unknown"),
                provider="google-genai",
                retrieved_books=similar_books,
                metrics={"latency_seconds": latency, "prompt_len": len(prompt), "response_len": len(content)}
            )
        except Exception:
            pass

        return content
