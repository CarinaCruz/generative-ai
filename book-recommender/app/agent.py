from rag import RAGSystem
from llm import LLMProvider

class BookRecommendationAgent:
    def __init__(self, rag_system: RAGSystem, llm_provider: LLMProvider):
        self.rag_system = rag_system
        self.llm_provider = llm_provider
        self.prompt_template = llm_provider.create_prompt_template()

    def ask(self, question: str, k: int = 5):
        # Search for similar books
        similar_books = self.rag_system.get_similar_books(question, k=k)

        if not similar_books:
            return "I couldn't find any books matching your query. Please try with different keywords."

        # Generate response
        prompt = self.prompt_template.format(books_list=similar_books, question=question)
        response = self.llm_provider.llm.invoke(prompt)

        return response.content
