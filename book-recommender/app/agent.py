from rag import RAGSystem
from llm import LLMProvider

class BookRecommendationAgent:
    def __init__(self, rag_system: RAGSystem, llm_provider: LLMProvider):
        self.rag_system = rag_system
        self.llm_provider = llm_provider
        self.prompt_template = llm_provider.create_prompt_template()

    def ask(self, question: str, k: int = 5):
        # Search for similar books
        similar_books = self.rag_system.search_books(question, k=k)

        if not similar_books:
            return "I couldn't find any books matching your query. Please try with different keywords."

        # Prepare book information
        books_info = []
        for i, book in enumerate(similar_books, 1):
            books_info.append({
                'title': book.metadata.get('title', 'Unknown'),
                'author': book.metadata.get('authors', 'Unknown'),
                'rating': book.metadata.get('rating', 0),
                'language': book.metadata.get('language', 'Unknown')
            })

        # Format books list for prompt
        books_formatted = [
            f"{i}. {book['title']} by {book['author']} (Rating: {book['rating']}/5, Language: {book['language']})"
            for i, book in enumerate(books_info, 1)
        ]
        books_list_str = "\n".join(books_formatted)

        # Generate response
        prompt = self.prompt_template.format(books_list=books_list_str, question=question)
        response = self.llm_provider.llm.invoke(prompt)

        return response.content