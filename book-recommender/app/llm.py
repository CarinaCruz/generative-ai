from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from prompts import SIMPLE_AGENT_BOOK, INPUT_VARIABLES
from config import settings

class LLMProvider:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_retries=2
        )

    def get_llm(self):
        return self.llm

    @staticmethod
    def create_prompt_template():
        return PromptTemplate(template=SIMPLE_AGENT_BOOK,
                               input_variables=INPUT_VARIABLES)