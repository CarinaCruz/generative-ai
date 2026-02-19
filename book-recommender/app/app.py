import os
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from rag import RAGSystem
from llm import LLMProvider
from agent import BookRecommendationAgent
from guardrails import SecurityGuardrails

# --------------------------------------------------
# Environment & logging
# --------------------------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("book-rag-api")

guardrails = SecurityGuardrails()

# --------------------------------------------------
# App state
# --------------------------------------------------
class AppState:
    rag_system: RAGSystem | None = None
    agent: BookRecommendationAgent | None = None


state = AppState()

# --------------------------------------------------
# Lifespan (startup / shutdown)
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting Book Recommendation API...")

        rag_system = RAGSystem()
        llm_provider = LLMProvider()
        agent = BookRecommendationAgent(rag_system, llm_provider)

        logger.info("Loading data and initializing vectorstore...")
        rag_system.initialize_vectorstore(
            force_recreate=False
        )
        state.rag_system = rag_system
        state.agent = agent

        logger.info("Startup completed successfully.")
        yield

    except Exception as e:
        logger.exception("Startup failed.")
        raise RuntimeError("Failed to initialize application") from e

    finally:
        logger.info("Shutting down Book Recommendation API...")


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Book Recommendation API",
    version="1.0.0",
    description="RAG-based book recommendation system",
    lifespan=lifespan
)

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    question = request.question
    result = guardrails.check_user_input(question)

    if not result.allowed:
        return QuestionResponse(
            answer=(
                "⚠️ Sorry, I can’t help with this type of question. "
                "Please ask something related to books or literature."
            )
        )

    try:
        answer = state.agent.ask(question)
        return QuestionResponse(answer=answer)

    except Exception:
        logger.exception("Error processing question.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# --------------------------------------------------
# UI
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Book Recommendation RAG</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 800px;
            margin: 60px auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        h1 {
            margin-top: 0;
            color: #333;
        }
        p {
            color: #555;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            font-size: 14px;
            border-radius: 6px;
            border: 1px solid #ccc;
            resize: vertical;
        }
        button {
            margin-top: 15px;
            padding: 10px 18px;
            font-size: 15px;
            background-color: #4f46e5;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        button:hover {
            background-color: #4338ca;
        }
        .answer {
            margin-top: 25px;
            padding: 15px;
            background: #f9fafb;
            border-left: 4px solid #4f46e5;
            white-space: pre-wrap;
        }
        .footer {
            margin-top: 20px;
            font-size: 12px;
            color: #888;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Book Recommendation RAG</h1>
        <p>Ask questions about books using semantic search.</p>

        <textarea id="question" placeholder="E.g. Suggest books about horror..."></textarea>
        <br/>
        <button onclick="ask()">Ask</button>

        <div id="answer" class="answer" style="display:none;"></div>

        <div class="footer">
            Powered by FastAPI + LangChain
        </div>
    </div>

    <script>
        async function ask() {
            const question = document.getElementById("question").value;
            const answerDiv = document.getElementById("answer");

            if (!question.trim()) {
                alert("Please enter a question.");
                return;
            }

            answerDiv.style.display = "block";
            answerDiv.innerText = "Thinking...";

            try {
                const response = await fetch("/ask", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                answerDiv.innerText = data.answer;
            } catch (error) {
                answerDiv.innerText = "Error fetching answer.";
            }
        }
    </script>
</body>
</html>
"""
