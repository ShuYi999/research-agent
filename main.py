from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import ResearchAgent

app = FastAPI(
    title="Research Agent",
    description="An autonomous research agent powered by Groq + web search",
    version="1.0.0",
)

agent = ResearchAgent()


class ResearchRequest(BaseModel):
    question: str


class ResearchResponse(BaseModel):
    answer: str
    steps: list
    iterations: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = agent.run(request.question)
    return ResearchResponse(**result)


@app.get("/")
def root():
    return {"message": "Research Agent is running. Visit /docs for the API."}
