from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from .indexing import build_index

# Build index on startup (for 100–200 docs this is usually fine)
search_engine = build_index()

app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description=(
        "Semantic search over local text files with caching, "
        "vector search, and ranking explanations."
    ),
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = search_engine.search(query=request.query, top_k=request.top_k)
    return {"results": results}
