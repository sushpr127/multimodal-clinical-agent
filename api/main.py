"""
api/main.py — FastAPI backend with deduplication support
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Clinical Document Intelligence Agent",
    description="Multimodal RAG over clinical research documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    source_file: str = ""

class CitationResponse(BaseModel):
    source_file: str
    page_number: int
    chunk_type: str
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    query_type: str
    reasoning_trace: str
    citations: list[CitationResponse]

class IngestResponse(BaseModel):
    filename: str
    status: str
    text_chunks: int
    table_chunks: int
    chart_chunks: int
    total_chunks: int
    message: str

class DocumentInfo(BaseModel):
    filename: str
    text_count: int
    table_count: int
    chart_count: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "clinical-doc-agent"}


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    from vectorstore.schema import get_client, TEXT_CLASS, TABLE_CLASS, CHART_CLASS
    client = get_client()
    docs = {}
    try:
        for class_name, field in [
            (TEXT_CLASS,  "text_count"),
            (TABLE_CLASS, "table_count"),
            (CHART_CLASS, "chart_count"),
        ]:
            col = client.collections.get(class_name)
            results = col.query.fetch_objects(limit=1000)
            for obj in results.objects:
                fname = obj.properties.get("sourceFile", "unknown")
                if fname not in docs:
                    docs[fname] = {"filename": fname, "text_count": 0,
                                   "table_count": 0, "chart_count": 0}
                docs[fname][field] += 1
    finally:
        client.close()
    return list(docs.values())


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    force_reindex: bool = Query(default=False, description="Re-index even if document already exists")
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Ingesting {file.filename} ({tmp_path.stat().st_size/1024/1024:.1f} MB)")

        from ingestion.pdf_parser import parse_pdf
        from ingestion.vision_analyzer import VisionAnalyzer
        from ingestion.chunker import chunk_document
        from vectorstore.schema import get_client, create_schema
        from vectorstore.indexer import index_document

        texts, tables, images = parse_pdf(str(tmp_path))
        analyzer = VisionAnalyzer()
        chart_analyses = analyzer.analyze_batch(images)
        chunks = chunk_document(texts, tables, chart_analyses)

        client = get_client()
        create_schema(client, force_recreate=False)
        client.close()

        counts = index_document(chunks, file.filename, force_reindex=force_reindex)

        if counts.get("skipped"):
            return IngestResponse(
                filename=file.filename,
                status="skipped",
                text_chunks=0, table_chunks=0, chart_chunks=0, total_chunks=0,
                message=f"{file.filename} already indexed. Use ?force_reindex=true to reindex.",
            )

        total = counts["text"] + counts["table"] + counts["chart"]
        return IngestResponse(
            filename=file.filename,
            status="success",
            text_chunks=counts.get("text", 0),
            table_chunks=counts.get("table", 0),
            chart_chunks=counts.get("chart", 0),
            total_chunks=total,
            message=f"Successfully indexed {total} chunks from {file.filename}",
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        from agent.graph import ask
        result = ask(query=request.query, source_file=request.source_file)
        return QueryResponse(
            answer=result["answer"],
            query_type=result["query_type"],
            reasoning_trace=result["reasoning_trace"],
            citations=[
                CitationResponse(
                    source_file=c["source_file"],
                    page_number=c["page_number"],
                    chunk_type=c["chunk_type"],
                    excerpt=c["excerpt"],
                )
                for c in result["citations"]
            ],
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)