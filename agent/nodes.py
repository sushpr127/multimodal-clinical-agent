"""
agent/nodes.py

Nodes for the LangGraph clinical document QA agent.

Changes from previous version:
  - Added rerank_chunks node between chart_retrieve and synthesize
  - Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free, no API key)
  - Reranker scores all retrieved chunks against the query and keeps top 4
  - Directly improves faithfulness and context precision
  - Added singleton Weaviate client
  - Classifier tiebreaker: explanation verbs beat numerical on ties
  - Chart retriever uses cross-references to pull nearby text
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from agent.state import RetrievedChunk, Citation
from vectorstore.schema import TEXT_CLASS, TABLE_CLASS, CHART_CLASS


# ── Weaviate singleton ────────────────────────────────────────────────────────

_weaviate_client = None

def _get_or_create_client():
    global _weaviate_client
    if _weaviate_client is None or not _weaviate_client.is_connected():
        from vectorstore.schema import get_client
        _weaviate_client = get_client()
    return _weaviate_client

def _close_client():
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
        except Exception:
            pass
        _weaviate_client = None


# ── Cross-encoder re-ranker singleton ────────────────────────────────────────
# Loaded once, reused across all queries. ~80MB, runs on CPU in ~50ms.

_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder re-ranker (first load only)...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Re-ranker loaded.")
    return _reranker


# ── Node 1: Query Classifier ──────────────────────────────────────────────────

EXPLANATION_VERBS = [
    "why", "explain", "describe", "what is", "what are",
    "how does", "tell me", "summarize", "define", "background"
]

def classify_query(state: dict) -> dict:
    query = state.get("query", "").lower().strip()

    visual_keywords    = ["chart","figure","diagram","graph","plot","image",
                          "shows","depicted","visuali","illustrat","curve","axis"]
    numerical_keywords = ["how many","what percentage","rate","number","count",
                          "percent","%","ratio","statistic","p-value","odds",
                          "mean","average","median","dose","concentration","ic50",
                          "auc","efficacy","toxicity level","probability"]
    narrative_keywords = ["what is","explain","describe","why","how does","define",
                          "what are","tell me about","summarize","background","mechanism"]

    visual_score    = sum(1 for k in visual_keywords    if k in query)
    numerical_score = sum(1 for k in numerical_keywords if k in query)
    narrative_score = sum(1 for k in narrative_keywords if k in query)

    scores = {
        "visual":    visual_score,
        "numerical": numerical_score,
        "narrative": narrative_score,
    }
    max_score = max(scores.values())

    if max_score == 0:
        query_type = _llm_classify(state.get("query", ""))
    elif sum(1 for v in scores.values() if v == max_score) > 1:
        starts_with_explanation = any(
            query.startswith(v) for v in EXPLANATION_VERBS
        )
        if starts_with_explanation and scores["narrative"] == max_score:
            query_type = "narrative"
        elif scores["visual"] == max_score:
            query_type = "visual"
        else:
            query_type = "mixed"
    else:
        query_type = max(scores, key=scores.get)

    logger.info(f"  Classified as: {query_type} (scores: {scores})")
    return {
        "query_type":      query_type,
        "reasoning_trace": f"Query type: {query_type}.\n",
    }


def _llm_classify(query: str) -> str:
    from google import genai
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = f"""Classify into exactly one: narrative, numerical, visual, mixed.
Question: {query}
Reply with one word only."""
    r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    result = r.text.strip().lower()
    return result if result in ("narrative","numerical","visual","mixed") else "mixed"


# ── Node 2: Text Retriever ────────────────────────────────────────────────────

def text_retriever(state: dict) -> dict:
    if state.get("query_type") == "visual":
        logger.info("  TextRetriever: skipped")
        return {
            "reasoning_trace": state.get("reasoning_trace","") + "Text retriever: skipped.\n"
        }

    client = _get_or_create_client()
    new_chunks = []
    try:
        from weaviate.classes.query import Filter, MetadataQuery
        collection = client.collections.get(TEXT_CLASS)
        filters = None
        if state.get("source_file"):
            filters = Filter.by_property("sourceFile").equal(state["source_file"])

        results = collection.query.near_text(
            query=state.get("query",""),
            limit=5,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )
        for obj in results.objects:
            new_chunks.append(RetrievedChunk(
                content=obj.properties["content"],
                source_file=obj.properties.get("sourceFile",""),
                page_number=obj.properties.get("pageNumber",0),
                chunk_type="text",
                score=obj.metadata.score or 0.0,
            ))
    except Exception as e:
        logger.error(f"  TextRetriever error: {e}")
        _close_client()

    logger.info(f"  TextRetriever: {len(new_chunks)} chunks")
    return {
        "retrieved_chunks": state.get("retrieved_chunks",[]) + new_chunks,
        "reasoning_trace":  state.get("reasoning_trace","") + f"Text retriever: found {len(new_chunks)} chunks.\n",
    }


# ── Node 3: Table Retriever ───────────────────────────────────────────────────

def table_retriever(state: dict) -> dict:
    if state.get("query_type") in ("visual","narrative"):
        logger.info("  TableRetriever: skipped")
        return {
            "reasoning_trace": state.get("reasoning_trace","") + "Table retriever: skipped.\n"
        }

    client = _get_or_create_client()
    new_chunks = []
    try:
        from weaviate.classes.query import Filter, MetadataQuery
        collection = client.collections.get(TABLE_CLASS)
        filters = None
        if state.get("source_file"):
            filters = Filter.by_property("sourceFile").equal(state["source_file"])

        results = collection.query.near_text(
            query=state.get("query",""),
            limit=3,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )
        for obj in results.objects:
            new_chunks.append(RetrievedChunk(
                content=obj.properties["content"],
                source_file=obj.properties.get("sourceFile",""),
                page_number=obj.properties.get("pageNumber",0),
                chunk_type="table",
                score=obj.metadata.score or 0.0,
            ))
    except Exception as e:
        logger.error(f"  TableRetriever error: {e}")
        _close_client()

    logger.info(f"  TableRetriever: {len(new_chunks)} chunks")
    return {
        "retrieved_chunks": state.get("retrieved_chunks",[]) + new_chunks,
        "reasoning_trace":  state.get("reasoning_trace","") + f"Table retriever: found {len(new_chunks)} chunks.\n",
    }


# ── Node 4: Chart Retriever ───────────────────────────────────────────────────

def chart_retriever(state: dict) -> dict:
    if state.get("query_type") in ("narrative","numerical"):
        logger.info("  ChartRetriever: skipped")
        return {
            "reasoning_trace": state.get("reasoning_trace","") + "Chart retriever: skipped.\n"
        }

    client = _get_or_create_client()
    new_chunks = []

    try:
        from weaviate.classes.query import Filter, MetadataQuery

        chart_col = client.collections.get(CHART_CLASS)
        filters = None
        if state.get("source_file"):
            filters = Filter.by_property("sourceFile").equal(state["source_file"])

        chart_results = chart_col.query.near_text(
            query=state.get("query",""),
            limit=3,
            filters=filters,
            return_metadata=MetadataQuery(score=True),
        )

        chart_pages_seen = set()
        for obj in chart_results.objects:
            new_chunks.append(RetrievedChunk(
                content=obj.properties["content"],
                source_file=obj.properties.get("sourceFile",""),
                page_number=obj.properties.get("pageNumber",0),
                chunk_type="chart",
                score=obj.metadata.score or 0.0,
                figure_type=obj.properties.get("figureType",""),
                title=obj.properties.get("title",""),
            ))
            chart_pages_seen.add(obj.properties.get("pageNumber", 0))

        # Pull cross-referenced nearby text chunks
        if chart_pages_seen and state.get("query_type") in ("visual","mixed"):
            text_col = client.collections.get(TEXT_CLASS)
            source   = state.get("source_file","")
            for page in chart_pages_seen:
                nearby = text_col.query.fetch_objects(
                    filters=(
                        Filter.by_property("sourceFile").equal(source) &
                        Filter.by_property("pageNumber").greater_or_equal(page - 1) &
                        Filter.by_property("pageNumber").less_or_equal(page + 1)
                    ),
                    limit=2,
                )
                for obj in nearby.objects:
                    already = any(
                        c.content == obj.properties["content"]
                        for c in state.get("retrieved_chunks",[])
                    )
                    if not already:
                        new_chunks.append(RetrievedChunk(
                            content=obj.properties["content"],
                            source_file=obj.properties.get("sourceFile",""),
                            page_number=obj.properties.get("pageNumber",0),
                            chunk_type="text",
                            score=0.7,
                        ))

    except Exception as e:
        logger.error(f"  ChartRetriever error: {e}")
        _close_client()

    chart_count = sum(1 for c in new_chunks if c.chunk_type == "chart")
    xref_count  = sum(1 for c in new_chunks if c.chunk_type == "text")
    logger.info(f"  ChartRetriever: {chart_count} charts + {xref_count} cross-ref text")

    return {
        "retrieved_chunks": state.get("retrieved_chunks",[]) + new_chunks,
        "reasoning_trace":  state.get("reasoning_trace","") + (
            f"Chart retriever: found {chart_count} charts "
            f"+ {xref_count} cross-referenced text chunks.\n"
        ),
    }


# ── Node 5: Re-ranker ─────────────────────────────────────────────────────────

def rerank_chunks(state: dict) -> dict:
    """
    Cross-encoder re-ranking node.

    Takes all retrieved chunks, scores each against the query using a
    cross-encoder model, keeps the top 4 by relevance score.

    Why this matters:
    - Vector similarity (used in retrieval) measures semantic closeness
      but doesn't understand query-document relevance precisely.
    - A cross-encoder reads the query AND document together, giving a
      much more accurate relevance score.
    - Cutting from 8 chunks to 4 focused chunks directly improves
      faithfulness (less noise for the synthesizer) and context
      precision (fewer irrelevant chunks in the final context).

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - 80MB, runs on CPU, ~50ms per batch of 8 chunks
    - Trained on MS MARCO passage ranking — generalises well to
      clinical Q&A without fine-tuning
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "reasoning_trace": state.get("reasoning_trace","") + "Re-ranker: no chunks to rank.\n"
        }

    query = state.get("query","")

    try:
        reranker = _get_reranker()

        # Score each chunk against the query
        pairs = [(query, chunk.content[:512]) for chunk in chunks]
        scores = reranker.predict(pairs)

        # Attach scores and sort descending
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        # Keep top 4 — enough context without noise
        top_chunks = [chunk for _, chunk in scored[:4]]

        # Update scores to reflect re-ranker scores for downstream use
        for i, (score, chunk) in enumerate(scored[:4]):
            top_chunks[i].score = float(score)

        dropped = len(chunks) - len(top_chunks)
        logger.info(
            f"  Re-ranker: {len(chunks)} → {len(top_chunks)} chunks "
            f"(dropped {dropped} low-relevance chunks)"
        )

        return {
            "retrieved_chunks": top_chunks,
            "reasoning_trace":  state.get("reasoning_trace","") + (
                f"Re-ranker: scored {len(chunks)} chunks, "
                f"kept top {len(top_chunks)} by relevance.\n"
            ),
        }

    except Exception as e:
        logger.error(f"  Re-ranker error: {e}")
        # Fallback: keep top 4 by original vector score
        fallback = sorted(chunks, key=lambda c: c.score, reverse=True)[:4]
        return {
            "retrieved_chunks": fallback,
            "reasoning_trace":  state.get("reasoning_trace","") + "Re-ranker: fallback to vector scores.\n",
        }


# ── Node 6: Synthesizer ───────────────────────────────────────────────────────

SYNTHESIZER_PROMPT = """You are a clinical document AI assistant. Answer the question using ONLY the provided context.

Rules:
1. Base your answer strictly on the context — never add outside knowledge.
2. Cite every claim with (source: filename, page X).
3. Reference chart/figure data explicitly when present.
4. If context is insufficient, say so clearly.
5. Be precise and clinical in tone.

Question: {query}

Context:
{context}

Provide a clear cited answer, then a "Sources used:" section."""


def synthesize(state: dict) -> dict:
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "answer":          "No relevant information found for this query.",
            "reasoning_trace": state.get("reasoning_trace","") + "Synthesizer: no chunks.\n",
            "citations":       [],
        }

    # Chunks are already top-4 from re-ranker — use all of them
    context_parts = []
    for i, chunk in enumerate(chunks):
        label = chunk.chunk_type.upper()
        if chunk.chunk_type == "chart" and chunk.title:
            label = f"CHART ({chunk.title})"
        context_parts.append(
            f"[{i+1}] [{label}] (source: {chunk.source_file}, page {chunk.page_number})\n"
            f"{chunk.content}"
        )

    from google import genai
    gclient = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    response = gclient.models.generate_content(
        model="gemini-2.0-flash",
        contents=SYNTHESIZER_PROMPT.format(
            query=state.get("query",""),
            context="\n\n".join(context_parts),
        ),
        config={"temperature": 0.2, "max_output_tokens": 1500},
    )

    answer = response.text.strip()
    citations = [
        Citation(
            source_file=c.source_file,
            page_number=c.page_number,
            chunk_type=c.chunk_type,
            excerpt=c.content[:120] + "...",
        )
        for c in chunks
    ]

    n_text  = sum(1 for c in chunks if c.chunk_type=="text")
    n_table = sum(1 for c in chunks if c.chunk_type=="table")
    n_chart = sum(1 for c in chunks if c.chunk_type=="chart")

    _close_client()

    logger.info(f"  Synthesizer: {len(answer)} chars")
    return {
        "answer":          answer,
        "citations":       citations,
        "reasoning_trace": state.get("reasoning_trace","") + (
            f"Synthesizer: used {len(chunks)} chunks "
            f"({n_text} text, {n_table} table, {n_chart} chart).\n"
        ),
    }