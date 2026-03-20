"""
vectorstore/schema.py

Defines three Weaviate classes:
  - TextChunk       — narrative paragraphs from clinical documents
  - TableChunk      — structured tables extracted from PDFs
  - ChartChunk      — Gemini Vision descriptions of figures/charts

Cross-references:
  - TextChunk  → hasNearbyCharts  → ChartChunk  (same page)
  - ChartChunk → hasNearbyText    → TextChunk   (same page)

This lets a single query retrieve a chart description AND the
surrounding clinical narrative simultaneously — which is what
makes this multimodal RAG, not just text RAG.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Class names ───────────────────────────────────────────────────────────────
TEXT_CLASS  = "TextChunk"
TABLE_CLASS = "TableChunk"
CHART_CLASS = "ChartChunk"


# ── Schema builder ────────────────────────────────────────────────────────────

def get_client():
    """Return a connected Weaviate client."""
    import weaviate
    from weaviate.auth import AuthApiKey

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    assert client.is_ready(), "Weaviate cluster not ready"
    return client


def create_schema(client, force_recreate: bool = False):
    """
    Create all three classes in Weaviate.

    Args:
        client:         connected Weaviate client
        force_recreate: if True, delete and recreate all classes.
                        Use this during development when schema changes.
                        Never use in production.
    """
    import weaviate.classes.config as wc

    existing = {c.name for c in client.collections.list_all().values()}

    if force_recreate:
        for name in [TEXT_CLASS, TABLE_CLASS, CHART_CLASS]:
            if name in existing:
                client.collections.delete(name)
                logger.info(f"  Deleted existing class: {name}")
        existing = set()

    # ── TextChunk ─────────────────────────────────────────────────────────
    if TEXT_CLASS not in existing:
        client.collections.create(
            name=TEXT_CLASS,
            description="Narrative text paragraph from a clinical document",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_weaviate(),
            generative_config=wc.Configure.Generative.cohere(),
            properties=[
                wc.Property(name="content",      data_type=wc.DataType.TEXT,
                            description="The paragraph text — this field gets vectorized"),
                wc.Property(name="sourceFile",   data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="pageNumber",   data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="chunkIndex",   data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="elementType",  data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="tokenEstimate",data_type=wc.DataType.INT,
                            skip_vectorization=True),
            ],
        )
        logger.info(f"  Created class: {TEXT_CLASS}")
    else:
        logger.info(f"  Class already exists: {TEXT_CLASS}")

    # ── TableChunk ────────────────────────────────────────────────────────
    if TABLE_CLASS not in existing:
        client.collections.create(
            name=TABLE_CLASS,
            description="Structured table extracted from a clinical document",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_weaviate(),
            generative_config=wc.Configure.Generative.cohere(),
            properties=[
                wc.Property(name="content",    data_type=wc.DataType.TEXT,
                            description="Plain text table content — gets vectorized"),
                wc.Property(name="asHtml",     data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="sourceFile", data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="pageNumber", data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="chunkIndex", data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="rowCount",   data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="colCount",   data_type=wc.DataType.INT,
                            skip_vectorization=True),
            ],
        )
        logger.info(f"  Created class: {TABLE_CLASS}")
    else:
        logger.info(f"  Class already exists: {TABLE_CLASS}")

    # ── ChartChunk ────────────────────────────────────────────────────────
    if CHART_CLASS not in existing:
        client.collections.create(
            name=CHART_CLASS,
            description="Gemini Vision analysis of a chart or figure from a clinical document",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_weaviate(),
            generative_config=wc.Configure.Generative.cohere(),
            properties=[
                wc.Property(name="content",    data_type=wc.DataType.TEXT,
                            description="Full natural language description — gets vectorized"),
                wc.Property(name="sourceFile", data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="pageNumber", data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="chunkIndex", data_type=wc.DataType.INT,
                            skip_vectorization=True),
                wc.Property(name="figureType", data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="title",      data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="keyValues",  data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="trends",     data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
                wc.Property(name="anomalies",  data_type=wc.DataType.TEXT,
                            skip_vectorization=True),
            ],
        )
        logger.info(f"  Created class: {CHART_CLASS}")
    else:
        logger.info(f"  Class already exists: {CHART_CLASS}")

    logger.info("  Schema setup complete")


def verify_schema(client):
    """Print a summary of what's in each class."""
    for name in [TEXT_CLASS, TABLE_CLASS, CHART_CLASS]:
        try:
            col = client.collections.get(name)
            count = col.aggregate.over_all(total_count=True).total_count
            print(f"  {name}: {count} objects")
        except Exception as e:
            print(f"  {name}: ERROR — {e}")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("\n=== Setting up Weaviate schema ===\n")
    client = get_client()

    # force_recreate=True during development so schema changes apply cleanly
    create_schema(client, force_recreate=True)

    print("\n=== Verifying classes ===")
    verify_schema(client)

    client.close()
    print("\nDone.\n")