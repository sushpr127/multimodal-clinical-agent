"""
vectorstore/indexer.py

Indexes chunks into Weaviate with deduplication — uploading the same
document twice is safe and idempotent.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from vectorstore.schema import get_client, TEXT_CLASS, TABLE_CLASS, CHART_CLASS


def document_exists(client, source_file: str) -> bool:
    """Check if a document is already indexed."""
    from weaviate.classes.query import Filter
    col = client.collections.get(TEXT_CLASS)
    results = col.query.fetch_objects(
        filters=Filter.by_property("sourceFile").equal(source_file),
        limit=1,
    )
    return len(results.objects) > 0


def index_document(chunks: dict, source_file: str, force_reindex: bool = False) -> dict:
    """
    Index all chunks for one document into Weaviate.

    Args:
        chunks:        output of chunker.chunk_document()
        source_file:   filename string e.g. "aspirin_label.pdf"
        force_reindex: if True, delete existing chunks and reindex.
                       if False and document exists, skip indexing.

    Returns:
        dict with counts: {"text": N, "table": N, "chart": N, "skipped": bool}
    """
    client = get_client()
    counts = {"text": 0, "table": 0, "chart": 0, "skipped": False}

    try:
        # ── Deduplication check ───────────────────────────────────────────
        if document_exists(client, source_file):
            if force_reindex:
                logger.info(f"  Document exists — clearing for reindex: {source_file}")
                _clear_document(client, source_file)
            else:
                logger.info(f"  Document already indexed, skipping: {source_file}")
                counts["skipped"] = True
                return counts

        # ── Index TextChunks ──────────────────────────────────────────────
        if chunks["text_chunks"]:
            collection = client.collections.get(TEXT_CLASS)
            with collection.batch.dynamic() as batch:
                for chunk in chunks["text_chunks"]:
                    batch.add_object({
                        "content":       chunk.content,
                        "sourceFile":    chunk.source_file,
                        "pageNumber":    chunk.page_number,
                        "chunkIndex":    chunk.chunk_index,
                        "elementType":   chunk.element_type,
                        "tokenEstimate": chunk.token_estimate,
                    })
            counts["text"] = len(chunks["text_chunks"])
            logger.info(f"  Indexed {counts['text']} TextChunks")

        # ── Index TableChunks ─────────────────────────────────────────────
        if chunks["table_chunks"]:
            collection = client.collections.get(TABLE_CLASS)
            with collection.batch.dynamic() as batch:
                for chunk in chunks["table_chunks"]:
                    batch.add_object({
                        "content":    chunk.content,
                        "asHtml":     chunk.as_html,
                        "sourceFile": chunk.source_file,
                        "pageNumber": chunk.page_number,
                        "chunkIndex": chunk.chunk_index,
                        "rowCount":   chunk.row_count,
                        "colCount":   chunk.col_count,
                    })
            counts["table"] = len(chunks["table_chunks"])
            logger.info(f"  Indexed {counts['table']} TableChunks")

        # ── Index ChartChunks ─────────────────────────────────────────────
        if chunks["chart_chunks"]:
            collection = client.collections.get(CHART_CLASS)
            with collection.batch.dynamic() as batch:
                for chunk in chunks["chart_chunks"]:
                    batch.add_object({
                        "content":    chunk.content,
                        "sourceFile": chunk.source_file,
                        "pageNumber": chunk.page_number,
                        "chunkIndex": chunk.chunk_index,
                        "figureType": chunk.figure_type,
                        "title":      chunk.title,
                        "keyValues":  chunk.key_values,
                        "trends":     chunk.trends,
                        "anomalies":  chunk.anomalies,
                    })
            counts["chart"] = len(chunks["chart_chunks"])
            logger.info(f"  Indexed {counts['chart']} ChartChunks")

        # ── Cross-references ──────────────────────────────────────────────
        _add_cross_references(client, source_file)

    finally:
        client.close()

    total = counts["text"] + counts["table"] + counts["chart"]
    logger.info(f"  Indexing complete: {total} total objects")
    return counts


def _clear_document(client, source_file: str):
    """Delete all chunks for a document."""
    from weaviate.classes.query import Filter
    for class_name in [TEXT_CLASS, TABLE_CLASS, CHART_CLASS]:
        col = client.collections.get(class_name)
        col.data.delete_many(
            where=Filter.by_property("sourceFile").equal(source_file)
        )


def _add_cross_references(client, source_file: str):
    """Link TextChunks ↔ ChartChunks on the same page."""
    from weaviate.classes.query import Filter

    text_col  = client.collections.get(TEXT_CLASS)
    chart_col = client.collections.get(CHART_CLASS)

    charts = chart_col.query.fetch_objects(
        filters=Filter.by_property("sourceFile").equal(source_file),
        limit=200,
    )
    if not charts.objects:
        return

    xref_count = 0
    for chart in charts.objects:
        chart_page = chart.properties.get("pageNumber", 0)
        nearby_texts = text_col.query.fetch_objects(
            filters=(
                Filter.by_property("sourceFile").equal(source_file) &
                Filter.by_property("pageNumber").greater_or_equal(chart_page - 1) &
                Filter.by_property("pageNumber").less_or_equal(chart_page + 1)
            ),
            limit=10,
        )
        for text in nearby_texts.objects:
            try:
                text_col.data.reference_add(
                    from_uuid=text.uuid,
                    from_property="hasNearbyCharts",
                    to=chart.uuid,
                )
                xref_count += 1
            except Exception:
                pass

    logger.info(f"  Added {xref_count} cross-references")


if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, ".")

    from ingestion.pdf_parser import parse_pdf
    from ingestion.vision_analyzer import VisionAnalyzer
    from ingestion.chunker import chunk_document
    from vectorstore.schema import get_client, create_schema, verify_schema

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/semaglutide_trial.pdf"
    force = "--force" in sys.argv

    print(f"\n=== Indexing: {pdf} (force={force}) ===\n")

    texts, tables, images = parse_pdf(pdf)
    analyzer = VisionAnalyzer()
    chart_analyses = analyzer.analyze_batch(images)
    chunks = chunk_document(texts, tables, chart_analyses)

    client = get_client()
    create_schema(client, force_recreate=False)
    client.close()

    source = pdf.split("/")[-1]
    counts = index_document(chunks, source, force_reindex=force)

    if counts.get("skipped"):
        print(f"\nSkipped — already indexed. Use --force to reindex.\n")
    else:
        client = get_client()
        verify_schema(client)
        client.close()
        print(f"\n=== Done: {counts} ===\n")