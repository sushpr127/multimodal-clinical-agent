"""
ingestion/chunker.py

Converts raw parsed elements into final chunks ready for Weaviate indexing.

Three separate strategies — one per modality:
  - Text:   semantic sliding window (~400 tokens, 50-token overlap)
  - Tables: one chunk per table, never split mid-row
  - Charts: one chunk per ChartAnalysis (full description = the chunk)

Each chunk is a plain dict matching the Weaviate class properties
defined in vectorstore/schema.py.
"""

import re
import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Target sizes ─────────────────────────────────────────────────────────────
# Gemini embedding model handles up to 2048 tokens.
# 400 tokens ≈ 300 words ≈ 1-2 paragraphs — good retrieval granularity.
TEXT_CHUNK_TARGET_TOKENS = 400
TEXT_CHUNK_OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4          # rough estimate: 1 token ≈ 4 chars in English


# ── Output types ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """One chunk of narrative text, ready for Weaviate TextChunk class."""
    content: str              # the actual text — this gets vectorized
    source_file: str
    page_number: int
    chunk_index: int          # position within the document
    element_type: str         # "NarrativeText" | "Title" | "ListItem"
    token_estimate: int


@dataclass
class TableChunk:
    """One complete table, ready for Weaviate TableChunk class."""
    content: str              # plain text representation — gets vectorized
    as_html: str              # HTML for rich display in frontend
    source_file: str
    page_number: int
    chunk_index: int
    row_count: int
    col_count: int


@dataclass
class ChartChunk:
    """One chart/figure analysis, ready for Weaviate ChartChunk class."""
    content: str              # full natural language description — gets vectorized
    source_file: str
    page_number: int
    chunk_index: int
    figure_type: str
    title: str
    key_values: str           # JSON string list
    trends: str               # JSON string list
    anomalies: str            # JSON string list


# ── Text chunker ─────────────────────────────────────────────────────────────

def chunk_text_elements(text_elements: list) -> List[TextChunk]:
    """
    Sliding window chunker for narrative text.

    Strategy:
    1. Sort elements by page number to preserve document order.
    2. Concatenate elements until we hit the target token count.
    3. Slide forward by (target - overlap) tokens for the next chunk.
    4. Never split in the middle of a sentence.

    Why not fixed-size character splits?
    Medical text has wildly varying sentence lengths. Splitting mid-sentence
    destroys the clinical context needed for accurate retrieval.
    """
    if not text_elements:
        return []

    # Sort by page then by order of appearance
    sorted_elements = sorted(text_elements, key=lambda e: e.page_number)

    # Build one big list of (text, page_number, element_type) segments
    segments = [
        (el.content, el.page_number, el.element_type)
        for el in sorted_elements
        if el.content.strip()
    ]

    chunks: List[TextChunk] = []
    chunk_index = 0

    # Sliding window over segments
    target_chars = TEXT_CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN
    overlap_chars = TEXT_CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    current_text = ""
    current_page = segments[0][1] if segments else 0
    current_type = segments[0][2] if segments else "NarrativeText"
    buffer = []  # (text, page, type)

    for text, page, etype in segments:
        buffer.append((text, page, etype))
        current_text += " " + text

        if len(current_text) >= target_chars:
            # Find a good sentence boundary to cut at
            cut_point = _find_sentence_boundary(current_text, target_chars)
            chunk_content = current_text[:cut_point].strip()

            if chunk_content:
                chunks.append(TextChunk(
                    content=chunk_content,
                    source_file=sorted_elements[0].source_file,
                    page_number=current_page,
                    chunk_index=chunk_index,
                    element_type=current_type,
                    token_estimate=len(chunk_content) // CHARS_PER_TOKEN,
                ))
                chunk_index += 1

            # Keep overlap — roll back by overlap_chars from cut_point
            overlap_start = max(0, cut_point - overlap_chars)
            current_text = current_text[overlap_start:].strip()
            # Update page to the page of the first kept segment
            current_page = page
            current_type = etype

    # Flush remaining text
    if current_text.strip():
        chunks.append(TextChunk(
            content=current_text.strip(),
            source_file=sorted_elements[0].source_file,
            page_number=current_page,
            chunk_index=chunk_index,
            element_type=current_type,
            token_estimate=len(current_text) // CHARS_PER_TOKEN,
        ))

    logger.info(f"  Text chunker: {len(text_elements)} elements → {len(chunks)} chunks")
    return chunks


def _find_sentence_boundary(text: str, target: int) -> int:
    """
    Find the nearest sentence-ending punctuation at or before `target` chars.
    Falls back to the nearest word boundary if no sentence end found.
    """
    # Look for sentence end (. ! ?) in a window around the target
    window_start = max(0, target - 100)
    window = text[window_start:target + 100]

    # Find last sentence-ending punctuation in the window before target
    matches = list(re.finditer(r'[.!?]\s', window))
    if matches:
        # Pick the last match that falls before target
        before_target = [m for m in matches if (window_start + m.end()) <= target]
        if before_target:
            return window_start + before_target[-1].end()

    # Fall back: nearest space before target
    space_idx = text.rfind(' ', 0, target)
    return space_idx if space_idx > 0 else target


# ── Table chunker ─────────────────────────────────────────────────────────────

def chunk_table_elements(table_elements: list) -> List[TableChunk]:
    """
    One chunk per table — never split tables.

    Rationale: a clinical data table only makes sense as a whole.
    Splitting a table mid-row destroys the row-column relationships
    that make it useful for numerical question answering.

    The content field is the plain text summary (for vectorization).
    The as_html field preserves the full structure for display.
    """
    chunks = []

    for i, table in enumerate(table_elements):
        # Build a vectorization-friendly text representation
        # Format: brief description + flattened row data
        plain_content = _table_to_plain_text(table)

        chunks.append(TableChunk(
            content=plain_content,
            as_html=table.as_html or "",
            source_file=table.source_file,
            page_number=table.page_number,
            chunk_index=i,
            row_count=table.row_count,
            col_count=table.col_count,
        ))

    logger.info(f"  Table chunker: {len(table_elements)} tables → {len(chunks)} chunks (1:1)")
    return chunks


def _table_to_plain_text(table) -> str:
    """
    Convert a TableElement to a clean plain text string for embedding.

    We prefix with a description so the vector captures the context,
    not just raw cell values.
    """
    lines = []
    lines.append(
        f"Clinical data table from page {table.page_number} "
        f"of {table.source_file} "
        f"({table.row_count} rows, {table.col_count} columns)."
    )

    if table.as_html:
        # Strip HTML tags to get raw text, preserve whitespace structure
        text = re.sub(r'<tr[^>]*>', '\nROW: ', table.as_html)
        text = re.sub(r'<td[^>]*>', ' | ', text)
        text = re.sub(r'<th[^>]*>', ' | ', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        lines.append(text)
    elif table.content:
        lines.append(table.content)

    return "\n".join(lines)


# ── Chart chunker ─────────────────────────────────────────────────────────────

def chunk_chart_analyses(chart_analyses: list) -> List[ChartChunk]:
    """
    One chunk per chart analysis — the full Gemini Vision description is the chunk.

    The description is already a rich natural language summary.
    We augment it with key values and trends for better retrieval.

    Rationale: chart descriptions are short (<200 words) and self-contained.
    There's no benefit to splitting them further.
    """
    import json
    chunks = []

    data_figures = [a for a in chart_analyses if a.is_data_figure]

    for i, analysis in enumerate(data_figures):
        # Build a rich content string that captures all extracted metadata
        # This is what gets embedded — make it as informative as possible
        content_parts = [analysis.description]

        if analysis.title:
            content_parts.insert(0, f"Figure title: {analysis.title}.")

        if analysis.key_values:
            content_parts.append(
                f"Key values: {', '.join(str(v) for v in analysis.key_values)}."
            )

        if analysis.trends:
            content_parts.append(
                f"Trends observed: {' '.join(analysis.trends)}"
            )

        if analysis.anomalies:
            content_parts.append(
                f"Notable findings: {' '.join(analysis.anomalies)}"
            )

        content_parts.append(
            f"Source: {analysis.source_file}, page {analysis.page_number}."
        )

        full_content = " ".join(content_parts)

        chunks.append(ChartChunk(
            content=full_content,
            source_file=analysis.source_file,
            page_number=analysis.page_number,
            chunk_index=i,
            figure_type=analysis.figure_type,
            title=analysis.title or "",
            key_values=json.dumps(analysis.key_values),
            trends=json.dumps(analysis.trends),
            anomalies=json.dumps(analysis.anomalies),
        ))

    logger.info(f"  Chart chunker: {len(data_figures)} figures → {len(chunks)} chunks (1:1)")
    return chunks


# ── Master pipeline function ──────────────────────────────────────────────────

def chunk_document(text_elements, table_elements, chart_analyses) -> dict:
    """
    Run all three chunkers and return a dict with all chunks.

    This is the main entry point called by the indexing pipeline.

    Returns:
        {
            "text_chunks":  List[TextChunk],
            "table_chunks": List[TableChunk],
            "chart_chunks": List[ChartChunk],
        }
    """
    text_chunks  = chunk_text_elements(text_elements)
    table_chunks = chunk_table_elements(table_elements)
    chart_chunks = chunk_chart_analyses(chart_analyses)

    total = len(text_chunks) + len(table_chunks) + len(chart_chunks)
    logger.info(
        f"  Chunking complete: "
        f"{len(text_chunks)} text + "
        f"{len(table_chunks)} table + "
        f"{len(chart_chunks)} chart = "
        f"{total} total chunks"
    )

    return {
        "text_chunks":  text_chunks,
        "table_chunks": table_chunks,
        "chart_chunks": chart_chunks,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, ".")

    from ingestion.pdf_parser import parse_pdf
    from ingestion.vision_analyzer import VisionAnalyzer

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/semaglutide_trial.pdf"
    print(f"\n=== Chunker test on {pdf} ===\n")

    # Step 1: Parse
    texts, tables, images = parse_pdf(pdf)

    # Step 2: Vision analysis
    analyzer = VisionAnalyzer()
    chart_analyses = analyzer.analyze_batch(images)

    # Step 3: Chunk
    result = chunk_document(texts, tables, chart_analyses)

    # Report
    print(f"\n=== Chunk summary ===")
    print(f"  Text chunks : {len(result['text_chunks'])}")
    print(f"  Table chunks: {len(result['table_chunks'])}")
    print(f"  Chart chunks: {len(result['chart_chunks'])}")

    if result["text_chunks"]:
        c = result["text_chunks"][0]
        print(f"\nSample text chunk (page {c.page_number}, ~{c.token_estimate} tokens):")
        print(f"  {c.content[:200]}...")

    if result["chart_chunks"]:
        c = result["chart_chunks"][0]
        print(f"\nSample chart chunk (page {c.page_number}, {c.figure_type}):")
        print(f"  {c.content[:200]}...")