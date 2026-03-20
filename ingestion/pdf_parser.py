"""
ingestion/pdf_parser.py

Parses a PDF using Unstructured.io and returns three separate lists:
  - text_elements:  narrative paragraphs and titles
  - table_elements: structured tables as dicts
  - image_elements: extracted images as bytes + metadata

Each element carries: source_file, page_number, element_type, content.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import base64
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class TextElement:
    content: str                  # raw paragraph / heading text
    page_number: int
    source_file: str
    element_type: str             # "NarrativeText" | "Title" | "ListItem"
    element_id: str = ""


@dataclass
class TableElement:
    content: str                  # table as plain text (fallback)
    as_html: Optional[str]        # table as HTML string (richer)
    page_number: int
    source_file: str
    element_id: str = ""
    row_count: int = 0
    col_count: int = 0


@dataclass
class ImageElement:
    image_bytes: bytes            # raw PNG/JPEG bytes
    image_b64: str                # base64-encoded — ready for Gemini Vision
    page_number: int
    source_file: str
    element_id: str = ""
    width: int = 0
    height: int = 0


# ── Main parser ──────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> tuple[List[TextElement], List[TableElement], List[ImageElement]]:
    """
    Parse a PDF into three modality buckets.

    Args:
        pdf_path: absolute or relative path to the PDF file

    Returns:
        (text_elements, table_elements, image_elements)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source_file = pdf_path.name
    logger.info(f"Parsing {source_file} ...")

    # ── Run Unstructured ────────────────────────────────────────────────────
    # strategy="hi_res" uses the vision model for layout detection — slower
    # but correctly identifies tables and figures in dense medical PDFs.
    # strategy="fast" is text-only and will miss most charts and tables.
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",               # required for table + image extraction
        infer_table_structure=True,       # extract tables as HTML
        extract_images_in_pdf=True,       # extract embedded figures
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=str(pdf_path.parent / "extracted_images"),
    )

    logger.info(f"  Unstructured returned {len(elements)} raw elements")

    text_elements: List[TextElement] = []
    table_elements: List[TableElement] = []
    image_elements: List[ImageElement] = []

    for el in elements:
        el_type = type(el).__name__
        page_num = el.metadata.page_number or 0
        el_id = el.id if hasattr(el, "id") else ""

        # ── Text ─────────────────────────────────────────────────────────
        if el_type in ("NarrativeText", "Title", "ListItem", "Text"):
            text = el.text.strip()
            if len(text) < 30:          # skip noise — headers, footers, page numbers
                continue
            text_elements.append(TextElement(
                content=text,
                page_number=page_num,
                source_file=source_file,
                element_type=el_type,
                element_id=el_id,
            ))

        # ── Tables ───────────────────────────────────────────────────────
        elif el_type == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            # Estimate row/col count from HTML if available
            row_count = html.count("<tr>") if html else 0
            col_count = html.count("<td>") // max(row_count, 1) if html else 0

            table_elements.append(TableElement(
                content=el.text.strip(),
                as_html=html,
                page_number=page_num,
                source_file=source_file,
                element_id=el_id,
                row_count=row_count,
                col_count=col_count,
            ))

        # ── Images ───────────────────────────────────────────────────────
        elif el_type == "Image":
            # Unstructured saves images to disk; load them back as bytes
            image_path = getattr(el.metadata, "image_path", None)
            if image_path and Path(image_path).exists():
                with open(image_path, "rb") as f:
                    img_bytes = f.read()

                # Get dimensions
                from PIL import Image as PILImage
                try:
                    with PILImage.open(io.BytesIO(img_bytes)) as img:
                        w, h = img.size
                except Exception:
                    w, h = 0, 0

                # Skip tiny images — likely logos, icons, or artifacts
                if w < 100 or h < 100:
                    continue

                image_elements.append(ImageElement(
                    image_bytes=img_bytes,
                    image_b64=base64.b64encode(img_bytes).decode("utf-8"),
                    page_number=page_num,
                    source_file=source_file,
                    element_id=el_id,
                    width=w,
                    height=h,
                ))

    logger.info(
        f"  Extracted → {len(text_elements)} text | "
        f"{len(table_elements)} tables | "
        f"{len(image_elements)} images"
    )

    return text_elements, table_elements, image_elements


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/keytruda.pdf"
    texts, tables, images = parse_pdf(pdf)

    print(f"\n=== Parse results for {pdf} ===")
    print(f"  Text elements : {len(texts)}")
    print(f"  Table elements: {len(tables)}")
    print(f"  Image elements: {len(images)}")

    if texts:
        print(f"\nSample text (page {texts[0].page_number}):")
        print(f"  {texts[0].content[:200]}...")

    if tables:
        t = tables[0]
        print(f"\nSample table (page {t.page_number}, {t.row_count} rows x {t.col_count} cols):")
        print(f"  {t.content[:200]}...")

    if images:
        img = images[0]
        print(f"\nSample image (page {img.page_number}, {img.width}x{img.height}px):")
        print(f"  base64 length: {len(img.image_b64)} chars")