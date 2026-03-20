"""
ingestion/vision_analyzer.py

Takes an ImageElement and sends it to Gemini 2.0 Flash Vision.
Returns a structured ChartAnalysis with chart type, key values,
trends, and anomalies — ready to store in Weaviate.

Uses google-genai SDK (replaces deprecated google-generativeai).
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

logger = logging.getLogger(__name__)


# ── Output schema ────────────────────────────────────────────────────────────

@dataclass
class ChartAnalysis:
    source_file: str
    page_number: int
    element_id: str
    figure_type: str
    description: str
    title: Optional[str]
    x_axis: Optional[str]
    y_axis: Optional[str]
    key_values: list
    trends: list
    anomalies: list
    raw_response: str = ""
    is_data_figure: bool = True


# ── Gemini Vision prompt ──────────────────────────────────────────────────────

VISION_PROMPT = """You are a medical/clinical document AI analyzing a figure extracted from a clinical research paper or FDA document.

Analyze this image carefully and respond with ONLY a valid JSON object — no markdown, no explanation, just the JSON.

JSON schema:
{
  "figure_type": "<one of: bar_chart, line_chart, kaplan_meier, scatter_plot, table_image, forest_plot, pie_chart, diagram, photograph, logo_or_icon, other>",
  "is_data_figure": <true if this contains scientific/clinical data, false if logo/icon/decorative>,
  "title": "<figure title if visible, or null>",
  "description": "<2-4 sentence natural language description of what this figure shows>",
  "x_axis": "<x-axis label if applicable, or null>",
  "y_axis": "<y-axis label if applicable, or null>",
  "key_values": ["<important numbers, percentages, p-values, or statistics visible>"],
  "trends": ["<describe trends, e.g. 'treatment group shows 40% reduction vs control'>"],
  "anomalies": ["<outliers, unexpected findings, warnings, or notable deviations>"]
}

Be precise. Use null for missing strings, [] for missing arrays. Never fabricate values."""


# ── Analyzer ──────────────────────────────────────────────────────────────────

class VisionAnalyzer:

    def __init__(self):
        from google import genai
        from google.genai import types
        from dotenv import load_dotenv
        load_dotenv()

        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.types = types
        logger.info("VisionAnalyzer initialized with gemini-2.0-flash (google-genai SDK)")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, image_element) -> ChartAnalysis:
        import PIL.Image
        import io

        pil_image = PIL.Image.open(io.BytesIO(image_element.image_bytes))

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[VISION_PROMPT, pil_image],
            config=self.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            )
        )

        raw_text = response.text.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            parts = raw_text.split("```")
            raw_text = parts[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed page {image_element.page_number}: {e}")
            return ChartAnalysis(
                source_file=image_element.source_file,
                page_number=image_element.page_number,
                element_id=image_element.element_id,
                figure_type="other",
                description="Figure could not be analyzed automatically.",
                title=None, x_axis=None, y_axis=None,
                key_values=[], trends=[], anomalies=[],
                raw_response=raw_text,
                is_data_figure=False,
            )

        return ChartAnalysis(
            source_file=image_element.source_file,
            page_number=image_element.page_number,
            element_id=image_element.element_id,
            figure_type=data.get("figure_type", "other"),
            description=data.get("description", ""),
            title=data.get("title"),
            x_axis=data.get("x_axis"),
            y_axis=data.get("y_axis"),
            key_values=data.get("key_values") or [],
            trends=data.get("trends") or [],
            anomalies=data.get("anomalies") or [],
            raw_response=raw_text,
            is_data_figure=data.get("is_data_figure", True),
        )

    def analyze_batch(self, image_elements: list) -> list:
        results = []
        data_figures = 0

        for i, img_el in enumerate(image_elements):
            logger.info(
                f"  Analyzing image {i+1}/{len(image_elements)} "
                f"(page {img_el.page_number}, {img_el.width}x{img_el.height}px)"
            )
            try:
                analysis = self.analyze(img_el)
                results.append(analysis)
                if analysis.is_data_figure:
                    data_figures += 1
                    logger.info(f"    → {analysis.figure_type}: {analysis.description[:80]}...")
                else:
                    logger.info(f"    → Skipped ({analysis.figure_type})")
            except Exception as e:
                logger.error(f"  Failed page {img_el.page_number}: {e}")

        logger.info(f"  Vision complete: {data_figures}/{len(image_elements)} are data figures")
        return results


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ingestion.pdf_parser import parse_pdf

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/semaglutide_trial.pdf"
    print(f"\n=== Vision analysis test on {pdf} ===\n")

    _, _, images = parse_pdf(pdf)
    if not images:
        print("No images found.")
        sys.exit(0)

    print(f"Found {len(images)} images. Analyzing...\n")
    analyzer = VisionAnalyzer()
    analyses = analyzer.analyze_batch(images)

    data_figs = [a for a in analyses if a.is_data_figure]
    print(f"\n=== {len(data_figs)} data figures found ===\n")

    for i, a in enumerate(data_figs):
        print(f"Figure {i+1} — Page {a.page_number}")
        print(f"  Type       : {a.figure_type}")
        print(f"  Title      : {a.title}")
        print(f"  Description: {a.description[:120]}...")
        if a.key_values:
            print(f"  Key values : {a.key_values[:3]}")
        if a.trends:
            print(f"  Trends     : {a.trends[:2]}")
        print()