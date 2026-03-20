"""
agent/state.py

AgentState as TypedDict — required for LangGraph compatibility.
LangGraph passes state between nodes as a plain dict with these keys.
"""

from typing import Optional
from dataclasses import dataclass
from typing_extensions import TypedDict


@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    page_number: int
    chunk_type: str        # "text" | "table" | "chart"
    score: float
    figure_type: str = ""
    title: str = ""


@dataclass
class Citation:
    source_file: str
    page_number: int
    chunk_type: str
    excerpt: str


class AgentState(TypedDict, total=False):
    """
    TypedDict state — LangGraph reads and writes these keys between nodes.
    total=False means all keys are optional (have defaults).
    """
    query:            str
    query_type:       str
    retrieved_chunks: list
    citations:        list
    answer:           str
    reasoning_trace:  str
    error:            str
    source_file:      str