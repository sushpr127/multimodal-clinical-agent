"""
agent/graph.py

LangGraph state machine for the clinical document QA agent.

Flow:
  START
    → classify_query
    → text_retriever
    → table_retriever
    → chart_retriever
    → rerank_chunks      ← NEW: cross-encoder re-ranking
    → synthesize
  END

The re-ranker sits between all retrievers and the synthesizer.
It scores every retrieved chunk against the query and keeps top 4,
reducing noise and improving faithfulness.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import (
    classify_query,
    text_retriever,
    table_retriever,
    chart_retriever,
    rerank_chunks,
    synthesize,
)


def build_graph():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "clinical-doc-agent")

    graph = StateGraph(AgentState)

    graph.add_node("classify",       classify_query)
    graph.add_node("text_retrieve",  text_retriever)
    graph.add_node("table_retrieve", table_retriever)
    graph.add_node("chart_retrieve", chart_retriever)
    graph.add_node("rerank",         rerank_chunks)      # ← new
    graph.add_node("synthesize",     synthesize)

    graph.add_edge(START,            "classify")
    graph.add_edge("classify",       "text_retrieve")
    graph.add_edge("text_retrieve",  "table_retrieve")
    graph.add_edge("table_retrieve", "chart_retrieve")
    graph.add_edge("chart_retrieve", "rerank")           # ← new
    graph.add_edge("rerank",         "synthesize")       # ← new
    graph.add_edge("synthesize",     END)

    return graph.compile()


def ask(query: str, source_file: str = "") -> dict:
    graph = build_graph()

    final_state = graph.invoke({
        "query":            query,
        "source_file":      source_file,
        "query_type":       "",
        "retrieved_chunks": [],
        "citations":        [],
        "answer":           "",
        "reasoning_trace":  "",
        "error":            "",
    })

    return {
        "answer":          final_state.get("answer", ""),
        "reasoning_trace": final_state.get("reasoning_trace", ""),
        "query_type":      final_state.get("query_type", ""),
        "citations": [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_type":  c.chunk_type,
                "excerpt":     c.excerpt,
            }
            for c in final_state.get("citations", [])
        ],
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("\n=== LangGraph Agent — Re-ranker Test ===\n")

    test_queries = [
        {
            "query": "What percentage of patients experienced nausea with semaglutide 14mg?",
            "source_file": "Ozempic.pdf",
            "expected_type": "numerical",
        },
        {
            "query": "What does the STAR diagram show about drug candidate classes?",
            "source_file": "semaglutide_trial.pdf",
            "expected_type": "visual",
        },
        {
            "query": "What are the contraindications for Ozempic?",
            "source_file": "Ozempic.pdf",
            "expected_type": "narrative",
        },
    ]

    for test in test_queries:
        print(f"Query       : {test['query']}")
        print(f"Source      : {test['source_file']}")
        print("-" * 60)

        result = ask(query=test["query"], source_file=test["source_file"])

        print(f"Type        : {result['query_type']}")
        print(f"Reasoning   : {result['reasoning_trace'].strip()}")
        print(f"\nAnswer      :\n{result['answer'][:300]}...")
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result["citations"]:
            print(f"  [{c['chunk_type']}] {c['source_file']} p.{c['page_number']}")
        print("\n" + "=" * 60 + "\n")