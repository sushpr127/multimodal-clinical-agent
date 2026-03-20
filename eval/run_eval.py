"""
eval/run_eval.py

Ragas evaluation harness for the multimodal clinical document agent.
Uses HuggingFace local embeddings for answer_relevancy (no API quota issues).

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --doc ozempic_label.pdf
    python -m eval.run_eval --baseline
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)


def clean_answer(text: str) -> str:
    """Strip Sources section and markdown before Ragas scoring."""
    for marker in ["Sources used:", "Sources:", "\n\n*", "\nSources"]:
        if marker in text:
            text = text[:text.index(marker)].strip()
    return text.replace("**", "").strip()


def run_eval(doc_filter=None, include_baseline=False):

    # ── Load test cases ───────────────────────────────────────────────────
    test_path = Path(__file__).parent / "test_cases.json"
    with open(test_path) as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if c["source_file"] == doc_filter] if doc_filter else all_cases

    print(f"\n{'='*60}")
    print(f"  Clinical Document Agent — Ragas Evaluation")
    print(f"  {len(cases)} test cases | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # ── Build LangGraph directly to capture full chunk content ────────────
    from langgraph.graph import StateGraph, START, END
    from agent.state import AgentState
    from agent.nodes import classify_query, text_retriever, table_retriever, chart_retriever, synthesize

    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    g = StateGraph(AgentState)
    g.add_node("classify",       classify_query)
    g.add_node("text_retrieve",  text_retriever)
    g.add_node("table_retrieve", table_retriever)
    g.add_node("chart_retrieve", chart_retriever)
    g.add_node("synthesize",     synthesize)
    g.add_edge(START,            "classify")
    g.add_edge("classify",       "text_retrieve")
    g.add_edge("text_retrieve",  "table_retrieve")
    g.add_edge("table_retrieve", "chart_retrieve")
    g.add_edge("chart_retrieve", "synthesize")
    g.add_edge("synthesize",     END)
    graph = g.compile()

    # ── Run agent on all cases ────────────────────────────────────────────
    results = []
    print("Running agent on all test cases...\n")

    for i, case in enumerate(cases):
        print(f"  [{i+1:02d}/{len(cases)}] {case['id']}: {case['question'][:55]}...")
        try:
            state = graph.invoke({
                "query":            case["question"],
                "source_file":      case["source_file"],
                "query_type":       "",
                "retrieved_chunks": [],
                "citations":        [],
                "answer":           "",
                "reasoning_trace":  "",
                "error":            "",
            })
            full_contexts = [
                chunk.content for chunk in state.get("retrieved_chunks", [])
            ] or ["No context retrieved."]

            results.append({
                "case":       case,
                "answer":     state.get("answer", ""),
                "query_type": state.get("query_type", ""),
                "contexts":   full_contexts,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "case":       case,
                "answer":     "",
                "query_type": "",
                "contexts":   ["No context retrieved."],
                "error":      str(e),
            })

    # ── Build Ragas dataset ───────────────────────────────────────────────
    from datasets import Dataset

    valid = [r for r in results if r["answer"] and not r.get("error")]

    dataset = Dataset.from_dict({
        "question":     [r["case"]["question"]      for r in valid],
        "answer":       [clean_answer(r["answer"])  for r in valid],
        "contexts":     [r["contexts"]              for r in valid],
        "ground_truth": [r["case"]["ground_truth"]  for r in valid],
    })

    # ── Configure Ragas ───────────────────────────────────────────────────
    print("\nScoring with Ragas...\n")

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    # Local embeddings — no API quota, works offline
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )

    answer_relevancy.embeddings = embeddings

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    scores_df = scores.to_pandas()

    # ── Per-question results ──────────────────────────────────────────────
    print(f"\n{'ID':<6} {'Type':<10} {'Modal':<7} {'Faith':>6} {'Relev':>6} {'Prec':>6}  Status")
    print("-" * 60)

    pass_count = 0
    for i, r in enumerate(valid):
        if i >= len(scores_df):
            break
        row   = scores_df.iloc[i]
        faith = float(row.get("faithfulness",      0) or 0)
        relev = float(row.get("answer_relevancy",  0) or 0)
        prec  = float(row.get("context_precision", 0) or 0)
        ok    = faith >= 0.7
        if ok:
            pass_count += 1
        print(
            f"  {r['case']['id']:<4}  "
            f"{r['case']['expected_type']:<10}"
            f"{r['case']['modality']:<7}"
            f"{faith:>6.2f}"
            f"{relev:>6.2f}"
            f"{prec:>6.2f}  "
            f"{'PASS' if ok else 'REVIEW'}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    mean_faith = scores_df["faithfulness"].fillna(0).mean()
    mean_relev = scores_df["answer_relevancy"].fillna(0).mean()
    mean_prec  = scores_df["context_precision"].fillna(0).mean()

    low_faith = [
        valid[i]["case"]["id"]
        for i in range(min(len(valid), len(scores_df)))
        if float(scores_df.iloc[i].get("faithfulness", 0) or 0) < 0.5
    ]
    hallu_rate = len(low_faith) / len(valid) * 100 if valid else 0

    print(f"\n{'─'*60}")
    print(f"  {'MEAN':<32} {mean_faith:>6.2f} {mean_relev:>6.2f} {mean_prec:>6.2f}")

    # ── Optional baseline ─────────────────────────────────────────────────
    if include_baseline:
        print("\nRunning text-only baseline (first 5 cases)...")
        from vectorstore.schema import get_client, TEXT_CLASS
        from weaviate.classes.query import Filter, MetadataQuery
        from google import genai as gai

        gclient = gai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        wv = get_client()
        b_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        try:
            for r in valid[:5]:
                col = wv.collections.get(TEXT_CLASS)
                res = col.query.near_text(
                    query=r["case"]["question"], limit=5,
                    filters=Filter.by_property("sourceFile").equal(r["case"]["source_file"]),
                    return_metadata=MetadataQuery(score=True),
                )
                ctx = [obj.properties["content"] for obj in res.objects] or ["No context."]
                prompt = f"Answer using only this context:\n\n{'---'.join(ctx[:3])}\n\nQuestion: {r['case']['question']}"
                resp = gclient.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                b_data["question"].append(r["case"]["question"])
                b_data["answer"].append(clean_answer(resp.text))
                b_data["contexts"].append(ctx)
                b_data["ground_truth"].append(r["case"]["ground_truth"])
        finally:
            wv.close()

        b_scores = evaluate(
            Dataset.from_dict(b_data),
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm, embeddings=embeddings, raise_exceptions=False,
        ).to_pandas()

        b_faith = b_scores["faithfulness"].fillna(0).mean()
        b_relev = b_scores["answer_relevancy"].fillna(0).mean()
        print(f"\n  {'MULTIMODAL':<32} {mean_faith:>6.2f} {mean_relev:>6.2f}")
        print(f"  {'TEXT-ONLY BASELINE':<32} {b_faith:>6.2f} {b_relev:>6.2f}")
        print(f"\n  Faithfulness improvement : {(mean_faith - b_faith)*100:+.1f}pp")
        print(f"  Relevancy improvement    : {(mean_relev - b_relev)*100:+.1f}pp")

    # ── Final scorecard ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SCORECARD — resume-ready numbers")
    print(f"{'='*60}")
    print(f"  Faithfulness          : {mean_faith:.0%}")
    print(f"  Answer relevancy      : {mean_relev:.0%}")
    print(f"  Context precision     : {mean_prec:.0%}")
    print(f"  Hallucination rate    : {hallu_rate:.0f}% answers below 0.5 faithfulness")
    print(f"  Pass rate (faith≥0.7) : {pass_count}/{len(valid)} questions")
    print(f"  Test cases            : {len(valid)}/{len(cases)} completed")
    print(f"  Documents tested      : 3 (ozempic, metformin, semaglutide)")
    print(f"  Modalities covered    : text, table, chart")
    if low_faith:
        print(f"  Flagged for review    : {', '.join(low_faith)}")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────
    out = Path(__file__).parent / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out, "w") as f:
        json.dump({
            "scores": {
                "faithfulness":       round(float(mean_faith), 4),
                "answer_relevancy":   round(float(mean_relev), 4),
                "context_precision":  round(float(mean_prec),  4),
                "hallucination_rate": round(hallu_rate, 1),
                "pass_rate":          f"{pass_count}/{len(valid)}",
            },
            "n_cases": len(valid),
            "results": [
                {"id": r["case"]["id"], "answer": r["answer"][:300]}
                for r in valid
            ],
        }, f, indent=2)
    print(f"  Results saved → {out}\n")

    return mean_faith, mean_relev, mean_prec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc",      type=str, default=None)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    run_eval(doc_filter=args.doc, include_baseline=args.baseline)