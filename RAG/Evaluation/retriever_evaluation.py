import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from RAG.Retrieval.retriever import setup_retriever


def load_questions(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_scores(questions: list, retriever, k: int = 1) -> pd.DataFrame:
    """
    For each question, run similarity_search_with_score and record:
      - distance (d)
      - rescaled sim_score = 1/(1+d)
      - raw_cosine_sim = 1 - d
    """
    records = []
    for q in questions:
        results = retriever.vectorstore.similarity_search_with_score(q["question"], k=k)
        if not results:
            # no hits, skip
            continue
        doc, dist = results[0]
        sim_scaled = 1 / (1 + dist)
        cos_sim    = 1 - dist
        records.append({
            "question_id":       q["question_id"],
            "category":          q["category"],
            "distance":          dist,
            "sim_scaled":        sim_scaled,
            "cosine_similarity": cos_sim
        })
    return pd.DataFrame(records)

def summarize_and_plot(
    df: pd.DataFrame,
    out_hist_path: str = "RAG/Evaluation/data/TestSet/cosine_similarity_histograms.pdf",
    out_summary_path: str = "RAG/Evaluation/data/TestSet/similarity_summary.txt",
    use_density: bool = False,            # set True if category sizes differ a lot
    xlim: tuple = (0.0, 0.85),
    xticks: np.ndarray | None = None,     # e.g., np.arange(0.0, 0.86, 0.1)
    ylim: tuple | None = None,            # e.g., (0, 200) if not using density
    yticks: list | None = None,
    nbins: int = 24,                      # identical bins for all panels
    mark_tau: float | None = 0.30,        # dashed line at τ
    band: tuple | None = (0.30, 0.35),    # light band (τ range)
):
    """
    Summarize cosine similarity per category and plot comparable histograms.

    Expects columns: ['category', 'cosine_similarity'] (others optional).
    """

    # ---- Summary table (count/min/median/mean/max) ----
    summary = (
        df.groupby("category")[["cosine_similarity"]]
          .agg(count=("cosine_similarity", "count"),
               min=("cosine_similarity", "min"),
               median=("cosine_similarity", "median"),
               mean=("cosine_similarity", "mean"),
               max=("cosine_similarity", "max"))
          .round(3)
    )
    print("\n=== Similarity Score Summary by Category ===")
    print(summary)

    # Persist summary
    with open(out_summary_path, "w") as f:
        f.write("=== Similarity Score Summary by Category ===\n")
        f.write(summary.to_string())

    # ---- Consistent plotting config ----
    if xticks is None:
        xticks = np.arange(xlim[0], xlim[1] + 1e-9, 0.1)
    bins = np.linspace(xlim[0], xlim[1], nbins + 1)

    # Fixed order & display names
    order = [
        "direct_domain_relevant",                 # Direct Domain
        "out_of_domain",                          # OOD
        "domain_relevant_with_typos",             # Noisy/Typo
        "direct_domain_relevant_relationship",    # Relational
    ]
    name_map = {
        "direct_domain_relevant": "Direct Domain Queries",
        "out_of_domain": "Out-of-Domain Queries",
        "domain_relevant_with_typos": "Noisy/Typo Queries",
        "direct_domain_relevant_relationship": "Relational Queries",
    }

    cats_present = [c for c in order if c in df["category"].unique()]

    # Create 2x2 grid; share axes for fair comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax_idx, cat in enumerate(cats_present):
        ax = axes[ax_idx]
        data = df.loc[df["category"] == cat, "cosine_similarity"].to_numpy()

        ax.hist(data, bins=bins, edgecolor="black", density=use_density)
        ax.set_title(name_map.get(cat, cat))
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency" if not use_density else "Density")
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)

        if ylim is not None and not use_density:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)

    # Force labels on all subplots
    for ax in axes:
        ax.tick_params(labelbottom=True, labelleft=True)

    plt.tight_layout()
    plt.savefig(out_hist_path, bbox_inches="tight")
    plt.show()

    return summary