# LLM-RAG Retrieval Optimisation: An Empirical Study of Fixed and Adaptive Parameters

This rpeo contains the code and data needed to reproduce the experiments reported in the paper on adaptive retrieval parameter selection for RAG.
It uses a single entry point, main.py, which runs the pipeline end-to-end in the same order as described in the paper.

---

## Prerequisites

- **Python**: version **3.10** is required.  
- **Conda** (For environment management). 

---

### 1. Create a Python environment

```bash
conda create -n datamite_env python=3.10 -y
conda activate datamite_env
```
---

### 2. Install required packages

From inside RAGDataMite:
```bash
pip install -r requirements.txt
```
---

### 3. Data & Paths

Test sets are available at:

```bash
RAG/Evaluation/data/TestSet/
├── test_set_direct_domain_questions.json
├── test_set_direct_domain_relationship_questions.json
├── test_set_domain_typo_questions.json
└── test_set_out_of_domain_trivia.json
```

The ChromaDB persist directory (vector store) defaults to:
```bash
RAG/ProcessedDocuments/chroma_db
```

If it doesn’t exist, main.py will create/use it as needed.


---

### 4. Run Everything 
```bash
python main.py
```

This will:
```bash
1. Load (or build) the vector store 
2. Run the full retrieval pipeline in the reported order
3. Evaluate each query category
4. Print and save per-category metrics (Precision, Recall, F1, MRR, RankFirst (hits-only), HitRate, AvgSim, OOD FP rate)
```
