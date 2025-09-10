import argparse
import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from RAG.Retrieval.adaptive_retriever import AdaptiveRetriever
from RAG.Evaluation.retriever_evaluation import (collect_scores,
                                                 load_questions,
                                                 summarize_and_plot)
from RAG.Evaluation.test_set_creation import (create_entity_name_typo,
                                              create_question_word_typo,
                                              generate_bsc_family_questions,
                                              generate_bsc_subfamily_questions,
                                              generate_criteria_questions,
                                              generate_kpi_questions,
                                              generate_objective_questions,
                                              generate_relationship_questions)
from RAG.Evaluation.test_set_out_of_domain import (convert_to_custom_format,
                                                   fetch_trivia_questions)
from RAG.Evaluation.tuning_params import (evaluate_grid, load_test_set,
                                          plot_precision_recall,
                                          plotFPR)
from RAG.KB.generating_embeddings import generate_embeddings
from RAG.KB.ingest_documents import (ingest_documents,
                                     load_all_csvs_as_documents)
from RAG.KB.preprocess_data import preprocess_data
from RAG.KB.vector_DB import create_vectorstore
from RAG.LLM.rag_controller import rag_with_validation
from RAG.Evaluation.evaluate_adaptive_retrieval import run_evaluation
from RAG.Retrieval.retriever import (get_retrieval_results,
                                     get_retrieval_with_threshold,
                                     setup_retriever)

def run_index(persist_dir: str):
    """
    Full indexing pipeline: preprocess -> ingest -> embeddings -> vector DB.
    """
    # 1) Preprocess CSVs
    output_dir = "RAG/Content/ProcessedFiles"
    processed_files = preprocess_data(output_dir)

    # 2) Ingest to LangChain Documents (+ writes all_documents.pkl)
    documents = ingest_documents()

    # 3) Generate embeddings
    generate_embeddings()

    # 4) Create/update Chroma vector DB
    _ = create_vectorstore()
    logging.info(f"Index build complete. Vector DB at: {persist_dir}")

if __name__ == '__main__':

    print("Interpreter:", sys.executable)
    parser = argparse.ArgumentParser(
        description="RAGForDatamite pipeline runner"
    )
    parser.add_argument(
        "--mode",
        choices=["index", "demo", "eval", "all"],
        default="demo",
        help=(
            "index = build vector DB (preprocess -> ingest -> embed -> vectorstore); "
        ),
    )
    parser.add_argument(
        "--persist_dir",
        default="RAG/ProcessedDocuments/chroma_db",
        help="Directory for Chroma vector DB."
    )

    args = parser.parse_args()

    logger = logging.getLogger("datamite")
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.mode in ("index", "all"):
        run_index(args.persist_dir)

    logging.info("Done.")

    # 1. Preprocess CSV Files to clean for JSON related issues

    output_dir = "RAG/Content/ProcessedFiles"

    try:
        processed_files = preprocess_data(output_dir)

        logger.info("\nPreprocessing Summary:")
        logger.info("=====================")
        for file_type, file_path in processed_files.items():
            logger.info(f"{file_type}: {file_path}")

        logger.info("\nTo use these files with KB.py:")
        logger.info("```python")
        logger.info("from KB import load_csv_as_documents, load_all_csvs_as_documents")
        logger.info("")
        logger.info("# Define processed file paths")
        logger.info("csv_files = {")
        for file_type, file_path in processed_files.items():
            logger.info(f"    '{file_type}': '{file_path}',")
        logger.info("}")
        logger.info("")
        logger.info("# Load all documents")
        logger.info("all_docs = load_all_csvs_as_documents(csv_files)")
        logger.info("print(f\"Loaded {len(all_docs)} total documents\")")
        logger.info("```")

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        sys.exit(1)

    
    # 2. Documents Ingestion - converting to Langchain documents

    documents = ingest_documents()

    if documents:
        logger.info("\nSample document:")
        logger.info("-" * 80)

        sample_doc = next((doc for doc in documents if doc.metadata.get("type") == "KPI"), documents[0])

        logger.info("Content:")
        logger.info(sample_doc.page_content[:500] + "..." if len(sample_doc.page_content) > 500 else sample_doc.page_content)
        logger.info("-" * 80)
        logger.info("Metadata:")
        logger.info(sample_doc.metadata)
    

    # 3. Analysing if there is a need for text chunking

    with open('RAG/ProcessedDocuments/all_documents.pkl', 'rb') as f:
        documents = pickle.load(f)

    doc_lengths = defaultdict(list)
    for doc in documents:
        doc_type = doc.metadata.get('type', 'Unknown')
        length = len(doc.page_content)
        doc_lengths[doc_type].append(length)

    for doc_type, lengths in doc_lengths.items():
        logger.info(f"\n{doc_type} Documents:")
        logger.info(f"  Count: {len(lengths)}")
        logger.info(f"  Average length: {np.mean(lengths):.1f} characters")
        logger.info(f"  Min length: {min(lengths)} characters")
        logger.info(f"  Max length: {max(lengths)} characters")

        if lengths:
            longest_idx = np.argmax(lengths)
            longest_doc = next((doc for i, doc in enumerate(documents)
                                if doc.metadata.get('type') == doc_type
                                and i == longest_idx), None)
            if longest_doc:
                logger.info(f"  Longest document name: {longest_doc.metadata.get('name', 'Unknown')}")
                logger.info(f"  First 200 chars: {longest_doc.page_content[:200]}...")

    # 4. Generating vector embeddings from Langchain documents

    generate_embeddings()
    

    # 5. Creating vector DB and save embeddings to Vector DB

    vectorstore = create_vectorstore()

# -----------------------------------------------------------------------------

    # *****************************************
    # RAG EVALUATION
    # *****************************************


    # Step 1: Generate test set with 5 different question types


    # 1. Direct Domain Relevant Questions

    csv_files = {
        'KPIs': 'RAG/Evaluation/data/clean_KPIs.csv',
        'Objectives': 'RAG/Evaluation/data/clean_Objectives.csv',
        'BSC_families': 'RAG/Evaluation/data/clean_BSC_families.csv',
        'BSC_subfamilies': 'RAG/Evaluation/data/clean_BSC_subfamilies.csv',
        'Criteria': 'RAG/Evaluation/data/clean_Criteria.csv'
    }

    all_questions = []

    print("Generating test questions...")

    # KPIs
    if os.path.exists(csv_files['KPIs']):
        df_kpis = pd.read_csv(csv_files['KPIs'])
        kpi_questions = generate_kpi_questions(df_kpis)
        all_questions.extend(kpi_questions)
        print(f"Generated {len(kpi_questions)} KPI questions")

    # Objectives
    if os.path.exists(csv_files['Objectives']):
        df_objectives = pd.read_csv(csv_files['Objectives'])
        objective_questions = generate_objective_questions(df_objectives)
        all_questions.extend(objective_questions)
        print(f"Generated {len(objective_questions)} Objective questions")

    # BSC Families
    if os.path.exists(csv_files['BSC_families']):
        df_families = pd.read_csv(csv_files['BSC_families'])
        family_questions = generate_bsc_family_questions(df_families)
        all_questions.extend(family_questions)
        print(f"Generated {len(family_questions)} BSC Family questions")

    # BSC Subfamilies
    if os.path.exists(csv_files['BSC_subfamilies']):
        df_subfamilies = pd.read_csv(csv_files['BSC_subfamilies'])
        subfamily_questions = generate_bsc_subfamily_questions(df_subfamilies)
        all_questions.extend(subfamily_questions)
        print(f"Generated {len(subfamily_questions)} BSC Subfamily questions")

    # Criteria
    if os.path.exists(csv_files['Criteria']):
        df_criteria = pd.read_csv(csv_files['Criteria'])
        criteria_questions = generate_criteria_questions(df_criteria)
        all_questions.extend(criteria_questions)
        print(f"Generated {len(criteria_questions)} Criteria questions")

    print(f"\nTotal questions generated: {len(all_questions)}")

    for idx, q in enumerate(all_questions, start=1):
        q['question_id'] = idx

        # Ground truth: "<document_type>::<entity_name>"
        # e.g. "KPI::Access Cost", "Objective::Data Platform Operator"
        q['ground_truth_docs'] = [
            f"{q['expected_document_type']}::{q['entity_name']}"
        ]

    # Save to JSON file
    output_file = 'RAG/Evaluation/data/TestSet/test_set_direct_domain_questions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Test set saved to: {output_file}")

    # Print summary statistics
    print("\nSummary by entity type:")
    entity_counts = {}
    for q in all_questions:
        entity_type = q['entity_type']
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count} questions")

    # Show sample questions
    print("\nSample questions:")
    for i in range(min(5, len(all_questions))):
        print(f"  - {all_questions[i]['question']}")

    # 2: Domain relevant with typos
    # Generating typo variations of direct domain questions.

    # Load the direct domain questions
    input_file = 'RAG/Evaluation/data/TestSet/test_set_direct_domain_questions.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        direct_questions = json.load(f)
    print(f"Loaded {len(direct_questions)} direct domain questions")

    typo_questions = []

    print(f"Creating typo variations for ALL {len(direct_questions)} questions...")

    for idx, question in enumerate(direct_questions):
        if idx % 2 == 0:
            # Entity name typo for even indices
            typo_question = create_entity_name_typo(question)
        else:
            # Question word typo for odd indices
            typo_question = create_question_word_typo(question)

        typo_questions.append(typo_question)

    print(f"Generated {len(typo_questions)} typo questions")

    for idx, q in enumerate(typo_questions, start=1):
        q['question_id'] = idx

        # Ground truth: "<document_type>::<entity_name>"
        # e.g. "KPI::Access Cost", "Objective::Data Platform Operator"
        q['ground_truth_docs'] = [
            f"{q['expected_document_type']}::{q['entity_name']}"
        ]

    output_file = 'RAG/Evaluation/data/TestSet/test_set_domain_typo_questions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(typo_questions, f, indent=2, ensure_ascii=False)

    print(f"Test set saved to: {output_file}")


    # 3. Out of Domain

    print("Fetching trivia questions from Open Trivia Database...")

    trivia_questions = fetch_trivia_questions(num_questions=986)
    convert_to_custom_format(trivia_questions, "RAG/Evaluation/data/TestSet/test_set_out_of_domain_trivia.json")

    print("Created simple out-of-domain questions for testing")
    print("\nTo get 500+ questions from Open Trivia DB, uncomment the API calls in the main section")


    # Domain Relevant with 2 ground truths, domain relationship ques

    # Load all entities by type
    entities_by_type = {}

    print("Loading entities from CSV files...")

    # KPIs
    if os.path.exists(csv_files['KPIs']):
        df_kpis = pd.read_csv(csv_files['KPIs'])
        entities_by_type['KPI'] = [{'name': row['name'].strip()} for _, row in df_kpis.iterrows()]
        print(f"Loaded {len(entities_by_type['KPI'])} KPIs")

    # Objectives
    if os.path.exists(csv_files['Objectives']):
        df_objectives = pd.read_csv(csv_files['Objectives'])
        entities_by_type['Objective'] = [{'name': row['name'].strip()} for _, row in df_objectives.iterrows()]
        print(f"Loaded {len(entities_by_type['Objective'])} Objectives")

    # BSC Families
    if os.path.exists(csv_files['BSC_families']):
        df_families = pd.read_csv(csv_files['BSC_families'])
        entities_by_type['BSC Family'] = [{'name': row['name'].strip()} for _, row in df_families.iterrows()]
        print(f"Loaded {len(entities_by_type['BSC Family'])} BSC Families")

    # BSC Subfamilies
    if os.path.exists(csv_files['BSC_subfamilies']):
        df_subfamilies = pd.read_csv(csv_files['BSC_subfamilies'])
        entities_by_type['BSC Subfamily'] = [{'name': row['name'].strip()} for _, row in df_subfamilies.iterrows()]
        print(f"Loaded {len(entities_by_type['BSC Subfamily'])} BSC Subfamilies")

    # Criteria
    if os.path.exists(csv_files['Criteria']):
        df_criteria = pd.read_csv(csv_files['Criteria'])
        entities_by_type['Criteria'] = [{'name': row['name'].strip()} for _, row in df_criteria.iterrows()]
        print(f"Loaded {len(entities_by_type['Criteria'])} Criteria")

    # Generate relationship questions
    print("\nGenerating relationship questions...")
    all_questions = generate_relationship_questions(entities_by_type, target_questions=987)

    print(f"\nTotal questions generated: {len(all_questions)}")

    # Add question IDs
    for idx, q in enumerate(all_questions, start=1):
        q['question_id'] = idx

    # Save to JSON file
    output_file = 'RAG/Evaluation/data/TestSet/test_set_direct_domain_relationship_questions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"Test set saved to: {output_file}")

    # ----------------------------------------------------------------------------------------------------------------


    # 2. Calculating similarity scores for top-1 hit for test sets to see similarity distributions

    # 1) Initialize retriever (top-1)

    k=1
    retriever = setup_retriever(
        persist_directory="RAG/ProcessedDocuments/chroma_db",
        vectorstore = vectorstore,
        k=k
    )

    # 2) Load test sets
    direct_qs = load_questions("RAG/Evaluation/data/TestSet/test_set_direct_domain_questions.json")
    out_qs = load_questions("RAG/Evaluation/data/TestSet/test_set_out_of_domain_trivia.json")
    domain_typo_qs = load_questions("RAG/Evaluation/data/TestSet/test_set_domain_typo_questions.json")
    domain_relation_qs = load_questions("RAG/Evaluation/data/TestSet/test_set_direct_domain_relationship_questions.json")

    # 3) Collect scores
    df_direct = collect_scores(direct_qs, retriever, k)
    df_out = collect_scores(out_qs, retriever, k)
    df_direct_typo = collect_scores(domain_typo_qs, retriever, k)
    df_direct_relation = collect_scores(domain_relation_qs, retriever, k)
    df_all = pd.concat([df_direct, df_out, df_direct_typo, df_direct_relation], ignore_index=True)

    # 4) Summarize and plot
    summarize_and_plot(
        df_all,
        out_hist_path="RAG/Evaluation/data/TestSet/cosine_similarity_histograms.pdf",
        out_summary_path="RAG/Evaluation/data/TestSet/similarity_summary.txt",
    )
    
    
    # 3. Running gird search to find optimal k and threshold

    direct_ques_testset = load_test_set("RAG/Evaluation/data/TestSet/test_set_direct_domain_questions.json")
    out_dom_ques_testset = load_test_set("RAG/Evaluation/data/TestSet/test_set_out_of_domain_trivia.json")
    in_dom_typos_ques_testset = load_test_set("RAG/Evaluation/data/TestSet/test_set_domain_typo_questions.json")
    in_dom_relation_ques_testset = load_test_set("RAG/Evaluation/data/TestSet/test_set_direct_domain_relationship_questions.json")

    ks = [1, 2, 3, 4, 5, 7, 10]
    thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.5, 0.7, 0.9]

    df_grid = evaluate_grid(
        test_questions=in_dom_relation_ques_testset,
        persist_directory="RAG/ProcessedDocuments/chroma_db",
        vectorstore=vectorstore,
        ks=ks,
        thresholds=thresholds
    )

    df_grid.to_csv("RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_relational.csv", index=False)
    print(df_grid.sort_values(["f1"], ascending=False).head(10))


    # Plotting

    # path_direct = "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_direct_domain.csv"
    # path_typos = "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_indomain_typo.csv"
    # path_oos = "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_outOf_domain.csv"
    # path_relation = "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_relational.csv"
    #
    # df_direct = pd.read_csv(path_direct)
    # df_typos = pd.read_csv(path_typos)
    # df_oos = pd.read_csv(path_oos)
    # df_relational = pd.read_csv(path_relation)
    #
    # # plot_precision_recall(df_direct, "Precision-Recall (Direct Domain Queries)", "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_direct_domain.pdf")
    # # plot_precision_recall(df_typos, "Precision-Recall (Noisy/Typos Queries)", "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_indomain_typo.pdf")
    # # plot_precision_recall(df_relational, "Precision-Recall (Relational Queries)", "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_relational.pdf")
    #
    # plotFPR(df_oos, "OOD FPR vs. Threshold (lines by k)", "RAG/Evaluation/data/TestSet/GridSearchResults/grid_search_outOf_domain.pdf")

# ----------------------------------------------------------------------------------------------------------------


# 3 step MVP for Adaptive Retriever

    # Loading vector store
    persist_directory = "RAG/ProcessedDocuments/chroma_db"
    model_name = "all-MiniLM-L6-v2"
    logger.info(f"Loading vector store from {persist_directory}")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    # vectorstore = Chroma(persist_directory=persist_directory,
    #                      embedding_function=embedding_function)

    csv_files = {
        'KPIs': 'RAG/Content/ProcessedFiles/clean_KPIs.csv',
        'Objectives': 'RAG/Content/ProcessedFiles/clean_Objectives.csv',
        'BSC_families': 'RAG/Content/ProcessedFiles/clean_BSC_families.csv',
        'BSC_subfamilies': 'RAG/Content/ProcessedFiles/clean_BSC_subfamilies.csv',
        'Criteria': 'RAG/Content/ProcessedFiles/clean_Criteria.csv'
    }
    documents = load_all_csvs_as_documents(csv_files)


# Simple initialization - no documents needed for dense-only
retriever = AdaptiveRetriever(
    vectorstore=vectorstore,
    k_init=30,
    pool_cap=20
)

# Testing adaptive retrieval with test sets

testsets = {
    "direct": "RAG/Evaluation/data/TestSet/test_set_direct_domain_questions.json",
    "relational": "RAG/Evaluation/data/TestSet/test_set_direct_domain_relationship_questions.json",
    "noisy": "RAG/Evaluation/data/TestSet/test_set_domain_typo_questions.json",
    "ood": "RAG/Evaluation/data/TestSet/test_set_out_of_domain_trivia.json",
}

all_results = {}

for name, path in testsets.items():
    results = run_evaluation([path], stage="final", ood_fp_mode="nonempty", vectorstore=vectorstore)
    all_results[name] = results
    print(f"\n{name.capitalize()} results:")
    print("Aggregate:", results["aggregate"])

