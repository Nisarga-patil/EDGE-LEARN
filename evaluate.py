# filename: evaluate.py (The Final, Definitive Version)

import pandas as pd
import os
from tqdm import tqdm

# --- 1. Import all necessary functions and the 'app' module itself ---
from app import (
    setup_application,
    process_hybrid_pdfs,
    get_document_summary_context,
    get_hybrid_enhanced_answer
)
import app

def run_evaluation(pdf_path_to_process):
    """
    Processes a given PDF and then runs the RAG system on the test dataset.
    """
    # --- 1. Process the specified PDF ---
    print(f"\n📚 Processing the document for evaluation: {pdf_path_to_process}")
    if not os.path.exists(pdf_path_to_process):
        print(f"❌ Error: PDF file not found at '{pdf_path_to_process}'")
        return

    # This function will now correctly update the global FAISS index in the 'app' module
    process_hybrid_pdfs([pdf_path_to_process])
    print("✅ Document processing complete.")

    # --- 2. Load the test dataset ---
    dataset_path = "test_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Test dataset not found at {dataset_path}")
        return

    test_df = pd.read_csv(dataset_path)
    questions = test_df["question"].tolist()
    ground_truths = test_df["ground_truth"].tolist()

    # --- 3. Run the RAG pipeline for each question ---
    results_for_ragas = []
    print(f"\n🚀 Running evaluation on {len(questions)} questions...")

    for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions), desc="Evaluating Questions"):
        answer_text, retrieved_images, context_text = get_hybrid_enhanced_answer(
            question,
            [],
            target_pdf=pdf_path_to_process
        )
        
        # AFTER (The corrected code)
        results_for_ragas.append({
            "question": question,
            "answer": answer_text,
            "contexts": context_text if context_text else [],
            "ground_truth": ground_truth
        })

    # --- 4. Save the results ---
    results_df = pd.DataFrame(results_for_ragas)
    results_path = "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n✅ Evaluation complete. Results for RAGAs saved to: {results_path}")
    print("\n--- Results Preview ---")
    print(results_df.head())


if __name__ == "__main__":
    # --- This is the final, correct setup ---
    print("🚀 Setting up the main application environment...")
    setup_application()
    print("✅ Environment is ready.")
    
    pdf_to_evaluate = "L4.pdf" 
    
    # Check that the FAISS index was loaded correctly in the main app module
    if app.faiss_index is None:
        print("⚠️  Knowledge base not pre-loaded. Processing PDF before evaluation.")
        run_evaluation(pdf_to_evaluate)
    else:
        print("✅ Knowledge base already loaded. Proceeding to evaluation.")
        run_evaluation(pdf_to_evaluate)