# filename: create_dataset.py

import pandas as pd
import os

def create_test_suite():
    """
    Creates and saves a benchmark dataset with questions and ground truth answers.
    """
    # --- Define the questions and their ideal answers ---
    test_data = [
        # Category 1: Basic Retrieval (Facts in the document)
        {
            "question": "What are the three main components of a balanced diet?",
            "ground_truth": "A balanced diet has all essential nutrients, roughage, and water in the right amount."
        },
        {
            "question": "What is the disease caused by a Vitamin C deficiency?",
            "ground_truth": "A deficiency of Vitamin C causes a disease called scurvy."
        },

        # Category 2: Plausible but Incorrect (Tests for hallucination)
        {
            "question": "What is the chemical formula for glucose mentioned in the text?",
            "ground_truth": "The document does not provide the chemical formula for glucose."
        },
        {
            "question": "According to the text, what university did the nutritionist Dr. Poshita attend?",
            "ground_truth": "The document does not mention which university Dr. Poshita attended."
        },

        # Category 3: Zero-Context (Tests for refusal to answer)
        {
            "question": "What is the capital of Brazil?",
            "ground_truth": "This information is not available in the provided document."
        }
    ]

    # --- Convert to a pandas DataFrame and save as a CSV file ---
    df = pd.DataFrame(test_data)
    dataset_path = "test_dataset.csv"
    df.to_csv(dataset_path, index=False)

    print(f"✅ Test dataset created and saved successfully at: {os.path.abspath(dataset_path)}")
    print("\n--- Dataset Preview ---")
    print(df)

if __name__ == "__main__":
    # Before running, make sure pandas is installed in your venv:
    # pip install pandas
    create_test_suite()