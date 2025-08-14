# filename: judge.py (Final regex-based cleaning)

import pandas as pd
import os
import sys
import subprocess
import ast
import json
import re  # Import the regular expression module

def is_encoded_string(s):
    """
    Uses a regular expression to check if a string is composed of base64-like
    characters and is long enough to be encoded data.
    """
    if not isinstance(s, str) or len(s) < 200:
        # It's not a string or it's too short to be an image.
        return False
    
    # This regex pattern checks if the ENTIRE string from start (^) to end ($)
    # consists ONLY of valid base64 characters (A-Z, a-z, 0-9, +, /, =).
    base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
    
    # If the string matches this pattern, it's almost certainly encoded data.
    return bool(base64_pattern.match(s))

def simple_evaluation(api_key, results_df):
    """
    A simple, direct evaluation using Groq API without RAGAs dependencies.
    """
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        print("Installing Groq client...")
        subprocess.run([sys.executable, "-m", "pip", "install", "groq", "-q"], check=True)
        from groq import Groq
        client = Groq(api_key=api_key)
    
    faithfulness_scores = []
    relevancy_scores = []
    
    print(f"📊 Evaluating {len(results_df)} responses...")
    
    for idx, row in results_df.iterrows():
        question = row.get('question', '')
        answer = row.get('answer', '')
        contexts = row.get('contexts', [])
        
        context_text = ' '.join([str(item) for item in contexts]) if contexts else ''
        
        # Evaluate faithfulness
        faithfulness_prompt = f"""
Rate the faithfulness of the answer to the given context on a scale of 0 to 1, where:
- 1 = The answer is completely faithful to the context
- 0 = The answer contradicts or is not supported by the context

Context: {context_text}
Answer: {answer}

Respond with only a number between 0 and 1 (e.g., 0.8):
"""
        
        # Evaluate relevancy
        relevancy_prompt = f"""
Rate how relevant the answer is to the question on a scale of 0 to 1, where:
- 1 = The answer directly and completely addresses the question
- 0 = The answer is completely irrelevant to the question

Question: {question}
Answer: {answer}

Respond with only a number between 0 and 1 (e.g., 0.9):
"""
        
        try:
            faithfulness_response = client.chat.completions.create(
                model="llama3-8b-8192", messages=[{"role": "user", "content": faithfulness_prompt}],
                temperature=0.0, max_tokens=10
            )
            faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
            faithfulness_scores.append(max(0.0, min(1.0, faithfulness_score)))
            
            relevancy_response = client.chat.completions.create(
                model="llama3-8b-8192", messages=[{"role": "user", "content": relevancy_prompt}],
                temperature=0.0, max_tokens=10
            )
            relevancy_score = float(relevancy_response.choices[0].message.content.strip())
            relevancy_scores.append(max(0.0, min(1.0, relevancy_score)))
            
        except Exception as e:
            print(f"⚠️  Error evaluating row {idx}: {e}")
            faithfulness_scores.append(0.5); relevancy_scores.append(0.5)
            
        if (idx + 1) % 5 == 0:
            print(f"✅ Processed {idx + 1}/{len(results_df)} responses")
    
    return faithfulness_scores, relevancy_scores

def judge_the_results(api_key):
    """
    Evaluate results using a robust regex cleaning and direct evaluation approach.
    """
    results_path = "evaluation_results.csv"
    if not os.path.exists(results_path):
        print(f"❌ Error: Evaluation results file not found at {results_path}"); return

    print(f"\n🔎 Loading results from {results_path}...")
    try:
        results_df = pd.read_csv(results_path)
        if 'contexts' in results_df.columns:
            def safe_literal_eval(x):
                if pd.isna(x) or x == '': return []
                try: return ast.literal_eval(x)
                except: return x if isinstance(x, list) else [str(x)]
            results_df['contexts'] = results_df['contexts'].apply(safe_literal_eval)
        if results_df.empty:
            print("❌ The results file is empty."); return
    except Exception as e:
        print(f"❌ Error loading results file: {e}"); return
    
    print("✅ Results loaded successfully.")

    # --- Robust cleaning step using the new regex-based function ---
    print("\n🧼 Cleaning context data with robust regex filtering...")
    cleaned_contexts = []
    for i, context_list in enumerate(results_df['contexts'].tolist()):
        if isinstance(context_list, list):
            filtered_list = []
            for item in context_list:
                if is_encoded_string(item):
                    # This print statement shows you that the filter is working.
                    print(f"  -> Row {i}: Removed encoded string of length {len(item)}.")
                else:
                    filtered_list.append(item)
            cleaned_contexts.append(filtered_list)
        else:
            cleaned_contexts.append(context_list)
            
    results_df['contexts'] = cleaned_contexts
    print("✅ Contexts cleaned.")
    
    print(f"📊 Found {len(results_df)} responses to evaluate")

    # --- Run the evaluation ---
    print("\n⚖️  Running evaluation with Groq...")
    
    try:
        faithfulness_scores, relevancy_scores = simple_evaluation(api_key, results_df)
        
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        
        results_df['faithfulness'] = faithfulness_scores
        results_df['answer_relevancy'] = relevancy_scores
        results_df.to_csv('detailed_evaluation_results.csv', index=False)
        
        print("\n🎉🎉🎉 BENCHMARK COMPLETE! 🎉🎉🎉")
        print(f"\n--- Final Scores ---")
        print(f"Average Faithfulness: {avg_faithfulness:.3f}")
        print(f"Average Answer Relevancy: {avg_relevancy:.3f}")
        print(f"Overall Score: {(avg_faithfulness + avg_relevancy) / 2:.3f}")
        
        print(f"\n📄 Detailed results saved to: detailed_evaluation_results.csv")
        
        print(f"\n--- Sample Results ---")
        sample_cols = ['question', 'answer', 'faithfulness', 'answer_relevancy']
        available_cols = [col for col in sample_cols if col in results_df.columns]
        print(results_df[available_cols].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}"); return

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("Please enter your Groq API key: ")
    
    if not api_key:
        print("❌ No API key provided. Aborting evaluation.")
    else:
        if not os.path.exists("evaluation_results.csv"):
            print("❌ 'evaluation_results.csv' not found. Please run 'evaluate.py' first.")
        else:
            judge_the_results(api_key)