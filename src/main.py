import os

# CRITICAL: Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from src.config import cfg
from src.data_loader import DataLoader
from src.core_modules import ProtStarCore
from src.model_engine import LLMEngine


def main():
    print("==================================================")
    print(" Prot-STAR: Neuro-Symbolic Omics Framework")
    print("==================================================")

    # 1. Initialize Modules
    loader = DataLoader()
    kg_matrix = loader.load_knowledge_graph()
    core = ProtStarCore(kg_matrix)

    cohort = "COADREAD"
    print(f"\n[Phase 1] Processing Cohort: {cohort}")

    # 2. Load Data
    X, y = loader.load_cohort(cohort)
    print(f" -> Loaded {len(X)} samples.")

    # 3. Structure-Grounded Pruning (Eq. 4)
    print(" -> Selecting features via Knowledge Graph...")
    top_genes = core.structure_grounded_pruning(X)
    X_selected = X[top_genes]
    print(f" -> Top features: {top_genes[:5]}...")

    # 4. Adaptive Semantic Quantization (Eq. 2 & 3)
    print(" -> Quantizing continuous values to semantic tokens...")
    X_sem = core.adaptive_semantic_quantization(X_selected)

    # 5. Prompt Construction (Eq. 5)
    print(" -> Building Instruction Tuning Dataset...")
    prompts = []
    processed_data = []  # To save intermediate results

    for idx, row in X_sem.iterrows():
        # Handle label extraction safely
        try:
            label = int(y.iloc[idx].values[0]) if hasattr(y, 'iloc') else int(y[idx])
        except:
            label = 0

        full_prompt = core.construct_prompt(row, label)
        prompts.append(full_prompt)

        # Save structured data for verification
        processed_data.append({
            "Sample_ID": idx,
            "Semantic_Profile": row.to_dict(),
            "Ground_Truth": label,
            "Full_Prompt": full_prompt
        })

    # Save the processed "Reasoning Input" immediately
    # This ensures you have files even if Training fails due to GPU/Token issues
    save_path = os.path.join(cfg.BASE_DIR, f"{cohort}_processed_prompts.csv")
    pd.DataFrame(processed_data).to_csv(save_path, index=False)
    print(f" -> Processed data saved to {save_path}")

    # 6. Run Training (Legitimate LLM Execution)
    print("\n[Phase 2] Initializing Neural Training (Llama-3-8B)")
    try:
        engine = LLMEngine()
        engine.run_training(prompts)
        print("\n[Success] Training pipeline finished successfully.")
    except Exception as e:
        print("\n[Error] Model Execution Failed.")
        print(f"Reason: {e}")
        print("-" * 50)
        print("Troubleshooting for Legitimate Run:")
        print("1. Access Token: Ensure you ran 'huggingface-cli login'.")
        print("2. Gate Access: Accept Llama-3 license on HuggingFace website.")
        print("3. Hardware: Check if CUDA GPU is available.")


if __name__ == "__main__":
    main()