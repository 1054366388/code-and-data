import os
import random
import numpy as np
import torch
import optuna

# CRITICAL: Fix for OMP error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from src.config import cfg
from src.data_loader import DataLoader
from src.core_modules import ProtStarCore
from src.model_engine import LLMEngine


def set_seed(seed):
    """Sets the robust random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def objective(trial, engine, prompts):
    """Optuna objective function for optimizing model hyperparameters."""
    # Define hyperparameter search spaces based on config
    lr = trial.suggest_float("learning_rate", *cfg.SEARCH_SPACE["learning_rate"], log=True)
    lora_r = trial.suggest_categorical("lora_r", cfg.SEARCH_SPACE["lora_r"])
    lora_alpha = trial.suggest_categorical("lora_alpha", cfg.SEARCH_SPACE["lora_alpha"])
    lora_dropout = trial.suggest_float("lora_dropout", *cfg.SEARCH_SPACE["lora_dropout"])
    batch_size = trial.suggest_categorical("batch_size", cfg.SEARCH_SPACE["batch_size"])
    
    print(f"\n[Optuna] Trial {trial.number}: Testing LR={lr:.2e}, R={lora_r}, Alpha={lora_alpha}, Dropout={lora_dropout:.2f}, BS={batch_size}")
    
    try:
        final_loss = engine.run_training(
            prompt_list=prompts,
            learning_rate=lr,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        return final_loss
    except Exception as e:
        print(f"[Optuna] Trial failed: {e}")
        # Return a large penalty upon failure
        return 999.0


def main():
    print("==================================================")
    print(" Prot-STAR: Framework for Omics Data Analysis")
    print("==================================================")

    # 0. System Configuration
    set_seed(cfg.RANDOM_SEED)
    print(f"[System] Seed set to {cfg.RANDOM_SEED}")

    # 1. Initialize Modules
    loader = DataLoader()
    kg_matrix = loader.load_knowledge_graph()
    core = ProtStarCore(kg_matrix)

    cohort = "COADREAD"
    print(f"\n[Phase 1] Processing Cohort: {cohort}")

    # 2. Load Data and Split into Train/Test
    X_train, X_test, y_train, y_test = loader.load_cohort(cohort)
    print(f" -> Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

    # 3. Structure-Grounded Pruning
    print(" -> Selecting features via structural abstraction...")
    # Prune based on the training set to prevent data leakage
    top_genes = core.structure_grounded_pruning(X_train)
    X_train_selected = X_train[top_genes]
    X_test_selected = X_test[top_genes]
    print(f" -> Top features: {top_genes[:5]}...")

    # 4. Adaptive Semantic Quantization
    print(" -> Quantizing continuous values to semantic tokens...")
    X_train_sem = core.adaptive_semantic_quantization(X_train_selected)
    
    # Process test set
    X_test_sem = core.adaptive_semantic_quantization(X_test_selected)

    # 5. Prompt Construction
    print(" -> Building Instruction Tuning Dataset...")
    train_prompts = []
    processed_train_data = []

    for idx, row in X_train_sem.iterrows():
        try:
            label = int(y_train.loc[idx].values[0]) if hasattr(y_train, 'loc') else int(y_train[idx])
        except:
            label = 0

        full_prompt = core.construct_prompt(row, label)
        train_prompts.append(full_prompt)

        processed_train_data.append({
            "Sample_ID": idx,
            "Split": "Train",
            "Semantic_Profile": row.to_dict(),
            "Ground_Truth": label,
            "Full_Prompt": full_prompt
        })

    # Save the processed data securely
    save_path = os.path.join(cfg.BASE_DIR, f"{cohort}_processed_train_prompts.csv")
    pd.DataFrame(processed_train_data).to_csv(save_path, index=False)
    print(f" -> Processed train data saved to {save_path}")

    # 6. Run Training with Hyperparameter Optimization
    print("\n[Phase 2] Initializing Hyperparameter Tuning & Model Training")
    try:
        engine = LLMEngine()
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, engine, train_prompts), n_trials=cfg.OPTUNA_TRIALS)

        print("\n==================================================")
        print("[Optimization Results]")
        print("==================================================")
        print(f"  Best Trial: {study.best_trial.number}")
        print(f"  Best Loss: {study.best_value}")
        print("  Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        print("\n[Success] Training pipeline finished successfully.")
    except Exception as e:
        print("\n[Error] Model Execution Failed.")
        print(f"Reason: {e}")
        print("-" * 50)
        print("Troubleshooting for Legitimate Run:")
        print("1. Access Token: Ensure you ran 'huggingface-cli login'.")
        print("2. Gate Access: Accept model license on HuggingFace website.")
        print("3. Hardware: Check if CUDA GPU is available.")


if __name__ == "__main__":
    main()