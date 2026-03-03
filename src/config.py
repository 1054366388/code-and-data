import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration file for Prot-STAR Framework.
    Defines paths, essential hyperparameters, and Optuna search spaces.
    """
    # --- Paths ---
    BASE_DIR = r"C:\Users\Jie\Desktop\Prot-STAR"
    DATA_ROOT = os.path.join(BASE_DIR, "Expression Data")
    KG_PATH = os.path.join(BASE_DIR, "S.xlsx")
    OUTPUT_DIR = os.path.join(BASE_DIR, "model_checkpoints")

    # --- Reproducibility ---
    RANDOM_SEED = 42

    # --- Data parameters ---
    # Top-K genes selection
    TOP_K_FEATURES = 20
    # Lambda for tradeoff between Data Variance and Centrality
    LAMBDA_VAL = 0.5

    # --- Model parameters (Llama-3) ---
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    MAX_SEQ_LENGTH = 2048

    # --- LoRA Base Hyperparameters ---
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]

    # --- Base Training parameters ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    GRADIENT_ACCUMULATION = 4
    
    # --- Optuna Search Space ---
    OPTUNA_TRIALS = 10
    SEARCH_SPACE = {
        "learning_rate": (1e-5, 5e-4),
        "lora_r": [8, 16, 32],
        "lora_alpha": [16, 32, 64],
        "lora_dropout": (0.01, 0.1),
        "batch_size": [2, 4, 8]
    }


cfg = Config()