import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration file for Prot-STAR Framework.
    References: Section 4.1 (Experimental Setup) & Section 3.3.
    """
    # --- Paths ---
    BASE_DIR = r"C:\Users\Jie\Desktop\Prot-STAR"
    DATA_ROOT = os.path.join(BASE_DIR, "Expression Data")
    KG_PATH = os.path.join(BASE_DIR, "S.xlsx")
    OUTPUT_DIR = os.path.join(BASE_DIR, "model_checkpoints")

    # --- Data parameters ---
    # Top-K genes selection via Eq. 4
    TOP_K_FEATURES = 20
    # Lambda for tradeoff between Data Variance and Centrality (Eq. 4)
    LAMBDA_VAL = 0.5

    # --- Model parameters (Llama-3) ---
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    MAX_SEQ_LENGTH = 2048

    # --- LoRA Hyperparameters (Eq. 6) ---
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]

    # --- Training parameters ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4  # Adjusted for typical GPU memory; Paper uses 16
    NUM_EPOCHS = 5
    GRADIENT_ACCUMULATION = 4


cfg = Config()