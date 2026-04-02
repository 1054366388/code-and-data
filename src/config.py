import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Config:
    """
    Configuration file for Prot-STAR Framework.
    Defines paths, essential hyperparameters, and Optuna search spaces.
    All paths are relative to the project root (the directory containing src/).
    """

    # --- Paths (derived from project root, no hardcoded absolute paths) ---
    PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)
    DATA_ROOT: str = ""
    KG_PATH: str = ""
    OUTPUT_DIR: str = ""

    # --- Cohort Definitions ---
    COHORT_LABELS: Dict[str, List[str]] = field(default_factory=lambda: {
        "COADREAD": ["COAD", "READ"],
        "GBMLGG": ["GBM", "LGG"],
        "KIPAN": ["KICH", "KIRC", "KIRP"],
        "LUAD": ["LUAD_tumor", "LUAD_normal"],
        "THCA": ["THCA_tumor", "THCA_normal"],
        "UCEC": ["CN", "MSI", "POLE", "Serous"],
    })
    ALL_COHORTS: List[str] = field(default_factory=lambda: [
        "COADREAD", "GBMLGG", "KIPAN", "LUAD", "THCA", "UCEC"
    ])

    # --- Reproducibility ---
    RANDOM_SEED: int = 42

    # --- Data parameters ---
    TOP_K_FEATURES: int = 20
    LAMBDA_VAL: float = 0.5
    NAN_THRESHOLD: float = 0.5  # Drop rows with more than this fraction of NaN

    # --- Model parameters (Llama-3) ---
    MODEL_ID: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    MAX_SEQ_LENGTH: int = 2048

    # --- LoRA Base Hyperparameters ---
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    TARGET_MODULES: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- Base Training parameters ---
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 4
    NUM_EPOCHS: int = 5
    GRADIENT_ACCUMULATION: int = 4

    # --- Optuna Search Space ---
    OPTUNA_TRIALS: int = 10
    SEARCH_SPACE: Dict = field(default_factory=lambda: {
        "learning_rate": (1e-5, 5e-4),
        "lora_r": [8, 16, 32],
        "lora_alpha": [16, 32, 64],
        "lora_dropout": (0.01, 0.1),
        "batch_size": [2, 4, 8],
    })

    def __post_init__(self):
        """Derive dependent paths after initialisation."""
        self.DATA_ROOT = os.path.join(self.PROJECT_ROOT, "Expression Data")
        self.KG_PATH = os.path.join(self.PROJECT_ROOT, "S.xlsx")
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "model_checkpoints")


cfg = Config()
