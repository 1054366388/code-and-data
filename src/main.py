"""
Prot-STAR main entry point.

Runs the full pipeline for every cohort defined in config:
  1. Data loading and NaN cleaning
  2. Structure-grounded pruning (feature selection)
  3. Adaptive semantic quantisation (fit on train, transform test)
  4. Prompt construction
  5. LoRA fine-tuning with Optuna HPO
  6. Inference on held-out test set
  7. Evaluation (accuracy, macro-F1, classification report)
"""

import os
import random

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

# CRITICAL: Fix for OMP duplicate-library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .config import cfg
from .core_modules import ProtStarCore
from .data_loader import DataLoader
from .model_engine import LLMEngine


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #
def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def objective(trial, engine: LLMEngine, prompts: list, cohort_name: str) -> float:
    """Optuna objective: fine-tune one trial and return final loss."""
    lr = trial.suggest_float("learning_rate", *cfg.SEARCH_SPACE["learning_rate"], log=True)
    lora_r = trial.suggest_categorical("lora_r", cfg.SEARCH_SPACE["lora_r"])
    lora_alpha = trial.suggest_categorical("lora_alpha", cfg.SEARCH_SPACE["lora_alpha"])
    lora_dropout = trial.suggest_float("lora_dropout", *cfg.SEARCH_SPACE["lora_dropout"])
    batch_size = trial.suggest_categorical("batch_size", cfg.SEARCH_SPACE["batch_size"])

    print(
        f"\n[Optuna] Trial {trial.number}: "
        f"LR={lr:.2e}, R={lora_r}, Alpha={lora_alpha}, "
        f"Dropout={lora_dropout:.2f}, BS={batch_size}"
    )

    try:
        final_loss = engine.run_training(
            prompt_list=prompts,
            learning_rate=lr,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            trial_number=trial.number,
        )
        return final_loss
    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
        print(f"[Optuna] Trial {trial.number} failed: {exc}")
        torch.cuda.empty_cache()
        return 999.0


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 60)
    print(" Prot-STAR: LLM Framework for Cancer Subtype Classification")
    print("=" * 60)

    # 0. Seed & modules ------------------------------------------------
    set_seed(cfg.RANDOM_SEED)
    print(f"[System] Seed set to {cfg.RANDOM_SEED}")

    loader = DataLoader()
    kg_matrix = loader.load_knowledge_graph()

    # ================================================================== #
    # Loop over ALL cohorts
    # ================================================================== #
    for cohort in cfg.ALL_COHORTS:
        class_names = cfg.COHORT_LABELS[cohort]
        print(f"\n{'=' * 60}")
        print(f"[Pipeline] Cohort: {cohort}  |  Classes: {class_names}")
        print("=" * 60)

        # 1. Load & split ------------------------------------------------
        try:
            X_train, X_test, y_train, y_test = loader.load_cohort(cohort)
        except FileNotFoundError as exc:
            print(f"[Skip] {cohort}: {exc}")
            continue

        print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

        # 2. Structure-grounded pruning (train only -> no leakage) --------
        core = ProtStarCore(kg_matrix)
        top_genes = core.structure_grounded_pruning(X_train)
        X_train_sel = X_train[top_genes]
        X_test_sel = X_test[top_genes]
        print(f"  Top-{cfg.TOP_K_FEATURES} features: {top_genes[:5]} ...")

        # 3. Adaptive Semantic Quantisation (fit on train, apply to test) -
        X_train_sem = core.fit_transform(X_train_sel)
        X_test_sem = core.transform(X_test_sel)

        # 4. Prompt construction ------------------------------------------
        train_prompts = []
        for idx, row in X_train_sem.iterrows():
            try:
                label_val = int(y_train.loc[idx, "target"])
            except (KeyError, TypeError):
                label_val = int(y_train.iloc[0, 0])
            prompt = core.construct_prompt(row, cohort, class_names, label=label_val)
            train_prompts.append(prompt)

        # Save processed prompts for reproducibility
        save_path = os.path.join(cfg.OUTPUT_DIR, f"{cohort}_train_prompts.csv")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        pd.DataFrame({"prompt": train_prompts}).to_csv(save_path, index=False)
        print(f"  Prompts saved -> {save_path}")

        # 5. HPO + Training -----------------------------------------------
        print(f"\n[Phase 2] Hyperparameter Optimisation for {cohort}")
        try:
            engine = LLMEngine()

            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(trial, engine, train_prompts, cohort),
                n_trials=cfg.OPTUNA_TRIALS,
            )

            best = study.best_trial
            print(f"\n  Best Trial: {best.number}  |  Loss: {study.best_value:.4f}")
            for k, v in best.params.items():
                print(f"    {k}: {v}")

            # Determine the adapter path of the best trial
            best_adapter = os.path.join(cfg.OUTPUT_DIR, f"trial_{best.number}")

            # 6. Inference on test set ----------------------------------------
            print(f"\n[Phase 3] Inference on {cohort} test set ({len(X_test)} samples)")
            test_prompts = []
            for idx, row in X_test_sem.iterrows():
                prompt = core.construct_prompt(row, cohort, class_names, label=None)
                test_prompts.append(prompt)

            pred_labels = engine.run_inference(
                prompt_list=test_prompts,
                class_names=class_names,
                adapter_path=best_adapter,
                lora_r=best.params.get("lora_r", cfg.LORA_R),
                lora_alpha=best.params.get("lora_alpha", cfg.LORA_ALPHA),
                lora_dropout=best.params.get("lora_dropout", cfg.LORA_DROPOUT),
            )

            # 7. Evaluation ---------------------------------------------------
            # Map integer ground-truth labels to class name strings
            y_true_names = []
            for idx in y_test.index:
                lbl = int(y_test.loc[idx, "target"])
                y_true_names.append(class_names[lbl] if lbl < len(class_names) else str(lbl))

            acc = accuracy_score(y_true_names, pred_labels)
            f1 = f1_score(y_true_names, pred_labels, average="macro", zero_division=0)

            print(f"\n  === {cohort} Results ===")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  Macro-F1 : {f1:.4f}")
            print(
                classification_report(
                    y_true_names,
                    pred_labels,
                    labels=class_names,
                    zero_division=0,
                )
            )

            # Save predictions
            results_df = pd.DataFrame(
                {"y_true": y_true_names, "y_pred": pred_labels}
            )
            results_path = os.path.join(cfg.OUTPUT_DIR, f"{cohort}_test_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"  Results saved -> {results_path}")

        except (RuntimeError, ValueError, OSError) as exc:
            print(f"\n[Error] {cohort} pipeline failed: {exc}")
            print("  Troubleshooting:")
            print("  1. Run 'huggingface-cli login' and accept the model licence.")
            print("  2. Ensure a CUDA GPU is available.")
            print("  3. Check disk space for model checkpoints.")
            continue

    print("\n" + "=" * 60)
    print("[Done] Prot-STAR pipeline finished for all cohorts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
