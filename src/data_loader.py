import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import cfg


class DataLoader:
    """
    Handles ingestion and processing of heterogeneous OMICS data formats (.xlsx).
    Includes robust NaN handling and flexible filename resolution.
    """

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_cohort(
        self, cohort_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads aligned expression data and labels for *cohort_name*,
        cleans NaN values, and returns stratified train/test splits.
        """
        cohort_dir = os.path.join(cfg.DATA_ROOT, cohort_name)

        # 1. Load feature names -------------------------------------------
        feat_path = os.path.join(cohort_dir, "feature_name.xlsx")
        feat_df = pd.read_excel(feat_path, header=None, skiprows=1)
        feature_names = feat_df.iloc[:, 0].astype(str).tolist()

        # 2. Load expression matrix (Proteins x Samples) -> transpose -----
        data_path = os.path.join(cohort_dir, "samples_protein.xlsx")
        raw_data = pd.read_excel(data_path, header=None)
        expression_df = raw_data.T  # now Samples x Proteins

        # 3. Load labels (handle both "labels.xlsx" and "lables.xlsx") -----
        labels_df = self._load_labels(cohort_dir)

        # 4. Assign column names ------------------------------------------
        n_samples, n_feats = expression_df.shape
        if n_feats == len(feature_names):
            expression_df.columns = feature_names
        else:
            print(
                f"[Warning] Feature count mismatch in {cohort_name}. "
                f"Data: {n_feats}, Names: {len(feature_names)}. Generating IDs."
            )
            expression_df.columns = [f"Protein_{i}" for i in range(n_feats)]

        # 5. Align sample counts between expression and labels -------------
        min_len = min(len(expression_df), len(labels_df))
        expression_df = expression_df.iloc[:min_len].reset_index(drop=True)
        labels_df = labels_df.iloc[:min_len].reset_index(drop=True)

        # 6. Handle NaN values --------------------------------------------
        expression_df, labels_df = self._clean_nan(expression_df, labels_df)

        # 7. Stratified train/test split -----------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            expression_df,
            labels_df,
            test_size=0.2,
            random_state=cfg.RANDOM_SEED,
            stratify=labels_df,
        )

        return X_train, X_test, y_train, y_test

    def load_knowledge_graph(self) -> Optional[pd.DataFrame]:
        """Loads the protein–protein interaction adjacency matrix from *S.xlsx*."""
        if os.path.exists(cfg.KG_PATH):
            # header=0 so the first row is used as column headers and the
            # gene names in the index are preserved properly.
            return pd.read_excel(cfg.KG_PATH, index_col=0, header=0)
        else:
            print(f"[Warning] Knowledge Graph not found at {cfg.KG_PATH}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_labels(cohort_dir: str) -> pd.DataFrame:
        """Try both possible label filenames (correct *and* typo)."""
        candidates = ["labels.xlsx", "lables.xlsx"]
        for fname in candidates:
            path = os.path.join(cohort_dir, fname)
            if os.path.exists(path):
                labels_df = pd.read_excel(path, header=None)
                labels_df.columns = ["target"]
                return labels_df
        raise FileNotFoundError(
            f"No label file found in {cohort_dir}. "
            f"Tried: {candidates}"
        )

    @staticmethod
    def _clean_nan(
        expression_df: pd.DataFrame, labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        1. Drop rows where more than NAN_THRESHOLD fraction of features are NaN.
        2. Impute remaining NaN values with the column (feature) median.
        """
        # Fraction of NaN per row
        nan_frac = expression_df.isna().mean(axis=1)
        keep_mask = nan_frac <= cfg.NAN_THRESHOLD

        n_dropped = (~keep_mask).sum()
        if n_dropped > 0:
            print(f"  [NaN] Dropped {n_dropped} samples with >{cfg.NAN_THRESHOLD*100:.0f}% missing values.")

        expression_df = expression_df.loc[keep_mask].reset_index(drop=True)
        labels_df = labels_df.loc[keep_mask].reset_index(drop=True)

        # Column-median imputation for remaining NaN
        if expression_df.isna().any().any():
            col_medians = expression_df.median()
            expression_df = expression_df.fillna(col_medians)
            print("  [NaN] Imputed remaining missing values with column medians.")

        return expression_df, labels_df
