import pandas as pd
import os
from .config import cfg


class DataLoader:
    """
    Handles ingestion of heterogeneous OMICS data formats (.xlsx).
    """

    def load_cohort(self, cohort_name):
        """
        Loads aligned expression data and labels.
        """
        cohort_dir = os.path.join(cfg.DATA_ROOT, cohort_name)

        # 1. Load Feature Names
        # File: feature_name.xlsx (Header in row 1, data starts row 2)
        feat_path = os.path.join(cohort_dir, "feature_name.xlsx")
        feat_df = pd.read_excel(feat_path, header=None, skiprows=1)
        # Ensure we get a flat list of strings
        feature_names = feat_df.iloc[:, 0].astype(str).tolist()

        # 2. Load Expression Data
        # File: samples_protein.xlsx (Rows=Proteins, Cols=Samples)
        data_path = os.path.join(cohort_dir, "samples_protein.xlsx")
        # header=None is critical as per user description
        raw_data = pd.read_excel(data_path, header=None)

        # Transpose: Convert [Proteins x Samples] -> [Samples x Proteins]
        expression_df = raw_data.T

        # 3. Load Labels
        label_path = os.path.join(cohort_dir, "lables.xlsx")
        labels_df = pd.read_excel(label_path, header=None)
        labels_df.columns = ["target"]

        # --- Alignment Logic ---
        # Robustly assign columns based on dimension matching
        n_samples, n_feats = expression_df.shape

        if n_feats == len(feature_names):
            expression_df.columns = feature_names
        else:
            print(f"[Warning] Feature count mismatch in {cohort_name}. "
                  f"Data: {n_feats}, Names: {len(feature_names)}. Generating IDs.")
            expression_df.columns = [f"Protein_{i}" for i in range(n_feats)]

        # Ensure label length matches sample length
        min_len = min(len(expression_df), len(labels_df))
        return expression_df.iloc[:min_len], labels_df.iloc[:min_len]

    def load_knowledge_graph(self):
        """Loads the PTM adjacency matrix."""
        if os.path.exists(cfg.KG_PATH):
            # Assumes S.xlsx is an adjacency matrix
            return pd.read_excel(cfg.KG_PATH, index_col=0, header=None)
        else:
            print(f"[Warning] Knowledge Graph not found at {cfg.KG_PATH}")
            return None