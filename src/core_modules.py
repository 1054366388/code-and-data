import pandas as pd
import numpy as np
from .config import cfg


class ProtStarCore:
    """
    Core operations for the Prot-STAR framework, handling feature pruning and expression quantization.
    """

    def __init__(self, kg_matrix=None):
        self.kg_matrix = kg_matrix
        # Semantic mapping thresholds
        self.bin_map = {
            (0.00, 0.10): "Very Low",
            (0.10, 0.35): "Low",
            (0.35, 0.65): "Medium",
            (0.65, 0.90): "High",
            (0.90, 1.01): "Very High"
        }

    def structure_grounded_pruning(self, df):
        """
        Selects Top-K features based on a balanced scoring of numerical variance and graph centrality.
        Score = lambda * Norm(Std) + (1-lambda) * Norm(Centrality)
        """
        # 1. Data Deviation (Sigma)
        std_dev = df.std()
        max_std = std_dev.max() if std_dev.max() > 0 else 1.0
        norm_std = std_dev / max_std

        scores = {}

        # 2. Knowledge Graph Centrality
        if self.kg_matrix is not None:
            # Compute degree centrality
            centrality = self.kg_matrix.sum(axis=1)
            max_cen = centrality.max() if not centrality.empty else 1.0

            for gene in df.columns:
                c_val = centrality.get(gene, 0) / max_cen
                d_val = norm_std.get(gene, 0)
                # Combine scores
                scores[gene] = (cfg.LAMBDA_VAL * d_val) + ((1 - cfg.LAMBDA_VAL) * c_val)
        else:
            # Fallback to pure variance if KG is missing
            scores = norm_std.to_dict()

        # Select Top-K
        ranked_features = pd.Series(scores).sort_values(ascending=False)
        return ranked_features.head(cfg.TOP_K_FEATURES).index.tolist()

    def adaptive_semantic_quantization(self, df):
        """
        Converts continuous float expressions to semantic categorical tokens using empirical distributions.
        """
        quantized_df = pd.DataFrame(index=df.index, columns=df.columns)

        for col in df.columns:
            # Rank values to percentiles
            ranks = df[col].rank(pct=True)

            # Mapping logic
            def get_token(r):
                for (low, high), token in self.bin_map.items():
                    if low < r <= high:
                        return token
                return "Very High"

            quantized_df[col] = ranks.apply(get_token)

        return quantized_df

    def construct_prompt(self, feature_series, label=None):
        """
        Constructs a structured text instruction prompt for the Language Model based on the quantized profile.
        """
        # Serialize: "EGFR: High, PTEN: Low"
        profile = ", ".join([f"{k}: {v}" for k, v in feature_series.items()])

        # Instructional Prompt Template
        prompt = (
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n"
            f"You are an expert oncologist. Analyze the proteomic profile to diagnose the cancer subtype (COAD vs READ).\n"
            f"Patient Profile: {profile}\n\n"
            f"### Response:\n"
        )

        # During training, we append the Ground Truth diagnosis
        if label is not None:
            diagnosis_str = "COAD" if label == 0 else "READ"
            reasoning = (
                "1. [Observation]: The profile shows specific regulation patterns.\n"
                "2. [Knowledge]: These markers align with known pathway deviations.\n"
                f"3. [Diagnosis]: The subtype is {diagnosis_str}."
            )
            prompt += reasoning

        return prompt