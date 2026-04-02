from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from .config import cfg


class ProtStarCore:
    """
    Core scientific modules for the Prot-STAR framework:
      - Structure-Grounded Pruning (feature selection)
      - Adaptive Semantic Quantisation (ASQ)
      - Knowledge-grounded prompt construction
    """

    # Semantic bins: each tuple is [low, high) except the last which is [low, high].
    # Boundary 0.0 maps to "Very Low" (>= 0.0).
    BIN_EDGES: List[Tuple[float, float, str]] = [
        (0.00, 0.10, "Very Low"),
        (0.10, 0.35, "Low"),
        (0.35, 0.65, "Medium"),
        (0.65, 0.90, "High"),
        (0.90, 1.01, "Very High"),
    ]

    def __init__(self, kg_matrix: Optional[pd.DataFrame] = None):
        self.kg_matrix = kg_matrix
        # Per-column ECDF functions fitted on training data (set by fit_transform)
        self._ecdfs: Dict[str, ECDF] = {}

    # ------------------------------------------------------------------ #
    # 1. Structure-Grounded Pruning
    # ------------------------------------------------------------------ #
    def structure_grounded_pruning(self, df: pd.DataFrame) -> List[str]:
        """
        Selects Top-K features using a weighted combination of
        numerical variance (sigma) and knowledge-graph degree centrality.

        Score_g = lambda * Norm(Std_g) + (1-lambda) * Norm(Centrality_g)
        """
        std_dev = df.std()
        max_std = std_dev.max() if std_dev.max() > 0 else 1.0
        norm_std = std_dev / max_std

        scores: Dict[str, float] = {}

        if self.kg_matrix is not None:
            centrality = self.kg_matrix.sum(axis=1)
            max_cen = centrality.max() if centrality.max() > 0 else 1.0

            for gene in df.columns:
                c_val = centrality.get(gene, 0) / max_cen
                d_val = norm_std.get(gene, 0)
                scores[gene] = cfg.LAMBDA_VAL * d_val + (1 - cfg.LAMBDA_VAL) * c_val
        else:
            scores = norm_std.to_dict()

        ranked = pd.Series(scores).sort_values(ascending=False)
        return ranked.head(cfg.TOP_K_FEATURES).index.tolist()

    # ------------------------------------------------------------------ #
    # 2. Adaptive Semantic Quantisation  (ASQ)
    # ------------------------------------------------------------------ #
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        **Training path**: fit one ECDF per column on *df* (training data)
        and return the semantically quantised DataFrame.

        This avoids data leakage: the ECDF is determined solely from
        the training split and later applied unchanged to the test split.
        """
        self._ecdfs = {}
        quantised = pd.DataFrame(index=df.index, columns=df.columns)

        for col in df.columns:
            values = df[col].dropna().values.astype(float)
            ecdf = ECDF(values)
            self._ecdfs[col] = ecdf
            quantised[col] = df[col].apply(lambda v: self._rank_to_token(ecdf(v)))

        return quantised

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        **Inference path**: apply ECDFs fitted on training data to *df*
        (test data).  Must call ``fit_transform`` first.
        """
        if not self._ecdfs:
            raise RuntimeError(
                "No fitted ECDFs found. Call fit_transform() on training data first."
            )

        quantised = pd.DataFrame(index=df.index, columns=df.columns)
        for col in df.columns:
            ecdf = self._ecdfs.get(col)
            if ecdf is None:
                raise KeyError(f"Column '{col}' was not present during fit_transform.")
            quantised[col] = df[col].apply(lambda v: self._rank_to_token(ecdf(v)))

        return quantised

    @staticmethod
    def _rank_to_token(rank: float) -> str:
        """
        Map a [0, 1] percentile rank to a semantic token.

        Boundary condition: rank == 0.0 maps to "Very Low" (uses >= for lower bound).
        """
        for low, high, token in ProtStarCore.BIN_EDGES:
            if low <= rank < high:
                return token
        # Fallback for rank == 1.0 (upper edge of the last bin)
        return "Very High"

    # ------------------------------------------------------------------ #
    # 3. Knowledge-Grounded Prompt Construction
    # ------------------------------------------------------------------ #
    def construct_prompt(
        self,
        feature_series: pd.Series,
        cohort_name: str,
        class_names: List[str],
        label: Optional[int] = None,
    ) -> str:
        """
        Builds a structured instruction-tuning prompt for the LLM.

        Parameters
        ----------
        feature_series : pd.Series
            One sample's semantically quantised protein profile.
        cohort_name : str
            E.g. "COADREAD", "KIPAN", etc.
        class_names : list[str]
            Ordered list of class labels for the cohort (index matches label int).
        label : int or None
            Ground-truth label index (training) or None (inference).
        """
        profile_str = ", ".join(f"{k}: {v}" for k, v in feature_series.items())
        class_options = " vs ".join(class_names)

        # Identify the most extreme proteins for reasoning context
        high_proteins = [k for k, v in feature_series.items() if v in ("High", "Very High")]
        low_proteins = [k for k, v in feature_series.items() if v in ("Low", "Very Low")]

        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"You are an expert oncologist specializing in the {cohort_name} cancer cohort. "
            f"Classify the patient's cancer subtype ({class_options}) "
            "based on the following semantically quantized proteomic profile.\n"
            f"Patient Profile: {profile_str}\n\n"
            "### Response:\n"
        )

        if label is not None:
            diagnosis = class_names[label] if label < len(class_names) else str(label)

            # Build knowledge-grounded reasoning referencing actual differential proteins
            obs_parts = []
            if high_proteins:
                obs_parts.append(
                    f"elevated expression of {', '.join(high_proteins[:5])}"
                )
            if low_proteins:
                obs_parts.append(
                    f"reduced expression of {', '.join(low_proteins[:5])}"
                )
            observation = (
                "The profile shows " + (" and ".join(obs_parts) if obs_parts else "mixed expression levels across the panel")
                + "."
            )

            reasoning = (
                f"1. [Observation]: {observation}\n"
                f"2. [Knowledge]: In the {cohort_name} cohort, this pattern of differential "
                f"protein expression is characteristic of the {diagnosis} subtype, reflecting "
                "known pathway-level alterations.\n"
                f"3. [Diagnosis]: {diagnosis}"
            )
            prompt += reasoning

        return prompt
