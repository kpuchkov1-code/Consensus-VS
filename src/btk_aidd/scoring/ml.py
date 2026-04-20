"""Machine-learning rescorer.

Trains a Random Forest classifier on Morgan fingerprints to predict the
probability that a molecule is a BTK active. Predictions on the held-out test
set are used as a per-ligand ML score (higher = more likely active, which we
re-sign so that "more negative = better" lines up with docking affinity).

The ML rescorer is deliberately a *classifier*, not a regressor: ChEMBL
activity labels are more reliable than absolute pChEMBL for a heterogeneous
target like BTK with many assay protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from btk_aidd.config import MLScoringConfig
from btk_aidd.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MLSplit:
    """Train / test split and predictions from :class:`MLRescorer`."""

    train_names: list[str]
    test_names: list[str]
    test_labels: np.ndarray
    test_probabilities: np.ndarray
    test_scores: np.ndarray  # negated probability so lower = better, matching docking


class MLRescorer:
    """Morgan-fingerprint Random Forest binary classifier."""

    def __init__(self, config: MLScoringConfig) -> None:
        self._cfg = config

    def fit_predict(
        self,
        names: list[str],
        smiles: list[str],
        labels: np.ndarray,
    ) -> MLSplit:
        """Train on a random subset and predict on the held-out test subset.

        Args:
            names: Stable identifiers (one per SMILES).
            smiles: SMILES strings, same length as ``names``.
            labels: Binary labels (1 = active, 0 = decoy), same length.

        Returns:
            An :class:`MLSplit` with test names, labels, probabilities, and
            sign-flipped scores (lower = better, matching Vina convention).
        """
        if not (len(names) == len(smiles) == len(labels)):
            raise ValueError("names, smiles, and labels must be the same length")
        if len(names) < 10:
            raise ValueError(
                f"Too few compounds for a train/test split: got {len(names)}, need >= 10"
            )

        fps = self._fingerprints(smiles)
        idx_train, idx_test = train_test_split(
            np.arange(len(names)),
            test_size=self._cfg.test_size,
            random_state=self._cfg.random_seed,
            stratify=labels,
        )

        clf = RandomForestClassifier(
            n_estimators=self._cfg.n_estimators,
            random_state=self._cfg.random_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(fps[idx_train], labels[idx_train])

        probas = clf.predict_proba(fps[idx_test])
        if probas.shape[1] < 2:  # pragma: no cover - degenerate single-class split
            p_active = np.zeros(len(idx_test), dtype=float)
        else:
            p_active = probas[:, list(clf.classes_).index(1)]

        logger.info(
            "MLRescorer trained on %d, predicted on %d (positive rate test=%.2f)",
            len(idx_train),
            len(idx_test),
            float(labels[idx_test].mean()),
        )

        return MLSplit(
            train_names=[names[i] for i in idx_train],
            test_names=[names[i] for i in idx_test],
            test_labels=labels[idx_test].astype(int),
            test_probabilities=p_active,
            test_scores=-p_active,  # negate so "lower == better", matching docking
        )

    # ------------------------------------------------------------------ helpers

    def _fingerprints(self, smiles: list[str]) -> np.ndarray:
        radius = self._cfg.morgan_radius
        n_bits = self._cfg.morgan_n_bits
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning("Skipping unparseable SMILES at index %d", i)
                continue
            fp = generator.GetFingerprint(mol)
            onbits = np.zeros(n_bits, dtype=np.uint8)
            for bit in fp.GetOnBits():
                onbits[bit] = 1
            matrix[i] = onbits
        return matrix
