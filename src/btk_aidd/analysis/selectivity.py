"""Kinase-selectivity scoring.

BTK sits in a family of structurally similar kinases (TEC family) whose ATP
pockets share high sequence and shape identity. A "BTK inhibitor" that also
hits **EGFR**, **ITK**, or **JAK2** at clinical concentrations is a dirty
drug; every major approved BTK inhibitor has a published selectivity panel.

This module closes the selectivity gap by training an independent Morgan-
fingerprint Random Forest classifier per off-target kinase and combining
their predictions into a single selectivity index:

``selectivity_index = P(active at BTK) - max_offtarget P(active at X)``

Positive values mean selective-for-BTK; negative values mean the ligand is
predicted to hit an off-target more strongly than BTK itself.

The off-target panel defaults to five TEC-family and related kinases that
are the routine selectivity cross-checks for BTK medchem:

* **EGFR** (ErbB1) — Cys797 is the covalent-inhibitor analogue of Cys481
* **ITK** — closest TEC-family relative
* **TEC** — founder of the TEC family
* **BMX** (ETK) — TEC family
* **JAK2** — broader kinome; tested because ibrutinib hits it weakly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

from btk_aidd.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OFF_TARGETS: tuple[str, ...] = ("EGFR", "ITK", "TEC", "BMX", "JAK2")


@dataclass(frozen=True)
class OffTargetModel:
    """Trained RF model + metadata for one off-target kinase."""

    kinase: str
    classifier: RandomForestClassifier
    n_train_actives: int
    n_train_decoys: int


@dataclass(frozen=True)
class SelectivityReport:
    """Per-ligand selectivity assessment."""

    name: str
    p_btk: float
    off_target_probabilities: dict[str, float]
    max_off_target: str
    max_off_target_probability: float
    selectivity_index: float  # p_btk - max_off_target_prob

    @property
    def is_selective(self) -> bool:
        """True when BTK predicted probability exceeds every off-target by >= 0.1."""
        return self.selectivity_index >= 0.1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_off_target_models(
    panels: dict[str, pd.DataFrame],
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    n_estimators: int = 200,
    random_seed: int = 42,
) -> dict[str, OffTargetModel]:
    """Train a classifier per off-target kinase.

    Args:
        panels: Mapping from kinase name -> DataFrame with columns
            ``canonical_smiles`` and ``label`` (1 = active, 0 = decoy).
        morgan_radius: Morgan FP radius.
        morgan_n_bits: Morgan FP bit length.
        n_estimators: Number of trees per Random Forest.
        random_seed: Seed for reproducibility.

    Returns:
        Mapping kinase -> :class:`OffTargetModel`.
    """
    models: dict[str, OffTargetModel] = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=morgan_radius, fpSize=morgan_n_bits
    )
    for kinase, df in panels.items():
        if {"canonical_smiles", "label"} - set(df.columns):
            raise ValueError(f"Panel for {kinase} missing required columns")
        if df.empty or df["label"].nunique() < 2:
            logger.warning("Panel for %s has <2 classes, skipping", kinase)
            continue
        fps = _fingerprints(df["canonical_smiles"].tolist(), generator, morgan_n_bits)
        labels = df["label"].to_numpy().astype(int)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(fps, labels)
        models[kinase] = OffTargetModel(
            kinase=kinase,
            classifier=clf,
            n_train_actives=int(labels.sum()),
            n_train_decoys=int(len(labels) - labels.sum()),
        )
        logger.info(
            "Trained off-target model for %s (%d actives, %d decoys)",
            kinase,
            models[kinase].n_train_actives,
            models[kinase].n_train_decoys,
        )
    return models


def score_selectivity(
    names: list[str],
    smiles: list[str],
    p_btk: np.ndarray,
    off_target_models: dict[str, OffTargetModel],
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
) -> list[SelectivityReport]:
    """Build :class:`SelectivityReport`s for a batch of ligands.

    Args:
        names: Ligand identifiers.
        smiles: SMILES strings (same order as ``names``).
        p_btk: BTK-activity probabilities (from the main ML rescorer).
        off_target_models: Output of :func:`train_off_target_models`.
        morgan_radius: Morgan FP radius (must match the training setting).
        morgan_n_bits: Morgan FP bit length.

    Returns:
        A list of :class:`SelectivityReport` in the same order as the input.
    """
    if not (len(names) == len(smiles) == len(p_btk)):
        raise ValueError("names, smiles, and p_btk must be the same length")
    if not off_target_models:
        raise ValueError("off_target_models is empty — cannot compute selectivity")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=morgan_radius, fpSize=morgan_n_bits
    )
    fps = _fingerprints(smiles, generator, morgan_n_bits)

    reports: list[SelectivityReport] = []
    for i, (name, p) in enumerate(zip(names, p_btk, strict=True)):
        per_kinase: dict[str, float] = {}
        for kinase, model in off_target_models.items():
            probas = model.classifier.predict_proba(fps[i : i + 1])
            classes = list(model.classifier.classes_)
            p_off = float(probas[0, classes.index(1)]) if 1 in classes else 0.0
            per_kinase[kinase] = p_off
        worst_kinase, worst_p = max(per_kinase.items(), key=lambda kv: kv[1])
        reports.append(
            SelectivityReport(
                name=name,
                p_btk=float(p),
                off_target_probabilities=per_kinase,
                max_off_target=worst_kinase,
                max_off_target_probability=worst_p,
                selectivity_index=float(p) - worst_p,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _fingerprints(
    smiles: list[str],
    generator: rdFingerprintGenerator.FingerprintGenerator64,
    n_bits: int,
) -> np.ndarray:
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Skipping unparseable SMILES at index %d", i)
            continue
        fp = generator.GetFingerprint(mol)
        for bit in fp.GetOnBits():
            matrix[i, bit] = 1
    return matrix
