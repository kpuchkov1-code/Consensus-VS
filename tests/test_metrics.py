"""Tests for enrichment metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from btk_aidd.metrics.enrichment import enrichment_factor, roc_auc, scorer_report


def test_roc_auc_perfect_separation() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([-9.0, -8.5, -8.0, -5.0, -4.5, -4.0])  # lower = better
    assert roc_auc(labels, scores) == 1.0


def test_roc_auc_random() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.normal(size=200)
    auc = roc_auc(labels, scores)
    assert 0.4 < auc < 0.6


def test_roc_auc_single_class_returns_nan() -> None:
    labels = np.ones(10, dtype=int)
    scores = np.linspace(-9, -1, 10)
    assert math.isnan(roc_auc(labels, scores))


def test_enrichment_factor_perfect() -> None:
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([-9, -8, -7, -6, -5, -4, -3, -2, -1, 0], dtype=float)
    # top 40% = 4 compounds; all actives. overall rate = 4/10 = 0.4; top rate = 4/4 = 1.0
    ef = enrichment_factor(labels, scores, 0.4)
    assert math.isclose(ef, 1.0 / 0.4, rel_tol=1e-6)


def test_enrichment_factor_random() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=1000)
    scores = rng.normal(size=1000)
    ef = enrichment_factor(labels, scores, 0.1)
    assert 0.5 < ef < 1.5


def test_enrichment_factor_rejects_bad_fraction() -> None:
    labels = np.ones(10, dtype=int)
    scores = np.zeros(10)
    with pytest.raises(ValueError):
        enrichment_factor(labels, scores, 0.0)
    with pytest.raises(ValueError):
        enrichment_factor(labels, scores, 1.5)


def test_scorer_report_structure() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([-9, -8, -4, -7, -6, -5, -3, -2, -1, 0], dtype=float)
    report = scorer_report("test", labels, scores, [0.1, 0.5])
    assert report.name == "test"
    assert 0.0 <= report.roc_auc <= 1.0
    assert set(report.enrichment.keys()) == {0.1, 0.5}
    assert report.n_actives == 3
    assert report.n_decoys == 7
    assert report.fpr.shape == report.tpr.shape
