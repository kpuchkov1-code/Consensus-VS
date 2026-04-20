"""Strongly-typed configuration for the BTK-AIDD pipeline.

The pipeline is driven entirely by a YAML config (see ``config/default.yaml``).
This module loads the YAML, validates it with Pydantic, and exposes a frozen
``Config`` object that every other module consumes. Keeping the config in one
typed place avoids string-key errors scattered through the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TargetConfig(BaseModel):
    """Target protein identifiers."""

    model_config = ConfigDict(frozen=True)

    name: str
    chembl_id: str
    pdb_id: str
    reference_ligand_resname: str


class PropertyWindows(BaseModel):
    """Absolute property deltas used to match decoys to actives (DUD-E style)."""

    model_config = ConfigDict(frozen=True)

    mw: float = 40.0
    logp: float = 0.75
    hba: int = 2
    hbd: int = 2
    rotatable: int = 2
    charge: int = 0


class ActivesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_pchembl: float = 7.0
    activity_types: list[str] = Field(default_factory=lambda: ["IC50", "Ki", "Kd"])
    max_actives: int = 500


class DecoysConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int = 2000
    similarity_cutoff: float = 0.35
    property_windows: PropertyWindows = Field(default_factory=PropertyWindows)


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    actives: ActivesConfig = Field(default_factory=ActivesConfig)
    decoys: DecoysConfig = Field(default_factory=DecoysConfig)
    random_seed: int = 42


class LigandPrepConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_conformers: int = 1
    mmff_variant: str = "MMFF94"
    max_mmff_iters: int = 200
    embed_seed: int = 42

    @field_validator("mmff_variant")
    @classmethod
    def _validate_mmff(cls, v: str) -> str:
        if v not in {"MMFF94", "MMFF94s"}:
            raise ValueError(f"mmff_variant must be MMFF94 or MMFF94s, got {v!r}")
        return v


class DockingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str = "cached"
    exhaustiveness: int = 16
    num_poses: int = 5
    box_padding: float = 8.0
    cached_scores_csv: str = "data/fixtures/cached_docking_scores.csv"

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, v: str) -> str:
        allowed = {"cached", "vina"}
        if v not in allowed:
            raise ValueError(f"docking.engine must be one of {allowed}, got {v!r}")
        return v


class PhysicsScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pocket_shell_radius: float = 6.0
    mmff_max_iters: int = 500
    scale_factor: float = 0.05


class MLScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fingerprint: str = "morgan"
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    test_size: float = 0.3
    n_estimators: int = 300
    random_seed: int = 42


class ConsensusWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    docking: float = 0.4
    physics: float = 0.3
    ml: float = 0.3

    @field_validator("docking", "physics", "ml")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Consensus weights must be non-negative")
        return v


class ConsensusConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: ConsensusWeights = Field(default_factory=ConsensusWeights)


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    physics: PhysicsScoringConfig = Field(default_factory=PhysicsScoringConfig)
    ml: MLScoringConfig = Field(default_factory=MLScoringConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)


class CovalentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    bonus_kcal_mol: float = -2.5


class ADMETConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_drug_likeness: float = 0.30


class SelectivityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    off_targets: list[str] = Field(
        default_factory=lambda: ["EGFR", "ITK", "TEC", "BMX", "JAK2"]
    )
    panel_glob: str = "data/fixtures/offtarget_{kinase}.csv"
    morgan_radius: int = 2
    morgan_n_bits: int = 2048
    n_estimators: int = 200
    random_seed: int = 42


class MoAConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    keep_assay_types: list[str] = Field(default_factory=lambda: ["B", "F"])


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    covalent: CovalentConfig = Field(default_factory=CovalentConfig)
    admet: ADMETConfig = Field(default_factory=ADMETConfig)
    selectivity: SelectivityConfig = Field(default_factory=SelectivityConfig)
    moa: MoAConfig = Field(default_factory=MoAConfig)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enrichment_fractions: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10])
    plot_formats: list[str] = Field(default_factory=lambda: ["png"])
    top_n_hits: int = 10

    @field_validator("enrichment_fractions")
    @classmethod
    def _fractions_in_range(cls, v: list[float]) -> list[float]:
        for f in v:
            if not (0.0 < f <= 1.0):
                raise ValueError(f"enrichment fraction must be in (0, 1], got {f}")
        return v


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str = "fast"
    fast_actives: int = 20
    fast_decoys: int = 20
    output_dir: str = "data/results"
    log_level: str = "INFO"

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in {"fast", "full"}:
            raise ValueError(f"runtime.mode must be 'fast' or 'full', got {v!r}")
        return v


class Config(BaseModel):
    """Top-level, immutable pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    target: TargetConfig
    data: DataConfig = Field(default_factory=DataConfig)
    ligand_prep: LigandPrepConfig = Field(default_factory=LigandPrepConfig)
    docking: DockingConfig = Field(default_factory=DockingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a pipeline configuration.

    Args:
        path: Optional path to a YAML config file. If ``None``, the packaged
            ``config/default.yaml`` is loaded.

    Returns:
        A frozen :class:`Config` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML fails schema validation.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
