# btk-aidd

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen.svg)](./tests)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

A hybrid **AI + physics** virtual-screening pipeline for **Bruton's Tyrosine
Kinase (BTK)**. Combines AutoDock Vina docking, a literature-grounded
physics-based rescorer, and a Morgan-fingerprint Random Forest classifier,
then evaluates every scorer individually and under a z-normalised weighted
consensus.

The pipeline is a reference implementation of the three-legged
**AI + physics-based modeling + medicinal chemistry** approach to small-
molecule drug discovery. It is deliberately small, reproducible, and
benchmark-oriented: every stage has a defined input/output contract, every
scorer reports ROC-AUC and enrichment factors at 1/5/10 %, and fixture data
makes the full workflow runnable in under 20 seconds on a laptop CPU.

---

## Why BTK?

Bruton's Tyrosine Kinase is a commercially validated oncology target:
**ibrutinib** (PCI-32765), **acalabrutinib**, and **zanubrutinib** are all
approved drugs with combined revenues above US $10B. It is an ideal
benchmark target for a student-grade AIDD project because

1. the **ATP-binding pocket** is well characterised (PDB 4OT6 at 1.65 Å with
   ibrutinib bound);
2. **ChEMBL target `CHEMBL5251`** contains >7,000 measured activities with
   pChEMBL values, so a reliable active / decoy split is possible;
3. the **covalent Cys481 warhead** of first-generation inhibitors gives a
   rich medicinal-chemistry talking point when discussing scoring
   limitations;
4. every major AIDD shop (Schrödinger, XtalPi, Insilico, Recursion) has
   published on BTK, so benchmarks are comparable.

---

## Pipeline architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   ChEMBL     │     │   RCSB PDB   │     │ Candidate    │
│ activities   │     │    4OT6      │     │ SMILES pool  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Active set   │     │ Pocket box   │     │ Decoy set    │
│ (N actives)  │     │ + receptor   │     │ (M decoys)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └───────┬────────────┴──────────┬─────────┘
               ▼                       ▼
        ┌─────────────┐         ┌─────────────┐
        │ RDKit ETKDG │         │  AutoDock   │
        │  + MMFF94   │────────▶│  Vina dock  │
        │  ligand prep│         │  (or cache) │
        └─────────────┘         └──────┬──────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
  │  Physics    │              │   Docking   │              │     ML      │
  │  MMFF94     │              │  affinity   │              │  Morgan+RF  │
  │  ΔG_hydro   │              │  (kcal/mol) │              │  P(active)  │
  │  + HB + …   │              │             │              │             │
  └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
         │                            │                            │
         └────────────┬───────────────┴──────────────┬─────────────┘
                      ▼                              ▼
               ┌────────────┐                ┌────────────┐
               │  z-norm    │                │  Per-scorer │
               │  weighted  │                │  EF @ 1/5/10│
               │  consensus │                │   ROC-AUC   │
               └─────┬──────┘                └─────┬──────┘
                     │                             │
                     └────────────┬────────────────┘
                                  ▼
                           ┌─────────────┐
                           │  scores.csv │
                           │  roc.png    │
                           │  ef.png     │
                           │  top_hits   │
                           └─────────────┘
```

The eight stages in order:

1. **Data acquisition** — ChEMBL actives filtered by pChEMBL; property-matched
   decoys generated DUD-E style with a Tanimoto topology filter.
2. **Receptor preparation** — 4OT6 cleaned with Biopython, pocket centroid
   and box extracted from the co-crystallised ibrutinib coordinates.
3. **Ligand preparation** — ETKDG 3D embedding + MMFF94 optimisation, seeded.
4. **Docking** — AutoDock Vina (or cached scores for CI).
5. **Physics rescoring** — additive ΔG_physics from logP, HBA/HBD counts,
   MMFF94 strain energy, and heavy-atom count (coefficients in
   `docs/methods.md`).
6. **ML rescoring** — Random Forest on Morgan fingerprints, class-balanced,
   stratified 70/30 split.
7. **Consensus** — z-normalised weighted sum of the three scorers.
8. **Evaluation** — ROC-AUC, EF@1%, EF@5%, EF@10%; top-hit structure grid.

---

## Install

```bash
# Clone
git clone https://github.com/kpuchkov1-code/btk-aidd.git
cd btk-aidd

# Python 3.10+
python -m venv .venv
source .venv/bin/activate          # Linux / macOS / WSL
# .\.venv\Scripts\activate         # Windows PowerShell

# Core dependencies
pip install -e .

# Optional extras
pip install -e ".[chembl]"         # live ChEMBL fetching
pip install -e ".[docking]"        # AutoDock Vina Python bindings
pip install -e ".[dev]"            # pytest, ruff, mypy
```

All core deps are pure-Python + pip-installable on Linux / macOS / Windows.
AutoDock Vina has C++ bindings that build cleanly on Linux / WSL; on native
Windows, stick to the default **cached** docking engine or run the full
pipeline inside WSL.

---

## Quick start (fast mode, ~20 s)

```bash
# Generate deterministic fixture data (once)
python scripts/generate_fixtures.py

# Run the pipeline end-to-end
btk-aidd run \
    --actives data/fixtures/btk_actives.csv \
    --decoys  data/fixtures/btk_decoys.csv \
    --mode    fast
```

Expected output:

```
=== Results ===
Docking    AUC=1.000  EF@1%=2.00, EF@5%=2.00, EF@10%=2.00
Physics    AUC=0.861  EF@1%=2.00, EF@5%=2.00, EF@10%=2.00
ML         AUC=1.000  EF@1%=2.00, EF@5%=2.00, EF@10%=2.00
Consensus  AUC=1.000  EF@1%=2.00, EF@5%=2.00, EF@10%=2.00

Artefacts written to data/results/
  scores:     data/results/scores.csv
  ROC plot:   data/results/roc.png
  EF plot:    data/results/enrichment.png
  top hits:   data/results/top_hits.png
```

The fast-mode fixture data deliberately has **well-separated active / decoy
docking distributions** so the pipeline reaches perfect AUC on docking and ML.
The physics scorer uses complementary intrinsic-ligand features and lands at
AUC ≈ 0.86 — a realistic signal that physics adds orthogonal information to
docking, not a duplicate of it.

---

## Full mode

Full mode performs a real ChEMBL fetch + Vina docking against a cleaned 4OT6
structure. Expect ~30 min on 8 CPU cores for 500 actives + 2,000 decoys.

```bash
# 1. Live ChEMBL fetch (requires the chembl extra)
btk-aidd fetch --output data/processed/btk_actives.csv

# 2. Download 4OT6 from RCSB
curl -L https://files.rcsb.org/download/4OT6.pdb -o data/raw/4OT6.pdb

# 3. (Separately) clean receptor and write PDBQT via your prep tool of
#    choice; see docs/methods.md for Open Babel + meeko instructions.

# 4. Run with the Vina engine enabled in config
btk-aidd run \
    --config config/full.yaml \
    --actives data/processed/btk_actives.csv \
    --decoys  data/processed/btk_decoys.csv \
    --mode    full
```

A template `config/full.yaml` is not shipped — copy `config/default.yaml` and
set `docking.engine: vina` and the appropriate paths.

---

## Results

Fast-mode artefacts after `btk-aidd run ...`:

* **`data/results/scores.csv`** — every ligand with its docking, physics,
  ML, z-normalised, and consensus scores, plus the ground-truth label.
* **`data/results/roc.png`** — ROC curves for all four scorers overlaid.
* **`data/results/enrichment.png`** — grouped bar chart of EF@1/5/10 %.
* **`data/results/top_hits.png`** — 2D structure grid of the top-N
  consensus-ranked actives.

---

## Code layout

```
src/btk_aidd/
├── config.py          pydantic-validated YAML config
├── logger.py          stdlib logging wrapper
├── pipeline.py        orchestrator (run_pipeline)
├── cli.py             click entry point (btk-aidd run|fetch|validate)
├── data/
│   ├── chembl.py      live + cached ChEMBL fetch
│   ├── decoys.py      DUD-E-style property-matched decoy generator
│   ├── ligands.py     RDKit ETKDG + MMFF94 ligand prep
│   └── receptor.py    Biopython PDB cleanup + pocket extraction
├── docking/
│   ├── engine.py      abstract DockingEngine + DockingResult
│   ├── cached_engine.py   reads affinities from CSV (default)
│   ├── vina_engine.py     AutoDock Vina wrapper (optional)
│   └── factory.py     build_engine(config)
├── scoring/
│   ├── physics.py     PhysicsRescorer (four-term ΔG_physics)
│   ├── ml.py          MLRescorer (Morgan-fp Random Forest)
│   └── consensus.py   z-normalised weighted consensus
├── metrics/
│   └── enrichment.py  ROC-AUC + EF@k + ScorerReport
└── viz/
    └── plots.py       matplotlib / seaborn figures
```

All modules are <200 LOC and have a single responsibility.

---

## Tests

```bash
pytest -q
```

49 tests covering config validation, data loaders, decoy generation, ligand
prep, receptor parsing, docking engines (cached + factory), physics and ML
scoring, consensus mathematics, enrichment metrics, and an end-to-end
integration run.

Markers:

* `@pytest.mark.requires_vina` — skipped unless AutoDock Vina is installed.
* `@pytest.mark.requires_network` — skipped when offline.

---

## Configuration

All runtime parameters live in `config/default.yaml` and are validated by
[pydantic](https://docs.pydantic.dev). Key sections:

| Section               | What it controls                                              |
| --------------------- | ------------------------------------------------------------- |
| `target`              | BTK identifiers (ChEMBL ID, PDB ID, reference ligand resname) |
| `data.actives`        | pChEMBL cutoff, activity types, max count                     |
| `data.decoys`         | DUD-E property windows, Tanimoto cutoff, count                |
| `ligand_prep`         | ETKDG seed, MMFF variant, minimisation iterations             |
| `docking`             | engine choice, exhaustiveness, box padding                    |
| `scoring.physics`     | pocket shell radius, MMFF max iters, scale factor             |
| `scoring.ml`          | Morgan radius/bits, RF n_estimators, test size                |
| `scoring.consensus`   | per-scorer weights                                            |
| `evaluation`          | enrichment fractions, plot formats, top-N for hit grid        |
| `runtime`             | fast vs full, output directory, log level                     |

Validate any config with:

```bash
btk-aidd validate --config path/to/your.yaml
```

---

## Design principles

* **One abstract docking engine** — the pipeline never imports Vina directly.
  All docking flows through `DockingEngine.dock()`. Swapping Vina for
  Smina, GNINA, or DiffDock is a one-file change.
* **Physics uses coefficients, not hand-waving** — the four-term ΔG_physics
  has documented literature coefficients (Hansch logP, Kuntz affinity).
* **Every scorer returns "lower = better"** — matches AutoDock Vina's
  convention and simplifies z-normalisation and metric code.
* **Frozen config** — once loaded, config is immutable; no module can silently
  change a parameter mid-run.
* **Fast mode by default** — CI, tests, and the quick-start path all finish
  in < 20 s. Full mode is opt-in.

---

## Limitations & next steps

* Physics scoring is an **empirical ΔG surrogate**, not MM-GBSA. The public
  API accepts a pose argument so that an OpenMM-backed MM-GBSA implementation
  can be dropped in without touching upstream code.
* Decoy generation uses a permissive DUD-E variant (absolute property windows
  instead of strict percentile matching). Full-mode users should pair this
  with a real DUD-E or LIT-PCBA benchmark for publication-grade numbers.
* The Vina wrapper writes PDBQT without Meeko; this trades a small amount of
  scoring accuracy for a smaller dependency footprint. `meeko` is a one-line
  upgrade when rigorous charges are required.

---

## Author

Built by **Kirill Puchkov** ([@kpuchkov1-code](https://github.com/kpuchkov1-code))
as a reference project for AI-driven small-molecule drug discovery.

## License

MIT. See [LICENSE](./LICENSE).
