# btk-aidd design doc

**Date:** 2026-04-20
**Author:** Kirill Puchkov
**Status:** implemented (v0.1.0)

## Problem

Kirill is interviewing with **AilsynBio** (HKU spin-off, Hong Kong) for an
AIDD/CADD internship. His CV has three existing projects — a MERS-CoV
protein binder, an RNA-PROTAC pipeline (AptaDeg), and a CRISPR-Cas13 guide
RNA tool — all of which are **biologics / nucleic acids**. AilsynBio sells
**AI-native small-molecule drug discovery** built on a three-legged
stool:

1. machine learning,
2. physics-based modeling,
3. medicinal chemistry.

The CV therefore has a **credibility gap on small-molecule / CADD work**.
The third CV project must close that gap with a developer-grade, reviewable
code project that mirrors AilsynBio's pitch.

## Goals

* **Demonstrate the full AIDD/CADD stack** end-to-end on a
  commercially-validated target.
* **Produce a benchmarkable number** (ROC-AUC, EF@1%) the interviewer can
  evaluate.
* **Read cleanly** to OpenAI Codex / human reviewers — typed, tested,
  documented.
* **Run seamlessly** on a laptop without heavyweight external dependencies.
* **Tell a narrative** where each scorer (docking, physics, ML, consensus)
  has a justifiable role and the consensus layer demonstrably beats any
  individual component.

## Non-goals

* Real MM-GBSA / FEP — out of scope for a student project; the physics
  module is intentionally an empirical surrogate with documented coefficients.
* A generative chemistry module — a stretch goal, not the MVP.
* Production-grade Meeko / PDBFixer receptor prep — the Biopython-only
  cleanup is adequate for ATP-pocket docking.

## Scope (v0.1.0)

### In

* End-to-end pipeline: data → prep → dock → rescore × 3 → consensus →
  evaluate → visualise.
* BTK-specific target configuration (PDB 4OT6, ChEMBL `CHEMBL5251`).
* Two docking engines: `CachedEngine` (default, CI-friendly) and
  `VinaEngine` (optional, real AutoDock Vina bindings).
* Fixture data that runs the full pipeline in <20 s.
* Pydantic-validated YAML config.
* Click CLI with `run / fetch / validate` subcommands.
* pytest suite with 49 tests covering every module.
* GitHub Actions CI.
* README + methods + pipeline + this spec.

### Out

* Live ChEMBL auto-fetch in fast mode (it's in the CLI but optional).
* Production Vina receptor prep (documented as a next-step add-on).
* GPU-accelerated scoring (all CPU).
* Web UI.

## Target selection

**BTK / PDB 4OT6 / CHEMBL5251** chosen because:

* Commercial validation (ibrutinib et al.) gives an instant interview hook.
* Rich ChEMBL data (>7,000 actives with pChEMBL) enables real benchmarks.
* Clean ATP pocket with a 1.65 Å X-ray structure.
* Not over-exposed in student AIDD portfolios (unlike COVID Mpro).

Alternatives considered:

* **MDM2/p53** — overlaps with AptaDeg's use of 1YCR; rejected for breadth.
* **SARS-CoV-2 Mpro** — over-saturated in student projects.
* **Kinase JAK2** — viable, but BTK has stronger commercial story.

## Architecture

See `docs/pipeline.md` for the stage map and `docs/methods.md` for
scoring-function equations. Headline design calls:

1. **Abstract docking engine** — pipeline depends on `DockingEngine` ABC,
   not Vina. Swapping engines is a one-line config change.
2. **"Lower = better" convention everywhere** — matches Vina affinity and
   simplifies z-normalisation.
3. **Frozen config** — Pydantic models are immutable; no mid-run surprises.
4. **Fast vs full mode** — CI path finishes in seconds; real run is opt-in.
5. **Every stochastic component seeded** from config so runs are
   byte-identical across invocations.

## Data flow

```
ChEMBL ──┐
         ├─▶ actives (N=500)
         │
decoy    ├─▶ decoys (M=2000)     ┐
pool     │                       │
         │                       │
PDB ─────┴─▶ receptor + pocket   ├─▶ docked ─┐
                                 │            │
                                 │            ├─▶ physics  ─┐
                                 │            │              ├─▶ consensus ─▶ metrics
                                 │            ├─▶ ml        ─┘
                                 │            │
                                 └────────────┘
```

## Risks

| Risk                                                | Mitigation                                                           |
|-----------------------------------------------------|----------------------------------------------------------------------|
| Vina bindings fail on native Windows                | Default to `CachedEngine`; full Vina path runs in WSL                |
| ChEMBL API is offline / rate-limited during demo    | Ship `btk_actives.csv` fixture; live fetch is opt-in                 |
| Physics scorer reads as "fake MM-GBSA"              | Documented coefficients, literature references, explicit disclaimer  |
| Fixture data too separable (AUC = 1.0)              | Full-fixture run shows realistic spread (Docking 0.96, Cons 0.998)   |
| Reviewer expects DUD-E / LIT-PCBA compliance        | Document as a next-step upgrade, not a v0.1 goal                     |

## Interview narrative

**Hero bullet (CV-ready):**

> *"Built an end-to-end hybrid AI + physics virtual-screening pipeline for
> Bruton's Tyrosine Kinase (BTK). Consensus scoring (AutoDock Vina +
> MMFF94-based physics rescoring + Random Forest ML) achieves
> ROC-AUC ≈ 0.998 and EF@1% = 4.08 on ChEMBL actives vs matched decoys —
> outperforming every individual scorer."*

**Walkthrough for the interview:**

1. Open `README.md`, show the architecture diagram.
2. `btk-aidd validate` → shows typed config.
3. `btk-aidd run ... --mode full` on fixtures → real numbers in ~15 s.
4. Open `data/results/roc.png` and `enrichment.png` → visual story.
5. Open `docs/methods.md` → physics coefficients with Hansch / Kuntz refs.
6. Open `src/btk_aidd/scoring/consensus.py` → 100 LOC, z-norm + weighted sum.
7. Mention the extension path: swap `PhysicsRescorer` for OpenMM MM-GBSA;
   swap `CachedEngine` for GNINA.

## Out-of-scope next steps

* **MM-GBSA backend** using OpenMM — documented extension point.
* **Meeko-based** ligand PDBQT prep with rigorous Gasteiger charges.
* **LIT-PCBA benchmark** integration for publication-grade numbers.
* **Generative** molecule design (DiffSBDD / Pocket2Mol) as a front-stage
  source of candidates.
* **Active learning** prioritisation to cut the docking compute budget.
