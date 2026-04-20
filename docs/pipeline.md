# Pipeline reference

A short per-stage walkthrough of what each module does and how data flows
between them. Read alongside `README.md` (architecture diagram) and
`docs/methods.md` (equations and references).

## Stage map

| # | Stage              | Module                               | Input                          | Output                                     |
|---|--------------------|--------------------------------------|--------------------------------|--------------------------------------------|
| 1 | Fetch actives       | `data/chembl.py`                     | ChEMBL target ID               | `actives.csv` with pChEMBL                 |
| 2 | Generate decoys     | `data/decoys.py`                     | actives SMILES + candidate pool| decoy SMILES list                          |
| 3 | Prepare receptor    | `data/receptor.py`                   | PDB file                       | cleaned PDB + pocket + docking box         |
| 4 | Prepare ligands     | `data/ligands.py`                    | SMILES                         | MMFF-minimised 3D `PreparedLigand`         |
| 5 | Dock                | `docking/{engine,cached,vina}.py`    | `PreparedLigand`               | `DockingResult`                            |
| 6 | Rescore physics     | `scoring/physics.py`                 | `PreparedLigand`               | `PhysicsBreakdown`                         |
| 7 | Rescore ML          | `scoring/ml.py`                      | SMILES + labels                | `MLSplit` (train/test scores)              |
| 8 | Consensus           | `scoring/consensus.py`               | three score vectors            | `ConsensusResult.frame`                    |
| 9 | Metrics + viz       | `metrics/enrichment.py`, `viz/plots` | scores + labels                | `ScorerReport` + PNG artefacts             |

## Stage 1 — Fetch actives

Two entry points:

* **`fetch_live(ChEMBLQuery)`** — queries the ChEMBL web service via the
  optional `chembl-webresource-client` dependency. Returns a cleaned
  `pandas.DataFrame` in the canonical schema.
* **`load_cached(path)`** — reads a previously-saved CSV (same schema).
  This is the default path for CI and reproducible runs.

Both paths run through `_deduplicate`, which collapses duplicate
`molecule_chembl_id` rows to the strongest measured activity.

## Stage 2 — Decoys

`DecoyGenerator.generate(actives, candidates, count)` applies two filters
to each candidate SMILES:

1. **Property window match** — `LigandProperties` (MW, logP, HBA, HBD,
   rotatable bonds, formal charge) must fall within configurable absolute
   deltas of at least one active. Defaults correspond to DUD-E defaults.
2. **Topological dissimilarity** — Tanimoto similarity on Morgan r=2
   fingerprints vs every active must be **below** `similarity_cutoff`
   (default 0.35). Prevents near-duplicates of known actives from being
   picked as decoys.

The generator is deterministic: iteration order of `candidate_smiles` is
preserved, and the first `count` acceptable candidates are returned.

## Stage 3 — Receptor

`ReceptorPreparer.clean()` writes a PDB with

* all water residues (HOH/WAT) removed,
* all HETATM residues except the reference ligand removed,
* standard protein residues untouched.

`pocket_from_reference_ligand()` then extracts the reference ligand's
heavy-atom coordinates, computes the centroid, and builds a
`DockingBox` padded by `box_padding` Å on every face.

## Stage 4 — Ligand preparation

For each SMILES:

1. Parse with RDKit, sanitise.
2. `Chem.AddHs(mol)` — add explicit hydrogens.
3. ETKDGv3 embedding (seeded). On failure, retry with random coords.
4. MMFF94 minimisation. Energy and convergence flag stored on
   `PreparedLigand`.

Failures are returned as a `PreparedLigand` with `embed_success=False`;
the pipeline filters these out before docking.

## Stage 5 — Docking

The `DockingEngine` ABC enforces a single public method:

```python
def dock(self, ligand: PreparedLigand,
         receptor_pdbqt: str | Path,
         box: DockingBox) -> DockingResult: ...
```

Two engines today, one knob away from three:

* **`CachedEngine`** — reads a CSV of pre-computed affinities keyed by
  ligand name. Default. Trivially reproducible.
* **`VinaEngine`** — runs AutoDock Vina via its Python bindings. Optional
  dependency. Caller-side parallelism via `dock_many`.

`build_engine(DockingConfig)` is the one place that instantiates them,
so swapping engines is a single config change.

## Stage 6 — Physics

See `docs/methods.md` for the equations. Returns one
`PhysicsBreakdown(hydrophobic, hbond, strain, size, total)` per ligand.

## Stage 7 — ML

Trains a Random Forest on Morgan fingerprints. Returns an `MLSplit`:

```python
MLSplit(
    train_names=[...],          # IDs used for training
    test_names=[...],           # IDs held out
    test_labels=np.ndarray,     # 1 = active, 0 = decoy
    test_probabilities=np.ndarray,
    test_scores=np.ndarray,     # negated probabilities (lower = better)
)
```

The rescorer deliberately scores only the held-out test set — there is
no leakage between training and evaluation data.

## Stage 8 — Consensus

`consensus_score(...)` returns a `ConsensusResult.frame` with columns:

```
name, docking, physics, ml, docking_z, physics_z, ml_z, consensus
```

`consensus` is the weighted sum of the three z-scores. Lower = better.

## Stage 9 — Metrics + viz

For each scorer (docking, physics, ml, consensus):

* `roc_auc(labels, scores)` — flips sign convention to match sklearn.
* `enrichment_factor(labels, scores, fraction)` — sorts ascending,
  counts actives in the top fraction, divides by the overall rate.
* `scorer_report(...)` — bundles AUC + EFs + ROC curve arrays.

The viz module writes three PNGs:

* `roc.png` — overlaid ROC curves.
* `enrichment.png` — grouped bar chart across scorers × fractions.
* `top_hits.png` — 2D structure grid of the top-N consensus-ranked
  actives.

## Error handling

The pipeline fails fast on:

* Unparseable / incomplete config (pydantic `ValidationError`).
* Missing cached docking CSV (`FileNotFoundError`).
* <4 ligands surviving ligand prep + docking (insufficient for an
  ML train/test split; `RuntimeError`).
* Missing columns in actives / decoys CSVs (`ValueError`).

All other failures are logged at WARNING and the offending ligand is
skipped so one bad molecule cannot abort a 2,000-compound run.
