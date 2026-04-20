# Methods

Reference documentation for the scoring functions, receptor preparation,
and benchmark metrics used by `btk-aidd`.

## 1. Receptor preparation

The canonical input is **PDB 4OT6**, a 1.65 Å co-crystal of BTK with
ibrutinib (residue name `1E8`).

Our receptor prep (`src/btk_aidd/data/receptor.py`) performs:

1. Parse with `Bio.PDB.PDBParser`.
2. Reject water (`HOH`/`WAT`) and all HETATM residues whose three-letter
   name is **not** the reference ligand.
3. Write the cleaned coordinates via `Bio.PDB.PDBIO`.
4. Compute the pocket centroid as the arithmetic mean of the reference
   ligand's heavy-atom coordinates.
5. Build an axis-aligned **docking box** by padding the ligand's bounding
   box by `box_padding` Å on every face (default 8 Å).

For production use (full hydrogen network, missing side-chain rebuilding)
run **PDBFixer** on the cleaned PDB; then feed through **Meeko's
`MoleculePreparation`** to produce a PDBQT with explicit AutoDock atom
types and Gasteiger charges. The Vina wrapper in this repo accepts
either our RDKit-derived PDBQT or a Meeko-prepared PDBQT.

## 2. Ligand preparation

`LigandPreparer` embeds SMILES in 3D using RDKit's **ETKDGv3** algorithm
(Experimental Torsion + Distance Geometry v3) with:

* a deterministic `embed_seed` (default 42),
* `useSmallRingTorsions=True`,
* a retry with `useRandomCoords=True` if the first attempt fails.

The embedded mol is then minimised with **MMFF94** (or MMFF94s) for up to
`max_mmff_iters=200` steps. The final MMFF energy and a convergence flag
are retained on the `PreparedLigand` dataclass.

## 3. Physics rescoring

`PhysicsRescorer` (`src/btk_aidd/scoring/physics.py`) computes an
additive ΔG_physics estimate from four terms:

| term                       | formula                     | coefficient       | source                         |
| -------------------------- | --------------------------- | ----------------- | ------------------------------ |
| Hydrophobic (Hansch)        | `α · logP`                 | α = −0.70 kcal/mol | Hansch & Leo, 1979           |
| Hydrogen-bond potential     | `β · (HBA + HBD)`          | β = −0.50 kcal/mol | Böhm, 1994 (estimate)         |
| MMFF94 strain penalty       | `γ · ΔE_strain`            | γ = +0.10          | empirical                      |
| Heavy-atom burial           | `δ · N_heavy`              | δ = −0.08 kcal/mol | Kuntz *et al.*, 1999 ("rule") |

The strain energy is `E(input_conformer) − E(minimised_conformer)`. Since
the input is already MMFF-minimised by `LigandPreparer`, strain is small
by construction; it is included so that upstream changes to conformer
generation do not silently bypass the term.

This function is **not MM-GBSA** and makes no absolute-affinity claim.
It is useful for *ranking* alongside docking, and it picks up orthogonal
signal (logP, polar contacts) that neither docking nor an unaware ML
classifier capture cleanly. To upgrade to MM-GBSA:

1. Extend `PhysicsBreakdown` with an `mm_gbsa` field.
2. Plug OpenMM into a new `MMGBSAEngine` class that accepts a docked pose
   and returns a ΔG_bind. No pipeline changes required.

## 4. ML rescoring

`MLRescorer` (`src/btk_aidd/scoring/ml.py`) trains a
`sklearn.ensemble.RandomForestClassifier` on:

* **Features:** Morgan fingerprints (radius 2, 2,048 bits, ECFP4-equivalent)
  computed with `rdFingerprintGenerator.GetMorganGenerator`.
* **Labels:** binary active (1) / decoy (0).
* **Split:** stratified train / test (default 70/30) seeded from
  `config.scoring.ml.random_seed`.
* **Hyperparameters:** `n_estimators=300`, `class_weight="balanced"`,
  `n_jobs=-1`.

The rescorer returns:

* the test-set probabilities (`test_probabilities`),
* their **sign-flipped copy** (`test_scores`) so that "lower = better"
  lines up with docking and physics conventions,
* the train / test split for downstream auditing.

## 5. Consensus scoring

`consensus_score` (`src/btk_aidd/scoring/consensus.py`) is a simple,
auditable weighted sum:

1. **Z-normalise** each scorer independently:
   `z_i = (x_i − mean) / std`. Zero-std arrays are returned mean-centred
   with scale unchanged, so ablation studies that zero out one scorer still
   run.
2. **Weight and sum:**
   `consensus = w_dock · dz + w_phys · pz + w_ml · mz`.

Weights are non-negative and do *not* have to sum to 1 — that keeps weight
ablations trivially simple.

## 6. Evaluation metrics

`src/btk_aidd/metrics/enrichment.py` computes:

* **ROC-AUC** via `sklearn.metrics.roc_auc_score`, passing in the negated
  score so the convention flips from "lower = better" to "higher = more
  likely active."
* **Enrichment factor** at `fraction f`:
  `EF(f) = (hits in top f) / (overall active rate × top-f size)`.
* **`ScorerReport`** bundles AUC, EF at multiple fractions, the ROC curve
  arrays, and active / decoy counts.

Fractions default to `[0.01, 0.05, 0.10]` — the standard virtual-screening
triple.

## 7. Reproducibility

Every stochastic component is seeded from config:

* `data.random_seed`           — subsample order in fast mode.
* `ligand_prep.embed_seed`     — ETKDG embedding.
* `scoring.ml.random_seed`     — RF bootstrap + train/test split.

Running the same config twice on the same fixture data therefore produces
byte-identical CSV outputs.
