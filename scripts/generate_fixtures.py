"""Generate deterministic fixture CSVs for the test / fast-mode pipeline.

Run once::

    python scripts/generate_fixtures.py

Writes under ``data/fixtures/``:

* ``btk_actives.csv``            ChEMBL-schema actives with assay_type + activity_comment
* ``btk_decoys.csv``             matched-property decoy table
* ``cached_docking_scores.csv``  synthetic Vina-like affinities for both
* ``offtarget_<KINASE>.csv``     per-kinase panels for selectivity training

The SMILES are drawn from a hand-curated list of published kinase inhibitors
(actives) and diverse drug-like scaffolds (decoys). All SMILES are validated
with RDKit at generation time; unparseable entries are dropped with a
warning. Docking scores follow realistic Gaussian distributions:

* actives:  N(mean = -9.2 kcal/mol, sigma = 0.9)
* decoys:   N(mean = -6.4 kcal/mol, sigma = 1.2)

Off-target panels use BTK actives + decoys as a proxy pool. Each kinase has
a deterministically sampled set of "actives" (drawn so that selectivity is
non-trivial: some BTK actives also hit EGFR/ITK, others do not).

The seed is fixed (42) so the generated CSVs are byte-identical across runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # quiet RDKit parser warnings for known-bad test inputs

# ---------------------------------------------------------------------- actives

# A curated set of published kinase-domain-binding drug molecules. These are
# used as *stand-in* actives for fixture purposes — the pipeline's full mode
# should fetch the real BTK activity set from ChEMBL via `btk-aidd fetch`.
ACTIVE_SMILES: list[tuple[str, str, float]] = [
    ("CHEMBL221959", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", 8.1),  # imatinib-like
    ("CHEMBL553", "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", 8.3),  # erlotinib
    ("CHEMBL939", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", 8.0),  # gefitinib
    ("CHEMBL1336", "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C", 7.9),  # sunitinib
    ("CHEMBL1201583", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", 8.2),  # sorafenib
    ("CHEMBL535", "COc1cc2c(cc1OC)c(Nc1ccc(Br)cc1F)ncn2", 7.5),
    ("CHEMBL1200976", "Cc1cc(Nc2cc(N3CCN(CCO)CC3)nc(Nc3ccc(N)cc3)n2)ccn1", 7.6),
    ("CHEMBL1231024", "Cn1cc(-c2cnc3[nH]ccc3n2)c(-c2ccc(F)cc2)n1", 8.4),
    ("CHEMBL2103875", "Nc1ncnc2[nH]cnc12", 7.2),
    ("CHEMBL1207490", "Cc1ncc(-c2ccc(F)cc2)c(-c2ccc3[nH]ncc3c2)n1", 7.8),
    ("CHEMBL1243977", "CC(C)(C)c1nc(-c2cccnc2)c(-c2ccncc2)[nH]1", 7.3),
    ("CHEMBL1614701", "CN1CCN(c2ccc(Nc3ncnc4[nH]ccc34)cc2)CC1", 7.4),
    ("CHEMBL1743047", "Clc1ccc(-c2nnc3c(-c4ccncc4)cccn23)cc1", 7.7),
    ("CHEMBL1800594", "Fc1ccc(-c2nc3ccccc3[nH]2)cc1", 7.1),
    ("CHEMBL1909120", "COc1ccc(-c2cnc3ccccc3n2)cc1", 7.0),
    ("CHEMBL2010150", "O=C(Nc1ccc(-c2ccncc2)cc1)c1cccnc1", 7.5),
    ("CHEMBL2114176", "CCn1c(=O)c2c(ncn2C)n(C)c1=O", 7.2),  # xanthine scaffold
    ("CHEMBL2215126", "Cc1nn(-c2ccccc2)c(=O)c1-c1ccncc1", 7.6),
    ("CHEMBL2335168", "CN(C)c1ccc(-c2nc(N)nc3ccccc23)cc1", 7.4),
    ("CHEMBL2454185", "Cc1cc(N)c(-c2cnc(N)nc2)cc1F", 7.3),
    ("CHEMBL2534321", "Nc1nc(-c2ccc(Cl)cc2)nc2[nH]ncc12", 7.8),
    ("CHEMBL2645587", "CCOc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OC", 7.7),
    ("CHEMBL2756819", "O=C(Nc1ccc(-c2nc(N)nc(N)c2)cc1)C1CC1", 7.5),
    ("CHEMBL2867043", "CC(=O)Nc1ccc(-c2ncc(C(=O)N)s2)cc1", 7.2),
    ("CHEMBL2978275", "COc1ccc(Nc2ncnc3[nH]ccc23)cc1", 7.1),
    ("CHEMBL3089501", "CC1CCN(c2ccnc(Nc3ccc(F)cc3)n2)CC1", 7.6),
    ("CHEMBL3200713", "Clc1ccc(Nc2ncnc3ccncc23)cc1", 7.4),
    ("CHEMBL3311957", "CC(C)N1CCN(c2cc(N)ncn2)CC1", 7.3),
    ("CHEMBL3423201", "Nc1nc(-c2cccnc2)nc2[nH]ccc12", 7.5),
    ("CHEMBL3534433", "O=C(Nc1ccccc1)Nc1cccc(-c2ccncc2)c1", 7.2),
    ("CHEMBL3645687", "COc1ccccc1Nc1ncc2ncn(C)c2n1", 7.7),
    ("CHEMBL3756929", "Fc1cccc(Nc2nccc(-c3ccncc3)n2)c1", 7.8),
    ("CHEMBL3868161", "Cc1ccc(-c2nc(N)nc(N3CCOCC3)n2)cc1", 7.3),
    ("CHEMBL3979413", "COc1cc(N)c(-c2nccc(N3CCOCC3)n2)cn1", 7.4),
    ("CHEMBL4090635", "CC(C)Oc1ccc(-c2nc(N)ncc2-c2ccncc2)cc1", 7.6),
    ("CHEMBL4201867", "Cn1ncc2c(Nc3ccc(F)cc3Cl)ncnc21", 8.0),
    ("CHEMBL4313089", "Clc1ccc(N2CCN(c3nccs3)CC2)cc1", 7.1),
    ("CHEMBL4424321", "O=C(c1ccncc1)N1CCN(c2ccccc2)CC1", 7.0),
    ("CHEMBL4535543", "Cc1cc(Nc2ncccn2)cc(-c2ccncc2)c1", 7.2),
    ("CHEMBL4646765", "Nc1ccc(-c2nc3ccccc3s2)cc1", 7.3),
]


# ----------------------------------------------------------------------- decoys

# Diverse drug-like scaffolds without explicit kinase-binding pharmacophores.
# Used only for fixture balance; full-mode decoys come from the DUD-E-style
# matched-property generator.
DECOY_SMILES: list[str] = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(=O)NC1=CC=C(O)C=C1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "OC(=O)C1=CC=CC=C1O",
    "NC1=NC(=O)N(C=C1)C1OC(CO)C(O)C1O",
    "CCN(CC)CCNC(=O)C1=CC=C(N)C=C1",
    "OC1=CC=C(CC2=CC=C(O)C=C2)C=C1",
    "CN(C)C1=CC=C(C=C1)S(=O)(=O)NC1=NC=CS1",
    "CC(C)(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1",
    "CN(C)CCCC1(C2=CC=CC=C2)OCC3=C1C=CC(=C3)C#N",
    "ClC1=CC=C(C=C1)C(=O)NC2=CC=C(N)C=C2",
    "CC1=NC(=NC(=N1)N)N",
    "NCCC1=CC(O)=C(O)C=C1",
    "CC(N)CC1=CC=CC=C1",
    "COC1=CC=CC=C1OC",
    "OCC1OC(O)C(O)C(O)C1O",
    "CCCCCCCC(O)=O",
    "CC(=O)OCC(=O)C1(O)CCC2C3CCC4=CC(=O)CCC4C3CCC12C",
    "CN1CCC23C4OC5=C(O)C=CC(C[C@@H]1C2C=CC3O)=C45",
    "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(O)=O",
    "COC1=CC2=C(C=C1OC)C(=CN2)CC(N)C(O)=O",
    "OCC(O)COC1=CC=C(CCN)C=C1",
    "COC1=C(OC)C=C2C=C(C(=O)NC3=CC=CC=C3)C(=O)OC2=C1",
    "CC(C)NCC(O)COC1=CC=CC2=C1C=CC=C2",
    "CCCCNC(=S)NC1=CC=CC=C1",
    "OC1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(O)C=C3O",
    "CC1=CC(=O)OC2=CC=CC=C12",
    "CC(=O)C1=CC=C(C=C1)N",
    "O=C(NC1=CC=CC=C1)NC1=CC=CC=C1",
    "CC1=CC(=NC=N1)N2CCOCC2",
    "OC1=CC=C2C=CC(=O)OC2=C1",
    "CC(=O)OC1=CC(=CC(=C1)O)C=CC(O)=O",
    "CCOC(=O)C1=CC=CN=C1",
    "COC1=CC=C(CNC(C)C)C=C1",
    "OC(=O)C1=CC2=CC=CC=C2C=C1",
    "CN1CCC(=CC1)C1=CC=CC=C1",
    "COC1=C(O)C=CC(=C1)C=CC(=O)O",
    "N#CC1=CC=C(C=C1)C(=O)N",
    "CC(=O)c1ccc(cc1)S(=O)(=O)N",
    "OC(=O)C1=CC=C(O)C(=C1)O",
    "CCN(CC)C(=S)SSC(=S)N(CC)CC",
    "CC1=CC=C(C=C1)C(=O)C",
    "CC1=CC(=C(C=C1)O)C=O",
    "CCC1=CC=C(N)C=C1",
    "CC(=O)NC1=CC=CC=C1",
    "O=C1NC(=O)NC(=O)N1",
    "CC1=CC=CC=C1N",
    "OCC1=CC=CC=C1",
    "CC1=C(C=CN1)C=O",
    "NC1=CC=C(C=C1)S(N)(=O)=O",
    "OC(=O)CC1=CC=CC=C1",
    "COC1=CC=CC=C1C=O",
    "CC1=CC(=CC=C1)C(O)=O",
    "CC(O)C1=CC=CC=C1",
    "OC1=CC=CC=C1C(O)=O",
    "CC1=CC=NC=C1",
    "NC(=O)C1=CC=CN=C1",
    "CN(C)C1=CC=CC=C1",
    "NC1=CC=CC=N1",
    "OCCOC1=CC=CC=C1",
    "CCC(=O)NC1=CC=CC=C1",
    "CCCCC(=O)NC1=CC=CC=C1",
    "NCCC1=CC=CC=C1",
    "OCC1=CC=C(C=C1)O",
    "CC(=O)OC1=CC=CC=C1",
    "NC1=NC(=CC=N1)N",
    "CC(C)C1=CC=CC=C1O",
    "NC1=CC(=O)NC=N1",
    "OC(=O)C1=CC=CC=C1",
    "COC1=C(O)C=CC=C1",
    "OC(=O)C1=CC=CN=C1",
    "CC1=NN(C=C1C)C",
    "OC1=NC=CC=C1",
    "CNC(=O)C1=CC=NC=C1",
    "CCOC1=CC=CC=C1",
    "CC(=O)NCC1=CC=CC=C1",
    "CC1=CC=C(C=C1)N(C)C",
    "NC1=CC=C(C=C1)C(=O)O",
    "O=C1C=CC(=O)NC1",
    "OC1=CC=C(C=C1)CCN",
    "CC1=CC(=O)NC(=S)N1",
    "OC(=O)CC(=O)O",
    "CN1CCN(CC1)C1=CC=CC=N1",
    "CC1=NC=CN1C",
    "NCC1=CC=C(O)C=C1",
    "CC(=O)NC1=CC=C(O)C=C1",
    "OCCN1CCOCC1",
    "CNC1=NC=CC=C1",
    "NC1=CC=CC(=N1)C(F)(F)F",
    "OC(=O)C1=CC(=CC=C1)O",
    "COC1=CC(=CC=C1)N",
    "NC1=CC=C(C=C1)S(N)(=O)=O",
    "OC1=CC(=CC=C1O)C(O)=O",
    "CCC1=CC=C(C=C1)N",
    "CN1C=CC=C1",
    "OC(=O)CC1=CN=CN1",
    "CCOC(=O)C1=CC=CC=C1",
    "NC1=NC=NC2=C1NC=N2",
    "CC1=CN=CN1C",
    "OC1=C(C=CC=C1)C(=O)O",
    "COC1=CC(=CN=C1)C",
    "NC1=CC=C(Cl)C=C1",
    "CC1=CC=CN=C1O",
    "OCC1=NC=CN1",
    "NC1=NC(=O)C=N1",
    "CC(C)(C)NC1=CC=CC=C1",
    "OC(=O)C(=CC1=CC=CC=C1)C=O",
    "CCN(CC)CC1=CC=CC=C1",
    "CC(=O)C1=CC=C(O)C=C1",
    "NC(=O)CC1=CC=CC=C1",
    "CC1=CC=C2NC=CC2=C1",
    "OC1=CC=CC=C1O",
    "COC1=CC(=CC=C1)C=O",
    "NCC1=NC=CN1",
    "OC1=CC=NC=C1",
    "CC1=C(Cl)C=CC=C1",
    "NC1=CC(=C(O)C=C1)O",
    "CN1C=NN=C1",
    "NC(=O)NC1=CC=CC=C1",
    "CCC1=CC=NC=C1",
    "NC1=CC=C(F)C=C1",
]


OFF_TARGET_KINASES: tuple[str, ...] = ("EGFR", "ITK", "TEC", "BMX", "JAK2")

# Seeded per-kinase cross-reactivity rates: fraction of BTK actives that also
# hit this off-target (plausible real-world numbers; JAK2 is the most
# selective-friendly).
_OFF_TARGET_HIT_RATE: dict[str, float] = {
    "EGFR": 0.25,
    "ITK": 0.45,
    "TEC": 0.60,
    "BMX": 0.35,
    "JAK2": 0.15,
}

# Ibrutinib is clinically known to have promiscuity with ITK, TEC, BMX and
# light EGFR. We inject one "ibrutinib-like" molecule into the actives so
# the covalent detector and selectivity scorer have a realistic example.
_IBRUTINIB_SMILES = "C=CC(=O)N1CCC[C@H](C1)n1nc(c2c1ncnc2N)-c1ccc(Oc2ccccc2)cc1"


def main(output_dir: str | None = None) -> None:
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "data" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Actives ----------------------------------------------------------------
    actives_rows: list[dict[str, object]] = []
    # Insert the ibrutinib archetype first so it is always present.
    actives_rows.append(
        {
            "molecule_chembl_id": "CHEMBL1873475",
            "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(_IBRUTINIB_SMILES)),
            "standard_type": "IC50",
            "standard_value_nM": 0.5,
            "pchembl_value": 9.3,
            "target_chembl_id": "CHEMBL5251",
            "assay_type": "F",
            "activity_comment": "inhibitor",
        }
    )

    for chembl_id, smi, pchembl in ACTIVE_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"WARN  unparseable active SMILES skipped: {chembl_id} {smi}", file=sys.stderr)
            continue
        canonical = Chem.MolToSmiles(mol)
        # 80% functional assays, 20% binding-only
        assay_type = "F" if rng.random() < 0.8 else "B"
        actives_rows.append(
            {
                "molecule_chembl_id": chembl_id,
                "canonical_smiles": canonical,
                "standard_type": "IC50",
                "standard_value_nM": float(10 ** (9 - pchembl)),
                "pchembl_value": float(pchembl),
                "target_chembl_id": "CHEMBL5251",
                "assay_type": assay_type,
                "activity_comment": "inhibitor" if rng.random() < 0.9 else "",
            }
        )
    actives_df = pd.DataFrame(actives_rows)
    actives_df.to_csv(out_dir / "btk_actives.csv", index=False)
    print(f"Wrote {len(actives_df)} actives -> {out_dir / 'btk_actives.csv'}")

    # --- Decoys ----------------------------------------------------------------
    decoy_rows: list[dict[str, str]] = []
    for i, smi in enumerate(DECOY_SMILES):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"WARN  unparseable decoy SMILES skipped: idx={i} {smi}", file=sys.stderr)
            continue
        canonical = Chem.MolToSmiles(mol)
        decoy_rows.append({"name": f"DECOY_{i:04d}", "canonical_smiles": canonical})
    decoys_df = pd.DataFrame(decoy_rows)
    decoys_df.to_csv(out_dir / "btk_decoys.csv", index=False)
    print(f"Wrote {len(decoys_df)} decoys -> {out_dir / 'btk_decoys.csv'}")

    # --- Cached docking scores -------------------------------------------------
    active_scores = rng.normal(loc=-9.2, scale=0.9, size=len(actives_df))
    decoy_scores = rng.normal(loc=-6.4, scale=1.2, size=len(decoys_df))

    cached_rows: list[dict[str, object]] = []
    for name, score in zip(actives_df["molecule_chembl_id"], active_scores, strict=True):
        cached_rows.append({"name": str(name), "affinity_kcal_mol": float(score)})
    for name, score in zip(decoys_df["name"], decoy_scores, strict=True):
        cached_rows.append({"name": str(name), "affinity_kcal_mol": float(score)})

    cached_df = pd.DataFrame(cached_rows)
    cached_df.to_csv(out_dir / "cached_docking_scores.csv", index=False)
    print(f"Wrote {len(cached_df)} cached scores -> {out_dir / 'cached_docking_scores.csv'}")

    # --- Off-target kinase panels ----------------------------------------------
    # For each off-target kinase, sample a subset of BTK actives as "also
    # active against this kinase" (polypharmacology) plus all decoys as
    # negatives. The cross-reactivity rate is set per kinase so ITK/TEC look
    # promiscuous and JAK2 looks cleanly selective, matching clinical data.
    for kinase in OFF_TARGET_KINASES:
        hit_rate = _OFF_TARGET_HIT_RATE[kinase]
        n_hits = int(round(len(actives_df) * hit_rate))
        hit_indices = rng.choice(len(actives_df), size=n_hits, replace=False)
        hit_mask = np.zeros(len(actives_df), dtype=bool)
        hit_mask[hit_indices] = True

        rows: list[dict[str, object]] = []
        for i, smi in enumerate(actives_df["canonical_smiles"]):
            rows.append(
                {
                    "canonical_smiles": str(smi),
                    "label": 1 if hit_mask[i] else 0,
                    "source": "btk_actives",
                }
            )
        for smi in decoys_df["canonical_smiles"]:
            rows.append(
                {
                    "canonical_smiles": str(smi),
                    "label": 0,
                    "source": "decoy",
                }
            )
        panel_df = pd.DataFrame(rows)
        panel_df.to_csv(out_dir / f"offtarget_{kinase}.csv", index=False)
        print(
            f"Wrote {len(panel_df)} rows ({n_hits} actives) -> "
            f"{out_dir / ('offtarget_' + kinase + '.csv')}"
        )


if __name__ == "__main__":
    out_override = sys.argv[1] if len(sys.argv) > 1 else None
    main(out_override)
