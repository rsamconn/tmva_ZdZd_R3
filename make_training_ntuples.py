#!/usr/bin/env python3
"""
make_training_ntuples.py
========================
Stage 1 of the TMVA pipeline: read ZdZd13TeV Nominal/llllTree ROOT files,
apply event preselection and quadruplet selection, flatten jagged arrays to a
fixed-size feature vector per event, and write to a Parquet file.

The Parquet file produced here is the input to the Stage 2 training script.
All mass and momentum quantities are stored in MeV, consistent with the
ZdZd13TeV tree convention.

SELECTIONS APPLIED
------------------
Event preselection (all three must pass):
    passCleaning == True
    passNPV      == True
    passTriggers != 0

Quadruplet selection (first candidate per event passing both):
    SFOS:      llll_charge == 0  AND  llll_dCharge == 0
    Kinematic: pT(l1) >= 20 000 MeV   (leading lepton)
               pT(l2) >= 15 000 MeV   (subleading)
               pT(l3) >= 10 000 MeV   (subsubleading)
    (l1–l4 are stored in pT-descending order within each quadruplet candidate,
    matching the convention in ZdZdPP_alg.cxx.)

No further cuts are applied so that the remaining selection quantities
(isolation, dR, d0, trigger match, …) survive as BDT input features.

OUTPUT COLUMNS
--------------
Metadata / labels (not BDT inputs):
    label              int    1 = signal, 0 = background
    mc_channel_number  int    MC dataset number
    eventNumber        int    event number (for debugging / cross-checks)
    truth_zdzd_avgM    float  MeV  truth avg Zd mass (signal only; 0 for bkg)
    evtWeight_total    float  evtWeight * PileupWeight * llll_scaleFactor

BDT input features (all in MeV unless stated):
    mu                 float  average interactions per bunch crossing (pile-up)
    m_4l               float  MeV  four-lepton invariant mass
    avgM               float  MeV  (mab + mcd) / 2
    dM                 float  MeV  mab - mcd
    mab                float  MeV  leading dilepton mass
    mcd                float  MeV  subleading dilepton mass
    mad                float  MeV  alt. leading dilepton mass
    mbc                float  MeV  alt. subleading dilepton mass
    mcd_over_mab       float  dimensionless  mcd / mab  (MediumSR discriminant)
    min_sf_dR          float  rad  min ΔR between same-flavour leptons
    min_of_dR          float  rad  min ΔR between opp.-flavour leptons
                               *** SENTINEL: 9999999 for 4e / 4mu quads ***
    vtx_reduced_chi2   float  quadruplet vertex fit reduced chi²
                               *** SENTINEL: -999 when fit did not converge ***
    max_el_d0Sig       float  max |d0/σ(d0)| for electrons in quadruplet
                               *** SENTINEL: 0.0 for 4mu quads (no electrons) ***
    max_mu_d0Sig       float  max |d0/σ(d0)| for muons in quadruplet
                               *** SENTINEL: 0.0 for 4e quads (no muons) ***
    nCTorSA            int    number of CT or SA muons in quadruplet
    l_isIsolCloseBy    int    isolation bitmask (15 = all four leptons isolated)
    triggerMatched     int    trigger-matching bitmask (non-zero = matched)
    is_4e              int    1 if pdgIdSum == 44 (eeee),  else 0
    is_2e2mu           int    1 if pdgIdSum == 48 (eeμμ),  else 0
    is_4mu             int    1 if pdgIdSum == 52 (μμμμ),  else 0
    pT_l1              float  MeV  leading lepton pT
    pT_l2              float  MeV  subleading lepton pT
    pT_l3              float  MeV  subsubleading lepton pT
    pT_l4              float  MeV  trailing lepton pT
    eta_l1             float  leading lepton η   (ordered same as pT_l*)
    eta_l2             float  subleading lepton η
    eta_l3             float  subsubleading lepton η
    eta_l4             float  trailing lepton η

Columns marked SENTINEL contain placeholder values for certain quadruplet
flavours and must be handled (excluded or imputed) in the training script
before being passed to TMVA.

USAGE
-----
Signal + background (full training set):
    python3 make_training_ntuples.py \\
        --signal   path/to/sig_mZd30.root path/to/sig_mZd60.root \\
        --background path/to/ZZstar_bkg.root \\
        --output   data/training_ntuples/training.parquet

Signal only (initial testing without background):
    python3 make_training_ntuples.py \\
        --signal   data/example_data/ZdZd13TeV/mc23a_mZd30/my.output.root \\
                   data/example_data/ZdZd13TeV/mc23a_mZd60/my.output.root \\
        --output   data/training_ntuples/signal_only_test.parquet

REQUIREMENTS
------------
    pip install uproot awkward numpy pandas pyarrow
"""

import argparse
import os
import sys

import awkward as ak
import numpy as np
import pandas as pd
import uproot


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TREE_PATH = "Nominal/llllTree"

# Kinematic pT thresholds in MeV, matching ZdZdPP_alg.cxx:
#   ls->at(llll.get<int>("l1")).Pt() < 20000.  (leading)
#   ls->at(llll.get<int>("l2")).Pt() < 15000.  (subleading)
#   ls->at(llll.get<int>("l3")).Pt() < 10000.  (subsubleading)
PT1_MIN_MEV = 20_000.0
PT2_MIN_MEV = 15_000.0
PT3_MIN_MEV = 10_000.0

# pdgIdSum codes for the three quadruplet flavour types
PDG_4E    = 44   # e e e e
PDG_2E2MU = 48   # e e μ μ
PDG_4MU   = 52   # μ μ μ μ

# Branches loaded from the tree.  truth_zdzd_avgM is in the OPTIONAL set
# because background files may not always carry a meaningful value.
BRANCHES_REQUIRED = [
    # Event-level scalars
    "mc_channel_number",
    "eventNumber",
    "evtWeight",
    "PileupWeight",
    "passCleaning",
    "passNPV",
    "passTriggers",
    "averageInteractionsPerCrossing",
    # Per-lepton (jagged)
    "l_tlv_pt",
    "l_tlv_eta",
    # Per-dilepton (jagged)
    "ll_tlv_m",
    # Per-quadruplet: selection fields (jagged)
    "llll_charge",
    "llll_dCharge",
    "llll_l1",
    "llll_l2",
    "llll_l3",
    "llll_l4",
    # Per-quadruplet: dilepton index fields (jagged)
    "llll_ll1",
    "llll_ll2",
    "llll_alt_ll1",
    "llll_alt_ll2",
    # Per-quadruplet: feature fields (jagged)
    "llll_tlv_m",
    "llll_avgM",
    "llll_dM",
    "llll_pdgIdSum",
    "llll_min_sf_dR",
    "llll_min_of_dR",
    "llll_vtx_reduced_chi2",
    "llll_max_el_d0Sig",
    "llll_max_mu_d0Sig",
    "llll_nCTorSA",
    "llll_l_isIsolCloseBy",
    "llll_triggerMatched",
    "llll_scaleFactor",
]

BRANCHES_OPTIONAL = [
    "truth_zdzd_avgM",   # present in signal MC; may be 0 in background MC
]

# Columns whose values contain known sentinels – flagged for the training script.
# Confirmed by inspecting both signal and background samples with uproot.
SENTINEL_NOTES = {
    "min_of_dR":        "9999999 for 4e/4mu quads (no opposite-flavour pairs)",
    "vtx_reduced_chi2": "-999 when vertex fit did not converge (~1-4% of events)",
    "max_el_d0Sig":     "0.0 for 4mu quads (no electrons present; computed via std::max with 0.0 fill)",
    "max_mu_d0Sig":     "0.0 for 4e quads (no muons present; computed via std::max with 0.0 fill)",
}


# ---------------------------------------------------------------------------
# Quadruplet selection
# ---------------------------------------------------------------------------

def find_first_passing_quad(charges, dcharges, l1s, l2s, l3s, l_pts):
    """Return the index of the first quadruplet candidate in one event that
    passes SFOS and kinematic pT cuts, or -1 if none pass.

    Parameters
    ----------
    charges, dcharges : 1-D arrays  llll_charge, llll_dCharge for this event
    l1s, l2s, l3s    : 1-D int arrays  lepton indices (pT-ordered) per quad
    l_pts            : 1-D float array  l_tlv_pt for this event (MeV)

    Notes
    -----
    l1/l2/l3/l4 are stored in pT-descending order within each quadruplet
    candidate, following the convention in ZdZdPP_alg.cxx which applies the
    kinematic cuts as pT(l1) >= 20 GeV, pT(l2) >= 15 GeV, pT(l3) >= 10 GeV
    without sorting first.
    """
    for j in range(len(charges)):
        # --- SFOS ---
        if charges[j] != 0 or dcharges[j] != 0:
            continue
        # --- Kinematic pT cuts ---
        if (l_pts[l1s[j]] < PT1_MIN_MEV
                or l_pts[l2s[j]] < PT2_MIN_MEV
                or l_pts[l3s[j]] < PT3_MIN_MEV):
            continue
        return j   # first passing candidate
    return -1


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(root_path, label, tree_path):
    """Read one ROOT file and return a list of flat row dicts, one per
    selected event.

    Parameters
    ----------
    root_path : str   path to the ZdZd13TeV output ROOT file
    label     : int   1 = signal, 0 = background
    tree_path : str   path to the TTree within the file
    """
    print(f"\n  File  : {root_path}")
    print(f"  Label : {'signal (1)' if label == 1 else 'background (0)'}")

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    with uproot.open(root_path) as f:
        if tree_path not in f:
            raise KeyError(
                f"Tree '{tree_path}' not found in {root_path}.\n"
                f"Available keys: {[k for k in f.keys() if 'Tree' in k]}"
            )
        tree = f[tree_path]
        n_total = tree.num_entries
        print(f"  Entries in tree: {n_total:,}")

        # Check for missing required branches
        available = set(tree.keys())
        missing_req = [b for b in BRANCHES_REQUIRED if b not in available]
        if missing_req:
            raise KeyError(
                f"Required branches not found in {root_path}: {missing_req}"
            )

        # Load optional branches where present
        branches_to_load = BRANCHES_REQUIRED + [
            b for b in BRANCHES_OPTIONAL if b in available
        ]
        missing_opt = [b for b in BRANCHES_OPTIONAL if b not in available]
        if missing_opt:
            print(f"  Note: optional branches absent, defaulting to 0: "
                  f"{missing_opt}")

        data = tree.arrays(branches_to_load)

    # -------------------------------------------------------------------------
    # Event-level preselection
    # -------------------------------------------------------------------------
    evt_mask = (
        data["passCleaning"]
        & data["passNPV"]
        & (data["passTriggers"] != 0)
    )
    n_presel = int(ak.sum(evt_mask))
    print(f"  Pass preselection (clean + NPV + trigger): "
          f"{n_presel:,} / {n_total:,} ({100*n_presel/n_total:.1f}%)")

    d = data[evt_mask]   # work only with preselected events from here on

    # -------------------------------------------------------------------------
    # Convert jagged arrays to Python lists once, before the event loop.
    # This is significantly faster than calling ak.to_numpy() per event.
    # -------------------------------------------------------------------------
    has_truth_m = "truth_zdzd_avgM" in d.fields

    # Selection arrays (needed in find_first_passing_quad)
    charges   = d["llll_charge"].tolist()
    dcharges  = d["llll_dCharge"].tolist()
    l1s_sel   = d["llll_l1"].tolist()
    l2s_sel   = d["llll_l2"].tolist()
    l3s_sel   = d["llll_l3"].tolist()
    lpts      = d["l_tlv_pt"].tolist()

    # Feature arrays (per-quadruplet)
    l4s       = d["llll_l4"].tolist()
    letas     = d["llll_l_isIsolCloseBy"].tolist()   # reused name below; alias set per row
    l_etas    = d["l_tlv_eta"].tolist()
    ll_ms     = d["ll_tlv_m"].tolist()
    ll1s      = d["llll_ll1"].tolist()
    ll2s      = d["llll_ll2"].tolist()
    alt_ll1s  = d["llll_alt_ll1"].tolist()
    alt_ll2s  = d["llll_alt_ll2"].tolist()
    m4ls      = d["llll_tlv_m"].tolist()
    avgMs     = d["llll_avgM"].tolist()
    dMs       = d["llll_dM"].tolist()
    pdgsums   = d["llll_pdgIdSum"].tolist()
    sf_dRs    = d["llll_min_sf_dR"].tolist()
    of_dRs    = d["llll_min_of_dR"].tolist()
    chi2s     = d["llll_vtx_reduced_chi2"].tolist()
    el_d0s    = d["llll_max_el_d0Sig"].tolist()
    mu_d0s    = d["llll_max_mu_d0Sig"].tolist()
    n_ctsa    = d["llll_nCTorSA"].tolist()
    isol      = d["llll_l_isIsolCloseBy"].tolist()
    trig_m    = d["llll_triggerMatched"].tolist()
    sfs       = d["llll_scaleFactor"].tolist()

    # Scalar arrays (event-level)
    mc_chans  = ak.to_numpy(d["mc_channel_number"]).tolist()
    evt_nums  = ak.to_numpy(d["eventNumber"]).tolist()
    evt_wts   = ak.to_numpy(d["evtWeight"]).tolist()
    pu_wts    = ak.to_numpy(d["PileupWeight"]).tolist()
    mus       = ak.to_numpy(d["averageInteractionsPerCrossing"]).tolist()
    truth_ms  = (ak.to_numpy(d["truth_zdzd_avgM"]).tolist()
                 if has_truth_m else [0.0] * len(d))

    # -------------------------------------------------------------------------
    # Quadruplet selection + feature extraction loop
    # -------------------------------------------------------------------------
    # For each event take the first quadruplet candidate passing SFOS + pT cuts.
    # A Python loop is used for clarity; for O(millions) of events this could
    # be vectorised with awkward-array operations.

    rows       = []
    n_no_cands = 0
    n_no_pass  = 0
    n_selected = 0

    for i in range(len(d)):
        if len(charges[i]) == 0:
            n_no_cands += 1
            continue

        q = find_first_passing_quad(
            charges[i], dcharges[i],
            l1s_sel[i], l2s_sel[i], l3s_sel[i],
            lpts[i],
        )
        if q < 0:
            n_no_pass += 1
            continue

        # Lepton indices (pT-ordered: l1 = leading, l4 = trailing)
        l1 = l1s_sel[i][q]
        l2 = l2s_sel[i][q]
        l3 = l3s_sel[i][q]
        l4 = l4s[i][q]

        # Dilepton masses via index lookup.
        # mab is defined as the larger of the two primary dilepton masses and
        # mcd as the smaller, mirroring the make_masses() convention in
        # ZdZd13TeV_alg.py.  The llll_ll1/ll2 indices do not guarantee this
        # ordering on their own.
        lpt_i  = lpts[i]
        leta_i = l_etas[i]
        ll_m_i = ll_ms[i]
        m_ll1 = ll_m_i[ll1s[i][q]]
        m_ll2 = ll_m_i[ll2s[i][q]]
        mab = m_ll1 if m_ll1 >= m_ll2 else m_ll2   # leading dilepton mass
        mcd = m_ll2 if m_ll1 >= m_ll2 else m_ll1   # subleading dilepton mass
        mad = ll_m_i[alt_ll1s[i][q]]
        mbc = ll_m_i[alt_ll2s[i][q]]
        mcd_over_mab = (mcd / mab) if mab > 0.0 else float("nan")

        # Total event weight: evtWeight * PileupWeight * per-quad scaleFactor
        weight = evt_wts[i] * pu_wts[i] * sfs[i][q]

        pdg_sum = pdgsums[i][q]

        rows.append({
            # --- Metadata / labels ---
            "label":             label,
            "mc_channel_number": mc_chans[i],
            "eventNumber":       evt_nums[i],
            "truth_zdzd_avgM":   truth_ms[i],
            "evtWeight_total":   weight,
            # --- Event-level features ---
            "mu":                mus[i],
            # --- Quadruplet mass features (MeV) ---
            "m_4l":              m4ls[i][q],
            "avgM":              avgMs[i][q],
            "dM":                dMs[i][q],
            "mab":               mab,
            "mcd":               mcd,
            "mad":               mad,
            "mbc":               mbc,
            "mcd_over_mab":      mcd_over_mab,
            # --- Angular / vertex features ---
            "min_sf_dR":         sf_dRs[i][q],
            "min_of_dR":         of_dRs[i][q],    # *** SENTINEL: 9999999 for 4e/4mu ***
            "vtx_reduced_chi2":  chi2s[i][q],      # *** SENTINEL: -999 on failed fit ***
            # --- Impact parameter features ---
            "max_el_d0Sig":      el_d0s[i][q],     # *** SENTINEL: 0.0 for 4mu (no electrons) ***
            "max_mu_d0Sig":      mu_d0s[i][q],     # *** SENTINEL: 0.0 for 4e  (no muons)    ***
            # --- Lepton quality / isolation features ---
            "nCTorSA":           n_ctsa[i][q],
            "l_isIsolCloseBy":   isol[i][q],
            "triggerMatched":    trig_m[i][q],
            # --- Flavour one-hot encoding (pdgIdSum: 44=4e, 48=2e2mu, 52=4mu) ---
            "is_4e":             int(pdg_sum == PDG_4E),
            "is_2e2mu":          int(pdg_sum == PDG_2E2MU),
            "is_4mu":            int(pdg_sum == PDG_4MU),
            # --- Lepton kinematics (MeV; l1=leading → l4=trailing) ---
            "pT_l1":             lpt_i[l1],
            "pT_l2":             lpt_i[l2],
            "pT_l3":             lpt_i[l3],
            "pT_l4":             lpt_i[l4],
            "eta_l1":            leta_i[l1],
            "eta_l2":            leta_i[l2],
            "eta_l3":            leta_i[l3],
            "eta_l4":            leta_i[l4],
        })
        n_selected += 1

    print(f"  No quadruplet candidates: {n_no_cands:,}")
    print(f"  Candidates fail SFOS/pT:  {n_no_pass:,}")
    print(f"  Selected events:          {n_selected:,}")
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

# Columns that are clean BDT features (no sentinel issues)
FEATURE_COLS = [
    "mu",
    "m_4l", "avgM", "dM", "mab", "mcd", "mad", "mbc", "mcd_over_mab",
    "min_sf_dR",
    "nCTorSA", "l_isIsolCloseBy", "triggerMatched",
    "is_4e", "is_2e2mu", "is_4mu",
    "pT_l1", "pT_l2", "pT_l3", "pT_l4",
    "eta_l1", "eta_l2", "eta_l3", "eta_l4",
]


def print_summary(df):
    """Print a human-readable summary of the output DataFrame."""
    sep = "=" * 64
    print(f"\n{sep}")
    print("Output DataFrame summary")
    print(sep)
    print(f"  Total rows    : {len(df):,}")
    print(f"  Signal rows   : {int((df['label'] == 1).sum()):,}")
    print(f"  Background rows: {int((df['label'] == 0).sum()):,}")
    if (df["label"] == 1).any():
        # Round to nearest 1000 MeV (1 GeV) to collapse floating-point jitter
        mzd_vals = sorted(
            df.loc[df["label"] == 1, "truth_zdzd_avgM"]
            .round(-3).unique().astype(int).tolist()
        )
        print(f"  Signal mZd values (MeV): {mzd_vals}")
    chan_nums = sorted(df["mc_channel_number"].unique().tolist())
    print(f"  MC channel numbers: {chan_nums}")

    print(f"\n  BDT feature columns (MeV; * = sentinel, handle before TMVA):")
    all_feat = FEATURE_COLS + list(SENTINEL_NOTES.keys())
    col_w = max(len(c) for c in all_feat) + 2
    for col in FEATURE_COLS + list(SENTINEL_NOTES.keys()):
        if col not in df.columns:
            continue
        s = df[col]
        sentinel_flag = "  *" if col in SENTINEL_NOTES else ""
        print(f"    {col:<{col_w}} n={s.notna().sum():>6,}  "
              f"min={s.min():>10.4g}  mean={s.mean():>10.4g}  "
              f"max={s.max():>10.4g}{sentinel_flag}")

    if SENTINEL_NOTES:
        print(f"\n  * Sentinel value notes:")
        for col, note in SENTINEL_NOTES.items():
            if col in df.columns:
                print(f"    {col}: {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 1 of the TMVA pipeline: convert ZdZd13TeV ROOT ntuples "
            "to a flat Parquet file for BDT training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--signal", nargs="+", metavar="FILE",
        help="One or more signal ROOT files (H→ZdZd→4ℓ MC).",
    )
    p.add_argument(
        "--background", nargs="+", metavar="FILE", default=[],
        help="One or more background ROOT files (e.g. ZZ*→4ℓ MC).",
    )
    p.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output Parquet file path.",
    )
    p.add_argument(
        "--tree", default=TREE_PATH,
        help=f"TTree path within the ROOT file (default: {TREE_PATH}).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.signal and not args.background:
        print("ERROR: at least one --signal or --background file is required.",
              file=sys.stderr)
        sys.exit(1)

    all_rows = []

    if args.signal:
        print(f"\nProcessing {len(args.signal)} signal file(s):")
        for path in args.signal:
            all_rows.extend(process_file(path, label=1,
                                         tree_path=args.tree))

    if args.background:
        print(f"\nProcessing {len(args.background)} background file(s):")
        for path in args.background:
            all_rows.extend(process_file(path, label=0,
                                         tree_path=args.tree))

    if not all_rows:
        print("ERROR: no events survived selection. "
              "Check input files and selection cuts.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print_summary(df)

    # Write output, creating intermediate directories if needed
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output, index=False)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nWritten: {args.output}  ({size_kb:.0f} kB, {len(df):,} rows, "
          f"{len(df.columns)} columns)")


if __name__ == "__main__":
    main()
