#!/usr/bin/env python3
"""
train_bdt.py
============
Stage 2 of the TMVA pipeline: read the Parquet file from Stage 1, split into
train/test sets, write ROOT TTrees, and train a TMVA BDT classifier.

DECISIONS
---------
Features    : 24 scalar variables (see BDT_VARIABLES).  truth_zdzd_avgM is
              excluded (unavailable at evaluation time on real data).  The four
              sentinel-bearing columns (min_of_dR, vtx_reduced_chi2,
              max_el_d0Sig, max_mu_d0Sig) are excluded per Stage 1 design.
Weights     : evtWeight_total (= evtWeight × PileupWeight × llll_scaleFactor)
              is passed as the per-event weight.  Negative weights (~0.6% of
              the H→ZZ*→4ℓ background sample) are passed to TMVA as-is.
Normalisation: NormMode=EqualNumEvents — TMVA rescales signal and background
              total weights to be equal before training.
Split       : 50/50 train/test using eventNumber parity (even → train,
              odd → test).  This is deterministic and reproducible without a
              random seed, and is a standard HEP convention.
Trees       : Signal and background are written to separate ROOT TTrees for
              training and testing (four trees total).  TMVA uses these
              pre-labelled trees directly, bypassing its own internal split.

PSEUDO-BACKGROUND NOTE
----------------------
When no real background is available, pass --pseudo-background with a Parquet
file containing multiple mZd samples, and specify which MC channel number(s)
should be relabelled as pseudo-background (label=0).  The remaining events
keep label=1 (signal).  Intended for pipeline testing only.

Example: use mZd60 (channel 561517) as pseudo-background from a signal-only
Parquet that also contains mZd30 (channel 561511):

    python3 train_bdt.py \\
        --input  data/training_ntuples/signal_only_test.parquet \\
        --pseudo-background --pseudo-bkg-channels 561517 \\
        --output-dir tmva_ZdZd_R3

Note: truth_zdzd_avgM is NOT used for this split.  The mc_channel_number
branch is used instead because truth_zdzd_avgM is only filled in the ROOT
tree for channel numbers that appear in ZdZdSamples in analysisJobOptions_run3.py.
Channel 561517 (mZd60) is absent from that list, so its truth_zdzd_avgM = 0.

USAGE
-----
Standard (real signal + background):
    python3 train_bdt.py \\
        --input  data/training_ntuples/sig_mZd30_bkg_HZZ4l.parquet \\
        --output-dir tmva_ZdZd_R3

Pseudo-background (two mZd samples, no real background):
    python3 train_bdt.py \\
        --input  data/training_ntuples/signal_only_test.parquet \\
        --pseudo-background --pseudo-bkg-channels 561517 \\
        --output-dir tmva_ZdZd_R3

Re-run training only (skip TTree writing, reuse existing trees file):
    python3 train_bdt.py \\
        --input  data/training_ntuples/sig_mZd30_bkg_HZZ4l.parquet \\
        --output-dir tmva_ZdZd_R3 \\
        --reuse-trees

REQUIREMENTS
------------
    ROOT >= 6.12 with TMVA, PyROOT, pandas, numpy, pyarrow
    On lxplus:  source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    (or the LCG release used by this analysis)

OUTPUT
------
    <output-dir>/weights/TMVAClassification_BDT.weights.xml  ← trained BDT
    <output-dir>/TMVAClassification.root                     ← ROC, overtrain plots
    <output-dir>/training_trees.root                         ← intermediate TTrees
"""

import argparse
import array
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# BDT input variables
# ---------------------------------------------------------------------------
# 24 scalar features passed to TMVA.
# Sentinel columns are excluded:
#   min_of_dR        (9 999 999 for 4e/4mu — no opposite-flavour pairs)
#   vtx_reduced_chi2 (-999 when vertex fit did not converge)
#   max_el_d0Sig     (0.0 for 4mu — no electrons)
#   max_mu_d0Sig     (0.0 for 4e  — no muons)
BDT_VARIABLES = [
    # Pile-up
    "mu",
    # Four-lepton and dilepton mass features (MeV)
    "m_4l", "avgM", "dM", "mab", "mcd", "mad", "mbc", "mcd_over_mab",
    # Angular separation
    "min_sf_dR",
    # Lepton quality / isolation / trigger
    "nCTorSA", "l_isIsolCloseBy", "triggerMatched",
    # Quadruplet flavour one-hot (pdgIdSum: 44=4e, 48=2e2μ, 52=4μ)
    "is_4e", "is_2e2mu", "is_4mu",
    # Lepton kinematics (MeV, pT-descending order l1→l4)
    "pT_l1", "pT_l2", "pT_l3", "pT_l4",
    "eta_l1", "eta_l2", "eta_l3", "eta_l4",
]

WEIGHT_COL = "evtWeight_total"

# ---------------------------------------------------------------------------
# TMVA BDT configuration
# ---------------------------------------------------------------------------
# These are the starting default hyperparameters.  Tune after inspecting the
# initial ROC curve and overtraining check (Kolmogorov-Smirnov test).
BDT_HYPERPARAMS = {
    "NTrees":           850,    # number of trees in the ensemble
    "MinNodeSize":      "5%",   # min fraction of training events per leaf node
    "MaxDepth":         3,      # max tree depth (controls complexity)
    "BoostType":        "AdaBoost",
    "AdaBoostBeta":     0.5,    # learning rate
    "UseBaggedBoost":   "False",
    "SeparationType":   "GiniIndex",
    "nCuts":            20,     # scan points per variable for split optimisation
    "PruneMethod":      "NoPruning",
}

# TMVA Factory options
FACTORY_OPTIONS = (
    "!V:"                         # suppress verbose output
    "!Silent:"                    # suppress silent mode (keep progress)
    "Color:"                      # coloured terminal output
    "DrawProgressBar:"            # show training progress bar
    "Transformations=I:"          # Identity — no variable transformations applied
    "AnalysisType=Classification"
)

# DataLoader PrepareTrainingAndTestTree options.
# NormMode=EqualNumEvents: TMVA scales signal and background total weights
# to be equal before training, regardless of the raw event counts.
# We supply pre-split kTraining/kTesting trees so no internal split is needed.
PREPARE_OPTIONS = "NormMode=EqualNumEvents:!V"


def _bdt_option_string():
    """Build the colon-separated TMVA BDT method option string."""
    parts = ["!H", "!V"]   # suppress HTML output, suppress verbose
    for k, v in BDT_HYPERPARAMS.items():
        parts.append(f"{k}={v}")
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_train_test(df):
    """Return (train_df, test_df) using eventNumber parity.

    Even eventNumber → training.  Odd → testing.
    This gives a reproducible, deterministic 50/50 split without a random seed.
    """
    train_mask = (df["eventNumber"].values % 2) == 0
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ROOT TTree writer
# ---------------------------------------------------------------------------

def df_to_ttree(df, tree_name, tree_title, feature_cols, weight_col, tfile):
    """Write *df* as a new ROOT TTree inside the open TFile *tfile*.

    All branches are written as 32-bit floats (Float_t / 'F').  The weight
    branch is named exactly *weight_col* so it matches SetWeightExpression.

    Parameters
    ----------
    df           : pd.DataFrame
    tree_name    : str          TTree name
    tree_title   : str          TTree title
    feature_cols : list[str]    columns to write as input variable branches
    weight_col   : str          column to write as the per-event weight branch
    tfile        : ROOT.TFile   open writable file; caller retains ownership

    Returns
    -------
    ROOT.TTree  (keep a Python reference until the file is closed)
    """
    import ROOT   # deferred so the module can be imported without ROOT present

    all_cols = feature_cols + [weight_col]

    # Pre-convert to float32 numpy arrays for fast indexed access
    data = {col: df[col].values.astype("float32") for col in all_cols}
    n_events = len(df)

    tree = ROOT.TTree(tree_name, tree_title)

    # One-element C-array buffers for TTree.Branch
    bufs = {}
    for col in all_cols:
        bufs[col] = array.array("f", [0.0])
        tree.Branch(col, bufs[col], f"{col}/F")

    for i in range(n_events):
        for col in all_cols:
            bufs[col][0] = float(data[col][i])
        tree.Fill()

    tfile.cd()
    tree.Write()
    print(f"    {tree_name:<20} {n_events:>7,} events")
    return tree


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_training(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)    # suppress GUI windows

    # ── Load Parquet ────────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # ── Pseudo-background relabelling ────────────────────────────────────────
    if args.pseudo_background:
        if not args.pseudo_bkg_channels:
            sys.exit(
                "ERROR: --pseudo-background requires --pseudo-bkg-channels "
                "(one or more MC channel numbers to relabel as background).\n"
                "Example: --pseudo-bkg-channels 561517"
            )
        n_existing_bkg = int((df["label"] == 0).sum())
        if n_existing_bkg > 0 and not args.force:
            sys.exit(
                "ERROR: --pseudo-background requested but Parquet already contains "
                f"{n_existing_bkg} label=0 events.  Use --force to override."
            )
        bkg_channels = set(args.pseudo_bkg_channels)
        # truth_zdzd_avgM is NOT used: it is only filled for channels listed in
        # ZdZdSamples (analysisJobOptions_run3.py).  Use mc_channel_number instead.
        bkg_mask = df["mc_channel_number"].isin(bkg_channels)
        n_no_match = int((~bkg_mask).sum()) - int((df["label"] == 1).sum())
        df["label"] = (~bkg_mask).astype(int)   # 1 = signal, 0 = pseudo-bkg
        n_sig_new = int((df["label"] == 1).sum())
        n_bkg_new = int((df["label"] == 0).sum())
        present = sorted(df.loc[bkg_mask, "mc_channel_number"].unique().tolist())
        print(
            f"\n  Pseudo-background mode:\n"
            f"    pseudo-bkg channels (requested) : {sorted(bkg_channels)}\n"
            f"    pseudo-bkg channels (found)     : {present}\n"
            f"    signal (label=1)                : {n_sig_new:,}\n"
            f"    pseudo-bkg (label=0)            : {n_bkg_new:,}"
        )
        if n_bkg_new == 0:
            sys.exit(
                "ERROR: No events matched --pseudo-bkg-channels.  "
                f"Available mc_channel_number values: "
                f"{sorted(df['mc_channel_number'].unique().tolist())}"
            )

    # ── Validate columns ─────────────────────────────────────────────────────
    required = BDT_VARIABLES + [WEIGHT_COL, "label", "eventNumber"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns in Parquet: {missing}")

    n_sig = int((df["label"] == 1).sum())
    n_bkg = int((df["label"] == 0).sum())
    if n_sig == 0 or n_bkg == 0:
        sys.exit("ERROR: Need both signal (label=1) and background (label=0) events.")

    n_neg = int((df[WEIGHT_COL] < 0).sum())
    sep = "─" * 60
    print(f"\n  {sep}")
    print(f"  Signal events      : {n_sig:>8,}")
    print(f"  Background events  : {n_bkg:>8,}")
    print(f"  Negative weights   : {n_neg:>8,}  ({100*n_neg/len(df):.2f}%)")
    print(f"  {sep}")

    # ── Train / test split ───────────────────────────────────────────────────
    sig_df  = df[df["label"] == 1].copy()
    bkg_df  = df[df["label"] == 0].copy()

    sig_train, sig_test = split_train_test(sig_df)
    bkg_train, bkg_test = split_train_test(bkg_df)

    print(f"\n  Train: signal={len(sig_train):,}  background={len(bkg_train):,}")
    print(f"  Test:  signal={len(sig_test):,}  background={len(bkg_test):,}")

    # Effective training weights (informational; NormMode will rescale these)
    sig_wt = sig_train[WEIGHT_COL].sum()
    bkg_wt = bkg_train[WEIGHT_COL].sum()
    print(f"\n  Sum of train weights (before NormMode rescaling):")
    print(f"    signal     : {sig_wt:.4g}")
    print(f"    background : {bkg_wt:.4g}")
    print(f"    ratio sig/bkg : {sig_wt/bkg_wt:.3f}")

    # ── Set up output paths ──────────────────────────────────────────────────
    out_dir     = os.path.abspath(args.output_dir)
    weights_dir = os.path.join(out_dir, "weights")
    trees_path  = os.path.join(out_dir, "training_trees.root")
    tmva_path   = os.path.join(out_dir, "TMVAClassification.root")
    os.makedirs(weights_dir, exist_ok=True)

    # ── Write ROOT TTrees (unless reusing) ───────────────────────────────────
    if args.reuse_trees:
        if not os.path.exists(trees_path):
            sys.exit(f"ERROR: --reuse-trees set but {trees_path} not found.")
        print(f"\nReusing existing TTree file: {trees_path}")
    else:
        print(f"\nWriting ROOT TTrees to: {trees_path}")
        trees_file_write = ROOT.TFile(trees_path, "RECREATE")
        df_to_ttree(sig_train, "sig_train", "Signal training",
                    BDT_VARIABLES, WEIGHT_COL, trees_file_write)
        df_to_ttree(sig_test,  "sig_test",  "Signal test",
                    BDT_VARIABLES, WEIGHT_COL, trees_file_write)
        df_to_ttree(bkg_train, "bkg_train", "Background training",
                    BDT_VARIABLES, WEIGHT_COL, trees_file_write)
        df_to_ttree(bkg_test,  "bkg_test",  "Background test",
                    BDT_VARIABLES, WEIGHT_COL, trees_file_write)
        trees_file_write.Close()

    # ── Open trees for TMVA ──────────────────────────────────────────────────
    trees_file = ROOT.TFile(trees_path, "READ")
    if trees_file.IsZombie():
        sys.exit(f"ERROR: Could not open {trees_path}")

    t_sig_train = trees_file.Get("sig_train")
    t_sig_test  = trees_file.Get("sig_test")
    t_bkg_train = trees_file.Get("bkg_train")
    t_bkg_test  = trees_file.Get("bkg_test")

    for name, tree in [("sig_train", t_sig_train), ("sig_test",  t_sig_test),
                        ("bkg_train", t_bkg_train), ("bkg_test",  t_bkg_test)]:
        if not tree:
            sys.exit(f"ERROR: Could not retrieve TTree '{name}' from {trees_path}")

    # ── TMVA Factory and DataLoader ──────────────────────────────────────────
    bdt_opts = _bdt_option_string()
    print(f"\nRunning TMVA training")
    print(f"  TMVA output    : {tmva_path}")
    print(f"  Weights dir    : {weights_dir}")
    print(f"  Variables      : {len(BDT_VARIABLES)}")
    print(f"  Hyperparameters:")
    for k, v in BDT_HYPERPARAMS.items():
        print(f"    {k:<22} = {v}")
    print(f"  NormMode       : EqualNumEvents")

    tmva_out = ROOT.TFile(tmva_path, "RECREATE")

    # TMVA writes its weights XML relative to the DataLoader path.
    # Passing the absolute weights_dir makes the output location explicit.
    factory    = ROOT.TMVA.Factory("TMVAClassification", tmva_out, FACTORY_OPTIONS)
    dataloader = ROOT.TMVA.DataLoader(weights_dir)

    # Register input variables
    for var in BDT_VARIABLES:
        dataloader.AddVariable(var, "F")

    # Add pre-split trees — TMVA respects kTraining/kTesting labels
    dataloader.AddSignalTree    (t_sig_train, 1.0, ROOT.TMVA.Types.kTraining)
    dataloader.AddSignalTree    (t_sig_test,  1.0, ROOT.TMVA.Types.kTesting)
    dataloader.AddBackgroundTree(t_bkg_train, 1.0, ROOT.TMVA.Types.kTraining)
    dataloader.AddBackgroundTree(t_bkg_test,  1.0, ROOT.TMVA.Types.kTesting)

    # Per-event weights
    dataloader.SetSignalWeightExpression    (WEIGHT_COL)
    dataloader.SetBackgroundWeightExpression(WEIGHT_COL)

    # NormMode=EqualNumEvents: scale signal and background total weights equally.
    # The four nTrain/nTest=0 options tell TMVA to use all provided events.
    dataloader.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""), PREPARE_OPTIONS)

    # Book BDT method
    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", bdt_opts)

    # Train → Test → Evaluate
    print("\n" + "=" * 60)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    print("=" * 60)

    tmva_out.Close()
    trees_file.Close()

    # ── Summary ──────────────────────────────────────────────────────────────
    xml_path = os.path.join(weights_dir, "TMVAClassification_BDT.weights.xml")
    print(f"\nTraining complete.")
    print(f"  Weights file   : {xml_path}")
    print(f"  TMVA ROOT file : {tmva_path}")
    print(f"\nTo view results interactively:")
    print(f"  root -l '{tmva_path}'")
    print(f"  // then in the ROOT prompt:")
    print(f"  TMVA::TMVAGui(\"{tmva_path}\")")
    print(f"\nKey plots to check:")
    print(f"  1. ROC curve (signal efficiency vs background rejection)")
    print(f"  2. BDT response overtraining check (train vs test KS test)")
    print(f"  3. Variable importance / ranking")
    print(f"  4. Correlation matrices for signal and background")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: train TMVA BDT from Stage 1 Parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", required=True, metavar="PARQUET",
        help="Input Parquet file from make_training_ntuples.py.",
    )
    p.add_argument(
        "--output-dir", default="tmva_ZdZd_R3", metavar="DIR",
        help="Output directory for weights/ and TMVAClassification.root. "
             "Default: tmva_ZdZd_R3",
    )
    p.add_argument(
        "--pseudo-background", action="store_true",
        help="Relabel events by mc_channel_number for pipeline testing when no real "
             "background is available.  Requires --pseudo-bkg-channels.",
    )
    p.add_argument(
        "--pseudo-bkg-channels", nargs="+", type=int, default=[], metavar="CHANNEL",
        help="MC channel number(s) to relabel as background (label=0) in "
             "--pseudo-background mode.  Example: --pseudo-bkg-channels 561517",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Allow --pseudo-background even when label=0 events are already present.",
    )
    p.add_argument(
        "--reuse-trees", action="store_true",
        help="Skip TTree writing and reuse an existing training_trees.root in "
             "--output-dir.  Useful when re-running with different hyperparameters.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        sys.exit(f"ERROR: Input file not found: {args.input}")
    run_training(args)


if __name__ == "__main__":
    main()
