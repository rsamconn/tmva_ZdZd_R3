#!/usr/bin/env python3
"""
train_bdt.py
============
Stage 2 of the TMVA pipeline: read the Parquet file from Stage 1, apply the
per-sample normalisation, split into train/test sets, write ROOT TTrees, and
train a TMVA BDT classifier.

DECISIONS
---------
Features    : 24 scalar variables (see BDT_VARIABLES).  truth_zdzd_avgM is
              excluded (unavailable at evaluation time on real data).  The four
              sentinel-bearing columns (min_of_dR, vtx_reduced_chi2,
              max_el_d0Sig, max_mu_d0Sig) are excluded per Stage 1 design.
Weights     : see WEIGHTING below.
Normalisation: NormMode=EqualNumEvents - TMVA rescales the signal and background
              CLASS total weights to be equal before training.  It applies one
              scale per class, so it does not disturb the composition WITHIN the
              background class; that composition is this script's job.
Split       : 50/50 train/test using eventNumber parity (even -> train,
              odd -> test).  This is deterministic and reproducible without a
              random seed, and is a standard HEP convention.
Trees       : Signal is written as one pair of TTrees (train/test).  Background
              is written as ONE PAIR PER PROCESS, and each pair is registered
              with TMVA separately.  TMVA accepts several trees per class and
              prints each tree's weight sum, which is the cheapest available
              check that the background composition came out as intended.
Spectator   : process_id is registered as a TMVA spectator, so it is written
              into TrainTree/TestTree without entering the training.  That is
              the only way the per-process breakdown survives into the trees
              Stage 3 reads - otherwise they carry nothing but classID.

WEIGHTING
---------
Stage 1 writes three per-event columns: `evtWeight_total` (the raw MC weight),
`scale_d` (= L * sigma * k * eps_filt / SumW_total for that sample) and
`sampling_fraction` (the fraction of that sample's entries that were read).
`--weighting` decides how they are combined:

  normalised (default)
      w = evtWeight_total * scale_d / sampling_fraction
      Each process enters in proportion to its expected yield at the campaign
      luminosity, with the subsampling divided out.  This is the mode that
      gives the BDT a physical background composition.  Use it unless a
      normalisation input is missing.

  equal-process
      Each background process is rescaled to the same total weight.  An interim
      mode for when a scale_d input is unavailable or distrusted; it trains a
      classifier that treats the four processes as equally important rather
      than as they actually occur.

  raw
      w = evtWeight_total, i.e. the pre-2026-10 behaviour.  With more than one
      background process this makes the background mix a function of MC
      generation statistics and generator weight scales rather than of physics.
      Kept only for reproducing earlier runs.

In every mode the signal class is rescaled as a whole by NormMode, so the
absolute signal normalisation is irrelevant; what matters is the RELATIVE
weight of the signal samples.  With --signal-equalise each signal process
(i.e. each mZd point) is given the same total weight; by default they enter in
proportion to their selection acceptance.

NEGATIVE WEIGHTS
----------------
Three of the four irreducible processes are Sherpa (701040, 701274 and the
Tribosons DSIDs) and carry a real negative-weight fraction, much larger than
the 0.22% seen from Powheg H_ZZ_4l alone.  AdaBoost is not stable with negative
weights, so --neg-weights defaults to `ignore`, which sets TMVA's
IgnoreNegWeightsInTraining.  The per-process table printed below reports
N_eff = (sum w)^2 / sum w^2 so the statistical cost is visible: a few large
negative weights can gut N_eff without changing the event count.

PSEUDO-BACKGROUND NOTE
----------------------
When no real background is available, pass --pseudo-background with a Parquet
file containing multiple mZd samples, and specify which MC channel number(s)
should be relabelled as pseudo-background (label=0).  The remaining events
keep label=1 (signal).  Intended for pipeline testing only.

Note: truth_zdzd_avgM is NOT used for this split.  The mc_channel_number
branch is used instead because truth_zdzd_avgM is only filled in the ROOT
tree for channel numbers that appear in ZdZdSamples in analysisJobOptions_run3.py.

USAGE
-----
Multi-process, normalised (the default):
    python3 train_bdt.py \\
        --input  data/training_ntuples/sig3_irreducible_mc23a.parquet \\
        --output-dir tmva_ZdZd_R3

Interim equal-weight-per-process:
    python3 train_bdt.py \\
        --input  data/training_ntuples/sig3_irreducible_mc23a.parquet \\
        --weighting equal-process --output-dir tmva_ZdZd_R3

Re-run training only (skip TTree writing, reuse existing trees file):
    python3 train_bdt.py \\
        --input  data/training_ntuples/sig3_irreducible_mc23a.parquet \\
        --output-dir tmva_ZdZd_R3 --reuse-trees

REQUIREMENTS
------------
    ROOT >= 6.12 with TMVA, PyROOT, pandas, numpy, pyarrow
    On lxplus:  run `setupATLAS` beforehand; this script performs no setup of
    its own.

OUTPUT
------
    <output-dir>/dataset/weights/TMVAClassification_BDT.weights.xml  <- trained BDT
    <output-dir>/TMVAClassification.root                     <- ROC, overtrain plots
    <output-dir>/training_trees.root                         <- intermediate TTrees
    <output-dir>/process_map.json                            <- process_id -> name
"""

import argparse
import array
import json
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# BDT input variables
# ---------------------------------------------------------------------------
# 24 scalar features passed to TMVA.
# Sentinel columns are excluded:
#   min_of_dR        (9 999 999 for 4e/4mu - no opposite-flavour pairs)
#   vtx_reduced_chi2 (-999 when vertex fit did not converge)
#   max_el_d0Sig     (0.0 for 4mu - no electrons)
#   max_mu_d0Sig     (0.0 for 4e  - no muons)
BDT_VARIABLES = [
    # Pile-up
    "mu",
    # Four-lepton and dilepton mass features (MeV)
    "m_4l", "avgM", "dM", "mab", "mcd", "mad", "mbc", "mcd_over_mab",
    # Angular separation
    "min_sf_dR",
    # Lepton quality / isolation / trigger
    "nCTorSA", "l_isIsolCloseBy", "triggerMatched",
    # Quadruplet flavour one-hot (pdgIdSum: 44=4e, 48=2e2mu, 52=4mu)
    "is_4e", "is_2e2mu", "is_4mu",
    # Lepton kinematics (MeV, pT-descending order l1->l4)
    "pT_l1", "pT_l2", "pT_l3", "pT_l4",
    "eta_l1", "eta_l2", "eta_l3", "eta_l4",
]

RAW_WEIGHT_COL = "evtWeight_total"   # as written by Stage 1
WEIGHT_COL = "train_weight"          # what TMVA is given
SPECTATORS = ["process_id"]

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
    "Transformations=I:"          # Identity - no variable transformations applied
    "AnalysisType=Classification"
)

# DataLoader PrepareTrainingAndTestTree options.
# NormMode=EqualNumEvents: TMVA scales the signal and background CLASS total
# weights to be equal before training.  We supply pre-split kTraining/kTesting
# trees so no internal split is needed.
PREPARE_OPTIONS = "NormMode=EqualNumEvents:!V"


def bdt_option_string(ignore_neg_weights):
    """Build the colon-separated TMVA BDT method option string."""
    parts = ["!H", "!V"]   # suppress HTML output, suppress verbose
    for k, v in BDT_HYPERPARAMS.items():
        parts.append(f"{k}={v}")
    parts.append(f"IgnoreNegWeightsInTraining={'True' if ignore_neg_weights else 'False'}")
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Weighting  (pure numeric - unit-testable without ROOT)
# ---------------------------------------------------------------------------

def _class_process_sums(df, label):
    """Signed weight sum per process within one class."""
    sel = df["label"] == label
    return df.loc[sel].groupby("process")["_w"].sum()


def compute_train_weights(df, mode, signal_equalise):
    """Return the per-event training weight as a numpy array.

    `df` must carry label, process, evtWeight_total and - for the `normalised`
    mode - scale_d and sampling_fraction.  Returns a float64 array aligned with
    df's rows; does not modify df.
    """
    raw = df[RAW_WEIGHT_COL].to_numpy(dtype=float)

    if mode == "raw":
        w = raw.copy()
    elif mode in ("normalised", "equal-process"):
        missing = [c for c in ("scale_d", "sampling_fraction")
                   if c not in df.columns]
        if missing and mode == "normalised":
            raise KeyError(
                f"--weighting normalised needs column(s) {missing}; the Parquet "
                f"was made by an older Stage 1. Re-run Stage 1 with a manifest, "
                f"or use --weighting raw.")
        if missing:
            w = raw.copy()
        else:
            frac = df["sampling_fraction"].to_numpy(dtype=float)
            if np.any(frac <= 0):
                raise ValueError("sampling_fraction contains non-positive values")
            w = raw * df["scale_d"].to_numpy(dtype=float) / frac
    else:
        raise ValueError(f"unknown weighting mode: {mode}")

    work = pd.DataFrame({"label": df["label"].to_numpy(),
                         "process": df["process"].to_numpy(),
                         "_w": w})

    if mode == "equal-process":
        sums = _class_process_sums(work, 0)
        for proc, total in sums.items():
            if total == 0:
                print(f"  WARNING: process {proc} has zero summed weight; "
                      f"left unscaled under equal-process")
                continue
            w[(work["label"] == 0) & (work["process"] == proc)] /= total

    if signal_equalise:
        sums = _class_process_sums(work, 1)
        for proc, total in sums.items():
            if total == 0:
                print(f"  WARNING: signal process {proc} has zero summed "
                      f"weight; left unscaled under --signal-equalise")
                continue
            w[(work["label"] == 1) & (work["process"] == proc)] /= total

    return w


def process_diagnostics(df, weight_col=WEIGHT_COL):
    """Per-process event counts, weight sums, N_eff and negative fraction."""
    out = []
    for proc, g in df.groupby("process", sort=True):
        w = g[weight_col].to_numpy(dtype=float)
        sw, sw2 = float(w.sum()), float(np.square(w).sum())
        out.append({
            "process": proc,
            "label": int(g["label"].iloc[0]),
            "n": len(g),
            "sum_w": sw,
            "sum_w2": sw2,
            "n_eff": (sw * sw / sw2) if sw2 > 0 else 0.0,
            "frac_neg": float((g[RAW_WEIGHT_COL].to_numpy() < 0).mean()),
        })
    return out


def print_process_diagnostics(rows):
    sep = "-" * 94
    print(f"\n  {sep}")
    print(f"  {'process':<18}{'class':<11}{'events':>10}{'sum w':>15}"
          f"{'sum w^2':>15}{'N_eff':>11}{'neg w':>9}")
    print(f"  {sep}")
    tot_sig = tot_bkg = 0.0
    for r in rows:
        cls = "signal" if r["label"] == 1 else "background"
        if r["label"] == 1:
            tot_sig += r["sum_w"]
        else:
            tot_bkg += r["sum_w"]
        print(f"  {r['process']:<18}{cls:<11}{r['n']:>10,}{r['sum_w']:>15.6g}"
              f"{r['sum_w2']:>15.6g}{r['n_eff']:>11.1f}"
              f"{100 * r['frac_neg']:>8.2f}%")
    print(f"  {sep}")
    print(f"  {'TOTAL signal':<29}{'':>10}{tot_sig:>15.6g}")
    print(f"  {'TOTAL background':<29}{'':>10}{tot_bkg:>15.6g}")
    if tot_bkg != 0:
        print(f"  signal / background (pre-NormMode): {tot_sig / tot_bkg:.4g}")
    print()
    for r in rows:
        if r["label"] == 0 and tot_bkg != 0:
            print(f"    {r['process']:<18} "
                  f"{100 * r['sum_w'] / tot_bkg:>6.2f}% of background")


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_train_test(df):
    """Return (train_df, test_df) using eventNumber parity.

    Even eventNumber -> training.  Odd -> testing.
    This gives a reproducible, deterministic 50/50 split without a random seed.
    """
    train_mask = (df["eventNumber"].values % 2) == 0
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ROOT TTree writer
# ---------------------------------------------------------------------------

def df_to_ttree(df, tree_name, tree_title, feature_cols, weight_col, tfile,
                extra_cols=()):
    """Write *df* as a new ROOT TTree inside the open TFile *tfile*.

    All branches are written as 32-bit floats (Float_t / 'F').  The weight
    branch is named exactly *weight_col* so it matches SetWeightExpression, and
    *extra_cols* (the TMVA spectators) are written the same way.

    Returns the TTree (keep a Python reference until the file is closed).
    """
    import ROOT   # deferred so the module can be imported without ROOT present

    all_cols = list(feature_cols) + [weight_col] + list(extra_cols)

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
    print(f"    {tree_name:<28} {n_events:>9,} events  "
          f"sum w = {df[weight_col].sum():.6g}")
    return tree


def tree_safe(name):
    """Make a process name safe to use as a ROOT object name."""
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def drop_nonfinite(df, columns):
    """Drop rows with non-finite values in any of *columns*; report per column.

    mcd_over_mab is NaN when mab == 0, which TMVA handles badly.  With several
    background processes this is worth checking explicitly rather than trusting
    that it never happens.
    """
    finite = np.ones(len(df), dtype=bool)
    offenders = {}
    for col in columns:
        bad = ~np.isfinite(df[col].to_numpy(dtype=float))
        n_bad = int(bad.sum())
        if n_bad:
            offenders[col] = n_bad
        finite &= ~bad
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"\n  Dropping {n_dropped:,} row(s) with non-finite features:")
        for col, n in sorted(offenders.items(), key=lambda kv: -kv[1]):
            print(f"    {col:<18} {n:,}")
    return df[finite].reset_index(drop=True), n_dropped


def ensure_process_columns(df):
    """Back-fill `process`/`process_id` for Parquet files from an older Stage 1."""
    if "process" not in df.columns:
        print("  Note: no `process` column (older Stage 1 output); labelling "
              "all background as one process 'background'.")
        df["process"] = np.where(df["label"] == 1, "signal", "background")
    if "process_id" not in df.columns:
        names = sorted(df.loc[df["label"] == 1, "process"].unique().tolist()) \
            + sorted(df.loc[df["label"] == 0, "process"].unique().tolist())
        mapping = {n: i for i, n in enumerate(names)}
        df["process_id"] = df["process"].map(mapping).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_training(args):
    import ROOT
    ROOT.gROOT.SetBatch(True)    # suppress GUI windows

    # -- Load Parquet --------------------------------------------------------
    print(f"\nLoading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    df = ensure_process_columns(df)

    sidecar_path = args.input + ".samples.json"
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as fh:
            sidecar = json.load(fh)
        print(f"  Stage 1 provenance: {sidecar_path}")
        print(f"    signal weight mode     : "
              f"{sidecar.get('signal_weight_mode')}")
        print(f"    per-process cap        : "
              f"{sidecar.get('target_selected_per_process')}")
        if sidecar.get("signal_weight_mode") == "legacy":
            print("    WARNING: Stage 1 ran in legacy signal-weight mode, so "
                  "the signal carries ~1e-18 generator weights.")

    # -- Pseudo-background relabelling ---------------------------------------
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
        bkg_mask = df["mc_channel_number"].isin(bkg_channels)
        df["label"] = (~bkg_mask).astype(int)   # 1 = signal, 0 = pseudo-bkg
        df.loc[bkg_mask, "process"] = "pseudo_background"
        df = df.drop(columns=["process_id"])
        df = ensure_process_columns(df)
        present = sorted(df.loc[bkg_mask, "mc_channel_number"].unique().tolist())
        print(
            f"\n  Pseudo-background mode:\n"
            f"    pseudo-bkg channels (requested) : {sorted(bkg_channels)}\n"
            f"    pseudo-bkg channels (found)     : {present}\n"
            f"    signal (label=1)                : {int((df['label'] == 1).sum()):,}\n"
            f"    pseudo-bkg (label=0)            : {int((df['label'] == 0).sum()):,}"
        )
        if int((df["label"] == 0).sum()) == 0:
            sys.exit(
                "ERROR: No events matched --pseudo-bkg-channels.  "
                f"Available mc_channel_number values: "
                f"{sorted(df['mc_channel_number'].unique().tolist())}"
            )

    # -- Validate columns ----------------------------------------------------
    required = BDT_VARIABLES + [RAW_WEIGHT_COL, "label", "eventNumber"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns in Parquet: {missing}")

    if not args.keep_nonfinite:
        df, _ = drop_nonfinite(df, BDT_VARIABLES + [RAW_WEIGHT_COL])

    n_sig = int((df["label"] == 1).sum())
    n_bkg = int((df["label"] == 0).sum())
    if n_sig == 0 or n_bkg == 0:
        sys.exit("ERROR: Need both signal (label=1) and background (label=0) events.")

    n_neg = int((df[RAW_WEIGHT_COL] < 0).sum())
    sep = "-" * 60
    print(f"\n  {sep}")
    print(f"  Signal events      : {n_sig:>8,}")
    print(f"  Background events  : {n_bkg:>8,}")
    print(f"  Negative weights   : {n_neg:>8,}  ({100*n_neg/len(df):.2f}%)")
    print(f"  Weighting mode     : {args.weighting}")
    print(f"  Signal equalise    : {args.signal_equalise}")
    print(f"  Negative weights   : {args.neg_weights} in training")
    print(f"  {sep}")

    # -- Per-event training weight -------------------------------------------
    try:
        df[WEIGHT_COL] = compute_train_weights(
            df, args.weighting, args.signal_equalise)
    except (KeyError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")

    diagnostics = process_diagnostics(df)
    print_process_diagnostics(diagnostics)

    bkg_processes = sorted(
        df.loc[df["label"] == 0, "process"].unique().tolist())
    sig_processes = sorted(
        df.loc[df["label"] == 1, "process"].unique().tolist())
    process_map = (
        df[["process_id", "process", "label"]]
        .drop_duplicates()
        .sort_values("process_id")
    )

    # -- Train / test split --------------------------------------------------
    sig_df = df[df["label"] == 1].copy()
    bkg_df = df[df["label"] == 0].copy()

    sig_train, sig_test = split_train_test(sig_df)
    print(f"\n  Train: signal={len(sig_train):,}  "
          f"background={int(((bkg_df['eventNumber'] % 2) == 0).sum()):,}")
    print(f"  Test:  signal={len(sig_test):,}  "
          f"background={int(((bkg_df['eventNumber'] % 2) != 0).sum()):,}")

    # -- Set up output paths -------------------------------------------------
    out_dir = os.path.abspath(args.output_dir)
    # TMVA builds the weights path as "<DataLoaderName>/weights/...", resolved
    # relative to the current working directory.  The DataLoader name must be a
    # plain token (no path separators) - see the note by the DataLoader call
    # below.  We chdir into out_dir so weights land at out_dir/<dl_name>/weights.
    dl_name     = "dataset"
    weights_dir = os.path.join(out_dir, dl_name, "weights")
    trees_path  = os.path.join(out_dir, "training_trees.root")
    tmva_path   = os.path.join(out_dir, "TMVAClassification.root")
    map_path    = os.path.join(out_dir, "process_map.json")
    os.makedirs(weights_dir, exist_ok=True)

    # -- Write ROOT TTrees (unless reusing) ----------------------------------
    # Background gets one tree pair per process so that TMVA reports each
    # process' weight sum, which is the check that the composition is right.
    bkg_tree_names = {
        proc: (f"bkg_train_{tree_safe(proc)}", f"bkg_test_{tree_safe(proc)}")
        for proc in bkg_processes
    }

    if args.reuse_trees:
        if not os.path.exists(trees_path):
            sys.exit(f"ERROR: --reuse-trees set but {trees_path} not found.")
        print(f"\nReusing existing TTree file: {trees_path}")
    else:
        print(f"\nWriting ROOT TTrees to: {trees_path}")
        trees_file_write = ROOT.TFile(trees_path, "RECREATE")
        df_to_ttree(sig_train, "sig_train", "Signal training",
                    BDT_VARIABLES, WEIGHT_COL, trees_file_write, SPECTATORS)
        df_to_ttree(sig_test,  "sig_test",  "Signal test",
                    BDT_VARIABLES, WEIGHT_COL, trees_file_write, SPECTATORS)
        for proc in bkg_processes:
            g = bkg_df[bkg_df["process"] == proc]
            g_train, g_test = split_train_test(g)
            name_train, name_test = bkg_tree_names[proc]
            df_to_ttree(g_train, name_train, f"Background training {proc}",
                        BDT_VARIABLES, WEIGHT_COL, trees_file_write, SPECTATORS)
            df_to_ttree(g_test, name_test, f"Background test {proc}",
                        BDT_VARIABLES, WEIGHT_COL, trees_file_write, SPECTATORS)
        trees_file_write.Close()

    # -- Open trees for TMVA -------------------------------------------------
    trees_file = ROOT.TFile(trees_path, "READ")
    if trees_file.IsZombie():
        sys.exit(f"ERROR: Could not open {trees_path}")

    wanted = ["sig_train", "sig_test"]
    for proc in bkg_processes:
        wanted.extend(bkg_tree_names[proc])

    trees = {}
    for name in wanted:
        tree = trees_file.Get(name)
        if not tree:
            sys.exit(f"ERROR: Could not retrieve TTree '{name}' from {trees_path}. "
                     f"If --reuse-trees was used, the existing file may predate "
                     f"the per-process trees; re-run without it.")
        trees[name] = tree

    # -- TMVA Factory and DataLoader -----------------------------------------
    ignore_neg = (args.neg_weights == "ignore")
    bdt_opts = bdt_option_string(ignore_neg)
    print(f"\nRunning TMVA training")
    print(f"  TMVA output    : {tmva_path}")
    print(f"  Weights dir    : {weights_dir}")
    print(f"  Variables      : {len(BDT_VARIABLES)}")
    print(f"  Spectators     : {', '.join(SPECTATORS)}")
    print(f"  Signal trees   : {len(sig_processes)} process(es) in 1 tree pair")
    print(f"  Bkg trees      : {len(bkg_processes)} process(es), "
          f"1 tree pair each")
    print(f"  Hyperparameters:")
    for k, v in BDT_HYPERPARAMS.items():
        print(f"    {k:<22} = {v}")
    print(f"    {'IgnoreNegWeightsInTraining':<22} = {ignore_neg}")
    print(f"  NormMode       : EqualNumEvents (per class; the within-background "
          f"mix is set by the per-event weights)")

    tmva_out = ROOT.TFile(tmva_path, "RECREATE")

    # IMPORTANT: the DataLoader name must be a PLAIN token with no path
    # separators.  TMVA uses it both as (a) a TDirectory name inside the output
    # ROOT file and (b) the on-disk folder that holds the weights XML.  Passing
    # an absolute path (with '/') makes ROOT try to cd into a bogus nested
    # directory ("Error in <TFile::cd>: Unknown directory Users"), which leaves
    # MethodBase::BaseDir() null and segfaults inside TrainAllMethods().
    # Instead we chdir into out_dir and pass a simple name, so the weights land
    # at out_dir/<dl_name>/weights (tmva_path is absolute, so it is unaffected).
    os.chdir(out_dir)

    factory    = ROOT.TMVA.Factory("TMVAClassification", tmva_out, FACTORY_OPTIONS)
    dataloader = ROOT.TMVA.DataLoader(dl_name)

    # Register input variables
    for var in BDT_VARIABLES:
        dataloader.AddVariable(var, "F")

    # Spectators ride into TrainTree/TestTree without entering the training.
    for spec in SPECTATORS:
        dataloader.AddSpectator(spec, "F")

    # Add pre-split trees - TMVA respects kTraining/kTesting labels.
    # The per-tree global weight is 1.0 because scale_d and 1/sampling_fraction
    # are already folded into the per-event weight: a process can span several
    # DSIDs with different scale_d, so it cannot be a single per-tree number.
    dataloader.AddSignalTree(trees["sig_train"], 1.0, ROOT.TMVA.Types.kTraining)
    dataloader.AddSignalTree(trees["sig_test"],  1.0, ROOT.TMVA.Types.kTesting)
    for proc in bkg_processes:
        name_train, name_test = bkg_tree_names[proc]
        dataloader.AddBackgroundTree(
            trees[name_train], 1.0, ROOT.TMVA.Types.kTraining)
        dataloader.AddBackgroundTree(
            trees[name_test], 1.0, ROOT.TMVA.Types.kTesting)

    # Per-event weights
    dataloader.SetSignalWeightExpression    (WEIGHT_COL)
    dataloader.SetBackgroundWeightExpression(WEIGHT_COL)

    dataloader.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""),
                                          PREPARE_OPTIONS)

    # Book BDT method
    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", bdt_opts)

    # Train -> Test -> Evaluate
    print("\n" + "=" * 60)
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    print("=" * 60)

    tmva_out.Close()
    trees_file.Close()

    # -- Process map for Stage 3 ---------------------------------------------
    payload = {
        "weighting": args.weighting,
        "signal_equalise": bool(args.signal_equalise),
        "neg_weights": args.neg_weights,
        "input": os.path.abspath(args.input),
        "processes": [
            {"process_id": int(r.process_id), "process": r.process,
             "label": int(r.label)}
            for r in process_map.itertuples()
        ],
        "diagnostics": diagnostics,
    }
    with open(map_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # -- Summary -------------------------------------------------------------
    xml_path = os.path.join(weights_dir, "TMVAClassification_BDT.weights.xml")
    print(f"\nTraining complete.")
    print(f"  Weights file   : {xml_path}")
    print(f"  TMVA ROOT file : {tmva_path}")
    print(f"  Process map    : {map_path}")
    print(f"\nTo view results interactively:")
    print(f"  root -l '{tmva_path}'")
    print(f"  // then in the ROOT prompt:")
    print(f"  TMVA::TMVAGui(\"{tmva_path}\")")
    print(f"\nKey plots to check:")
    print(f"  1. ROC curve (signal efficiency vs background rejection)")
    print(f"  2. BDT response overtraining check (train vs test KS test)")
    print(f"  3. Variable importance / ranking")
    print(f"  4. Per-process background rejection (Stage 3)")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
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
        help="Output directory for dataset/weights/ and TMVAClassification.root. "
             "Default: tmva_ZdZd_R3",
    )
    p.add_argument(
        "--weighting", choices=["normalised", "equal-process", "raw"],
        default="normalised",
        help="How to combine evtWeight_total, scale_d and sampling_fraction. "
             "See WEIGHTING in this script's docstring. Default: normalised.",
    )
    p.add_argument(
        "--signal-equalise", action="store_true",
        help="Give each signal process (mZd point) the same total weight.",
    )
    p.add_argument(
        "--neg-weights", choices=["ignore", "keep"], default="ignore",
        help="Whether TMVA ignores negative-weight events in training "
             "(IgnoreNegWeightsInTraining). Default: ignore, because three of "
             "the four irreducible processes are Sherpa and AdaBoost is "
             "unstable with negative weights.",
    )
    p.add_argument(
        "--keep-nonfinite", action="store_true",
        help="Do not drop rows with NaN/inf feature values (e.g. mcd_over_mab "
             "when mab == 0).",
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
    p.add_argument(
        "--no-fast-exit", action="store_true",
        help="Return from main() normally instead of calling os._exit(0). The "
             "fast exit suppresses a harmless PyROOT/TMVA interpreter teardown "
             "segfault that occurs after all outputs are written.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.input):
        sys.exit(f"ERROR: Input file not found: {args.input}")
    run_training(args)
    if not args.no_fast_exit:
        # The ROOT files are closed and the weights XML is on disk by this
        # point.  PyROOT/TMVA segfaults during interpreter teardown on some
        # builds (observed on lxplus with /usr/bin/python3), after
        # "Thank you for using TMVA".  Exit before teardown runs.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
