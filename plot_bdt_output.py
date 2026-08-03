#!/usr/bin/env python3
"""
plot_bdt_output.py
==================
Stage 3 (lightweight) of the TMVA pipeline: read the trained BDT output and
produce a standard set of demonstration plots as PNG files.

It reads two products of train_bdt.py:
  * <output-dir>/TMVAClassification.root   -- contains dataset/TrainTree and
                                              dataset/TestTree, each with the 24
                                              input variables plus the columns
                                              classID (0=Signal, 1=Background),
                                              weight, and BDT (the classifier
                                              response).
  * <output-dir>/dataset/weights/TMVAClassification_BDT.weights.xml
                                           -- the trained forest, used to derive
                                              a variable-importance ranking.

Plots produced (the "standard set"):
  1. roc.png                     -- ROC curve (signal eff vs background rejection),
                                    test sample, with weighted AUC.
  2. bdt_response_overtrain.png  -- BDT response for signal/background, train vs
                                    test overlaid (TMVA-style overtraining check),
                                    with a Kolmogorov-Smirnov statistic per class.
  3. variable_importance.png     -- variable-importance bar chart derived from the
                                    forest (boost-weighted split frequency).
  4. top_variables.png           -- signal-vs-background distributions of the top
                                    discriminating input variables (test sample).

Design note
-----------
Everything that touches ROOT/uproot is confined to load_tmva_trees() and
parse_importance_from_xml().  All numeric and plotting helpers take plain numpy
arrays so they can be unit-tested without ROOT installed.

Requirements
------------
    uproot, numpy, matplotlib   (all present in recent LCG views)
No ROOT, PyROOT, or scipy needed.

Usage
-----
    python3 plot_bdt_output.py --output-dir tmva_ZdZd_R3
    python3 plot_bdt_output.py \
        --tmva-file tmva_ZdZd_R3/TMVAClassification.root \
        --weights   tmva_ZdZd_R3/dataset/weights/TMVAClassification_BDT.weights.xml \
        --plot-dir  tmva_ZdZd_R3/plots --top-n 6
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless / batch — no display needed on lxplus
import matplotlib.pyplot as plt


# TMVA convention: the first booked class (Signal) is classID 0.
SIGNAL_ID = 0
BACKGROUND_ID = 1


# ---------------------------------------------------------------------------
# Pure numeric helpers (no ROOT dependency — unit-testable)
# ---------------------------------------------------------------------------

def weighted_roc(score, is_signal, weight, n_points=400):
    """Weighted ROC as (signal_efficiency, background_rejection, auc).

    score      : 1D array of classifier responses
    is_signal  : 1D bool/int array, True/1 for signal
    weight     : 1D array of per-event weights (may be negative)
    n_points   : number of threshold points to scan

    Returns (sig_eff, bkg_rej, auc) where auc is the integral of
    background rejection vs signal efficiency (higher = better, ~0.5 = random).
    """
    score = np.asarray(score, dtype=float)
    is_signal = np.asarray(is_signal).astype(bool)
    weight = np.asarray(weight, dtype=float)

    s_tot = weight[is_signal].sum()
    b_tot = weight[~is_signal].sum()
    if s_tot == 0 or b_tot == 0:
        raise ValueError("weighted_roc: signal or background total weight is zero")

    lo, hi = np.min(score), np.max(score)
    thresholds = np.linspace(lo, hi, n_points)

    sig_eff = np.empty(n_points)
    bkg_eff = np.empty(n_points)
    for i, thr in enumerate(thresholds):
        passed = score >= thr
        sig_eff[i] = weight[is_signal & passed].sum() / s_tot
        bkg_eff[i] = weight[~is_signal & passed].sum() / b_tot
    bkg_rej = 1.0 - bkg_eff

    # Integrate rejection vs efficiency.  Sort by efficiency for a clean integral.
    order = np.argsort(sig_eff)
    auc = float(np.trapz(bkg_rej[order], sig_eff[order]))
    return sig_eff, bkg_rej, auc


def ks_statistic(a, wa, b, wb, n_bins=400):
    """Weighted two-sample Kolmogorov-Smirnov statistic (max CDF distance).

    Returns a value in [0, 1]; small means the two samples are consistent.
    A weighted analogue of scipy.stats.ks_2samp's statistic, computed on a
    common binning so negative weights are handled gracefully.
    """
    a, wa = np.asarray(a, float), np.asarray(wa, float)
    b, wb = np.asarray(b, float), np.asarray(wb, float)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=edges, weights=wa)
    hb, _ = np.histogram(b, bins=edges, weights=wb)
    ca = np.cumsum(ha) / ha.sum()
    cb = np.cumsum(hb) / hb.sum()
    return float(np.max(np.abs(ca - cb)))


def normalized_hist(values, weight, edges):
    """Density-normalised histogram (integral = 1) with bin centres."""
    h, _ = np.histogram(values, bins=edges, weights=weight, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return h, centres


# ---------------------------------------------------------------------------
# XML parsing for variable importance (no ROOT dependency)
# ---------------------------------------------------------------------------

def parse_importance_from_xml(xml_path):
    """Derive a variable-importance ranking from a TMVA BDT weights file.

    Importance(var) = sum over all trees of  boostWeight * (# split nodes on var),
    normalised to sum to 1.  This is the AdaBoost-style split-frequency measure;
    it approximates TMVA's built-in gain-weighted ranking and needs only the XML.

    Returns a list of (variable_name, importance) sorted descending.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Map variable index -> expression (name).
    idx_to_name = {}
    variables = root.find("Variables")
    if variables is not None:
        for var in variables.findall("Variable"):
            idx = int(var.get("VarIndex"))
            idx_to_name[idx] = var.get("Expression", var.get("Label", f"var{idx}"))

    counts = {i: 0.0 for i in idx_to_name}

    weights = root.find("Weights")
    trees = [] if weights is None else weights.findall("BinaryTree")
    for btree in trees:
        try:
            bw = float(btree.get("boostWeight", "1.0"))
        except (TypeError, ValueError):
            bw = 1.0
        for node in btree.iter("Node"):
            ivar = node.get("IVar")
            if ivar is None:
                continue
            k = int(ivar)
            if k < 0:            # leaf node
                continue
            counts[k] = counts.get(k, 0.0) + bw

    total = sum(counts.values())
    if total <= 0:
        # Fall back to raw split counts if boost weights were unavailable.
        total = 1.0
    ranking = [
        (idx_to_name.get(k, f"var{k}"), counts[k] / total)
        for k in counts
    ]
    ranking.sort(key=lambda kv: kv[1], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# ROOT / uproot I/O (isolated so the rest of the module is testable)
# ---------------------------------------------------------------------------

def load_tmva_trees(tmva_file, dataset="dataset"):
    """Read TrainTree and TestTree from a TMVAClassification.root file.

    Returns a dict with keys 'train' and 'test', each a dict of numpy arrays
    keyed by branch name (includes 'BDT', 'classID', 'weight', and the inputs).
    """
    import uproot  # deferred so the module imports without uproot present
    out = {}
    with uproot.open(tmva_file) as f:
        for key, tname in (("train", "TrainTree"), ("test", "TestTree")):
            path = f"{dataset}/{tname}"
            if path not in f and f"{dataset}/{tname};1" not in f:
                raise KeyError(
                    f"'{path}' not found in {tmva_file}. "
                    f"Available: {list(f.keys())[:20]}"
                )
            out[key] = f[path].arrays(library="np")
    return out


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def plot_roc(test, out_path):
    sig = test["classID"] == SIGNAL_ID
    sig_eff, bkg_rej, auc = weighted_roc(test["BDT"], sig, test["weight"])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(sig_eff, bkg_rej, lw=2, color="#1f4e79", label=f"BDT (AUC = {auc:.3f})")
    ax.plot([0, 1], [1, 0], ls="--", lw=1, color="grey", label="random")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection (1 − bkg eff)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("BDT ROC curve (test sample)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return auc


def plot_overtraining(train, test, out_path, n_bins=40):
    lo = min(train["BDT"].min(), test["BDT"].min())
    hi = max(train["BDT"].max(), test["BDT"].max())
    edges = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    styles = {
        "signal":     dict(mask_id=SIGNAL_ID,     color="#c0392b", label="Signal"),
        "background": dict(mask_id=BACKGROUND_ID, color="#2471a3", label="Background"),
    }
    ks_txt = []
    for name, st in styles.items():
        tr_mask = train["classID"] == st["mask_id"]
        te_mask = test["classID"] == st["mask_id"]
        tr_h, centres = normalized_hist(train["BDT"][tr_mask],
                                        train["weight"][tr_mask], edges)
        te_h, _ = normalized_hist(test["BDT"][te_mask],
                                  test["weight"][te_mask], edges)
        # Test: filled histogram.  Train: points with error-bar style markers.
        ax.bar(centres, te_h, width=(edges[1] - edges[0]), align="center",
               color=st["color"], alpha=0.35, label=f"{st['label']} (test)")
        ax.plot(centres, tr_h, "o", ms=4, color=st["color"],
                label=f"{st['label']} (train)")
        ks = ks_statistic(train["BDT"][tr_mask], train["weight"][tr_mask],
                          test["BDT"][te_mask], test["weight"][te_mask])
        ks_txt.append(f"KS {st['label'].lower()}: {ks:.3f}")

    ax.set_xlabel("BDT response")
    ax.set_ylabel("(1/N) dN/dx")
    ax.set_title("BDT overtraining check")
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax.text(0.02, 0.97, "\n".join(ks_txt), transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_variable_importance(ranking, out_path, top_n=None):
    if top_n:
        ranking = ranking[:top_n]
    names = [r[0] for r in ranking][::-1]
    vals = [r[1] for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(4, 0.32 * len(names) + 1)))
    ax.barh(names, vals, color="#1f4e79")
    ax.set_xlabel("Relative importance (boost-weighted split frequency)")
    ax.set_title("BDT variable importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_top_variables(test, ranking, out_path, top_n=6, n_bins=40):
    top_vars = [name for name, _ in ranking if name in test][:top_n]
    ncol = 3
    nrow = int(np.ceil(len(top_vars) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    sig = test["classID"] == SIGNAL_ID
    for ax, var in zip(axes, top_vars):
        v = test[var]
        lo, hi = np.percentile(v, [0.5, 99.5])
        if lo == hi:
            lo, hi = v.min(), v.max() + 1e-9
        edges = np.linspace(lo, hi, n_bins + 1)
        for mask, color, lab in ((sig, "#c0392b", "Signal"),
                                 (~sig, "#2471a3", "Background")):
            h, centres = normalized_hist(v[mask], test["weight"][mask], edges)
            ax.step(centres, h, where="mid", color=color, label=lab, lw=1.5)
        ax.set_title(var, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(top_vars):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Top discriminating variables (test sample, area-normalised)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot TMVA BDT demonstration output.")
    p.add_argument("--output-dir", default=None,
                   help="train_bdt.py output dir; used to locate the ROOT file "
                        "and weights XML if --tmva-file/--weights are not given.")
    p.add_argument("--tmva-file", default=None,
                   help="Path to TMVAClassification.root")
    p.add_argument("--weights", default=None,
                   help="Path to TMVAClassification_BDT.weights.xml")
    p.add_argument("--dataset-name", default="dataset",
                   help="DataLoader name / TDirectory holding the trees "
                        "(default 'dataset').")
    p.add_argument("--plot-dir", default=None,
                   help="Where to write PNGs (default <output-dir>/plots).")
    p.add_argument("--top-n", type=int, default=6,
                   help="How many top variables to show in top_variables.png.")
    args = p.parse_args()

    out_dir = args.output_dir
    tmva_file = args.tmva_file or (
        os.path.join(out_dir, "TMVAClassification.root") if out_dir else None)
    weights = args.weights or (
        os.path.join(out_dir, args.dataset_name, "weights",
                     "TMVAClassification_BDT.weights.xml") if out_dir else None)
    if not tmva_file:
        sys.exit("ERROR: provide --tmva-file or --output-dir.")
    if not os.path.exists(tmva_file):
        sys.exit(f"ERROR: ROOT file not found: {tmva_file}")

    plot_dir = args.plot_dir or (
        os.path.join(out_dir, "plots") if out_dir
        else os.path.join(os.path.dirname(os.path.abspath(tmva_file)), "plots"))
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Reading trees from : {tmva_file}")
    trees = load_tmva_trees(tmva_file, dataset=args.dataset_name)
    train, test = trees["train"], trees["test"]
    print(f"  train events: {len(train['BDT']):,}   test events: {len(test['BDT']):,}")

    # Variable importance (optional — degrade gracefully if XML missing).
    ranking = None
    if weights and os.path.exists(weights):
        print(f"Reading weights    : {weights}")
        ranking = parse_importance_from_xml(weights)
    else:
        print(f"  (weights XML not found at {weights}; skipping importance-based "
              f"plots)")

    print(f"Writing plots to   : {plot_dir}")
    auc = plot_roc(test, os.path.join(plot_dir, "roc.png"))
    print(f"  roc.png                    (AUC = {auc:.3f})")
    plot_overtraining(train, test,
                      os.path.join(plot_dir, "bdt_response_overtrain.png"))
    print("  bdt_response_overtrain.png")
    if ranking:
        plot_variable_importance(
            ranking, os.path.join(plot_dir, "variable_importance.png"))
        print("  variable_importance.png")
        plot_top_variables(
            test, ranking, os.path.join(plot_dir, "top_variables.png"),
            top_n=args.top_n)
        print("  top_variables.png")
        print("\n  Top variables by importance:")
        for name, imp in ranking[:args.top_n]:
            print(f"    {name:<16} {imp:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
