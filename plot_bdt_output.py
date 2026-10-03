#!/usr/bin/env python3
"""
plot_bdt_output.py
==================
Stage 3 of the TMVA pipeline: read the trained BDT output and produce a
standard set of plots as PNG files.

It reads three products of train_bdt.py:
  * <output-dir>/TMVAClassification.root   -- contains dataset/TrainTree and
                                              dataset/TestTree, each with the 24
                                              input variables plus the columns
                                              classID (0=Signal, 1=Background),
                                              weight, BDT (the classifier
                                              response) and the spectator
                                              process_id.
  * <output-dir>/dataset/weights/TMVAClassification_BDT.weights.xml
                                           -- the trained forest, used to derive
                                              a variable-importance ranking.
  * <output-dir>/process_map.json          -- process_id -> process name, plus
                                              the Stage 2 weighting mode and the
                                              per-process diagnostics.

Plots produced:
  1. roc.png                     -- ROC curve (signal eff vs background rejection),
                                    test sample, with weighted AUC.
  2. bdt_response_overtrain.png  -- BDT response for signal/background, train vs
                                    test overlaid (TMVA-style overtraining check),
                                    with a Kolmogorov-Smirnov statistic per class.
  3. variable_importance.png     -- variable-importance bar chart derived from the
                                    forest (boost-weighted split frequency).
  4. top_variables.png           -- signal-vs-background distributions of the top
                                    discriminating input variables (test sample).
  5. bdt_response_stack.png      -- background stacked by physics process at the
                                    Stage 2 weights, with signal overlaid.  This
                                    is the plot that shows the background
                                    COMPOSITION, which the area-normalised
                                    overtraining plot deliberately hides.
  6. per_process_rejection.png   -- background efficiency per process against
                                    signal efficiency, i.e. one ROC per
                                    background process.  Shows which process the
                                    BDT fails to reject.

Plots 5 and 6 need the process_id spectator; they are skipped with a note if the
trees predate it (train_bdt.py before 2026-10) or if there is only one
background process.

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
        --tmva-file   tmva_ZdZd_R3/TMVAClassification.root \
        --weights     tmva_ZdZd_R3/dataset/weights/TMVAClassification_BDT.weights.xml \
        --process-map tmva_ZdZd_R3/process_map.json \
        --plot-dir    tmva_ZdZd_R3/plots --top-n 6
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless / batch - no display needed on lxplus
import matplotlib.pyplot as plt


# TMVA convention: the first booked class (Signal) is classID 0.
SIGNAL_ID = 0
BACKGROUND_ID = 1

SPECTATOR_PROCESS = "process_id"

# Qualitative colours for the background processes, in manifest order.
PROCESS_COLORS = [
    "#2471a3", "#1e8449", "#b9770e", "#7d3c98", "#117a65", "#922b21",
]


def _trapezoid(y, x):
    """np.trapezoid where available (numpy >= 2), else np.trapz."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


# ---------------------------------------------------------------------------
# Pure numeric helpers (no ROOT dependency - unit-testable)
# ---------------------------------------------------------------------------

def threshold_grid(score, n_points=400):
    """Common threshold grid spanning the score range."""
    return np.linspace(np.min(score), np.max(score), n_points)


def efficiency_curve(score, weight, thresholds):
    """Weighted fraction of `score` at or above each threshold.

    Returns an array of the same length as `thresholds`.  The denominator is the
    signed weight sum, so the curve starts at 1.0 and falls to 0.0.
    """
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    total = weight.sum()
    if total == 0:
        raise ValueError("efficiency_curve: total weight is zero")
    return np.array([weight[score >= thr].sum() / total for thr in thresholds])


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

    if weight[is_signal].sum() == 0 or weight[~is_signal].sum() == 0:
        raise ValueError("weighted_roc: signal or background total weight is zero")

    thresholds = threshold_grid(score, n_points)
    sig_eff = efficiency_curve(score[is_signal], weight[is_signal], thresholds)
    bkg_eff = efficiency_curve(score[~is_signal], weight[~is_signal], thresholds)
    bkg_rej = 1.0 - bkg_eff

    # Integrate rejection vs efficiency.  Sort by efficiency for a clean integral.
    order = np.argsort(sig_eff)
    auc = _trapezoid(bkg_rej[order], sig_eff[order])
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


def weighted_hist(values, weight, edges):
    """Raw weighted histogram (no normalisation) with bin centres."""
    h, _ = np.histogram(values, bins=edges, weights=weight)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return h, centres


def rejection_at_signal_eff(sig_eff, bkg_eff, target):
    """Background efficiency interpolated at a target signal efficiency."""
    order = np.argsort(sig_eff)
    return float(np.interp(target, np.asarray(sig_eff)[order],
                           np.asarray(bkg_eff)[order]))


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
    keyed by branch name (includes 'BDT', 'classID', 'weight', the inputs and
    any spectators).
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


def load_process_map(path):
    """Read process_map.json written by train_bdt.py.

    Returns (id_to_name, meta) or (None, None) if the file is absent.
    """
    if not path or not os.path.exists(path):
        return None, None
    with open(path) as fh:
        payload = json.load(fh)
    id_to_name = {
        int(p["process_id"]): p["process"]
        for p in payload.get("processes", [])
        if int(p.get("label", 0)) == 0
    }
    return id_to_name, payload


def background_processes(test, id_to_name):
    """Return [(process_id, name, mask)] for the background processes present."""
    if SPECTATOR_PROCESS not in test:
        return []
    bkg = test["classID"] == BACKGROUND_ID
    ids = np.unique(np.rint(test[SPECTATOR_PROCESS][bkg]).astype(int))
    out = []
    for pid in ids:
        mask = bkg & (np.rint(test[SPECTATOR_PROCESS]).astype(int) == pid)
        name = (id_to_name or {}).get(int(pid), f"process {int(pid)}")
        out.append((int(pid), name, mask))
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
    ax.set_ylabel("Background rejection (1 - bkg eff)")
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
    # Legend top-centre, KS box below it on the left so the two cannot overlap.
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax.text(0.02, 0.80, "\n".join(ks_txt), transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
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


def plot_response_stack(test, procs, out_path, n_bins=40, weighting=None):
    """Background stacked by process at the training weights, signal overlaid.

    Unlike the overtraining plot this is NOT area-normalised: the point is to
    show how much of the background each process contributes and where in the
    BDT response it sits.  The signal is scaled to the background total so its
    shape is visible on the same axes.
    """
    lo, hi = float(test["BDT"].min()), float(test["BDT"].max())
    edges = np.linspace(lo, hi, n_bins + 1)
    width = edges[1] - edges[0]

    # Largest process at the bottom of the stack.
    sized = []
    for pid, name, mask in procs:
        h, centres = weighted_hist(test["BDT"][mask], test["weight"][mask], edges)
        sized.append((float(np.sum(h)), name, h, centres))
    sized.sort(key=lambda t: -t[0])

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    bottom = np.zeros(n_bins)
    total = sum(s[0] for s in sized)
    for i, (ssum, name, h, centres) in enumerate(sized):
        frac = 100.0 * ssum / total if total else 0.0
        ax.bar(centres, h, width=width, bottom=bottom, align="center",
               color=PROCESS_COLORS[i % len(PROCESS_COLORS)],
               label=f"{name} ({frac:.1f}%)", linewidth=0)
        bottom += h

    sig = test["classID"] == SIGNAL_ID
    sig_h, centres = weighted_hist(test["BDT"][sig], test["weight"][sig], edges)
    sig_sum = float(np.sum(sig_h))
    scale = (total / sig_sum) if sig_sum else 1.0
    ax.step(centres, sig_h * scale, where="mid", color="#c0392b", lw=2,
            label=f"Signal (x{scale:.3g})")

    ax.set_xlabel("BDT response")
    ax.set_ylabel("Weighted events / bin")
    title = "BDT response: background by process"
    if weighting:
        title += f"  [weighting = {weighting}]"
    ax.set_title(title, fontsize=11)
    ax.set_yscale("log")
    # Headroom so the legend cannot sit on top of the distributions.
    finite = np.concatenate([bottom[bottom > 0], (sig_h * scale)[sig_h > 0]])
    if finite.size:
        ax.set_ylim(top=float(finite.max()) * 1e3)
    ax.legend(loc="upper center", fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return [(name, ssum) for ssum, name, _, _ in sized]


def plot_per_process_rejection(test, procs, out_path, n_points=400,
                               targets=(0.5, 0.8, 0.9)):
    """One ROC per background process: background efficiency vs signal efficiency.

    Returns {process: {target_signal_eff: background_eff}} so the numbers can be
    printed alongside the plot.
    """
    thresholds = threshold_grid(test["BDT"], n_points)
    sig = test["classID"] == SIGNAL_ID
    sig_eff = efficiency_curve(test["BDT"][sig], test["weight"][sig], thresholds)

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    summary = {}
    for i, (pid, name, mask) in enumerate(procs):
        try:
            bkg_eff = efficiency_curve(test["BDT"][mask], test["weight"][mask],
                                       thresholds)
        except ValueError:
            print(f"    (skipping {name}: total weight is zero)")
            continue
        ax.plot(sig_eff, np.clip(bkg_eff, 1e-6, None), lw=1.8,
                color=PROCESS_COLORS[i % len(PROCESS_COLORS)], label=name)
        summary[name] = {
            t: rejection_at_signal_eff(sig_eff, bkg_eff, t) for t in targets
        }

    all_bkg = test["classID"] == BACKGROUND_ID
    bkg_eff_tot = efficiency_curve(test["BDT"][all_bkg], test["weight"][all_bkg],
                                   thresholds)
    ax.plot(sig_eff, np.clip(bkg_eff_tot, 1e-6, None), lw=2.2, ls="--",
            color="black", label="all background")
    summary["all background"] = {
        t: rejection_at_signal_eff(sig_eff, bkg_eff_tot, t) for t in targets
    }

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_title("Per-process background efficiency (test sample)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot TMVA BDT output.")
    p.add_argument("--output-dir", default=None,
                   help="train_bdt.py output dir; used to locate the ROOT file, "
                        "weights XML and process map if not given explicitly.")
    p.add_argument("--tmva-file", default=None,
                   help="Path to TMVAClassification.root")
    p.add_argument("--weights", default=None,
                   help="Path to TMVAClassification_BDT.weights.xml")
    p.add_argument("--process-map", default=None,
                   help="Path to process_map.json from train_bdt.py.")
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
    process_map_path = args.process_map or (
        os.path.join(out_dir, "process_map.json") if out_dir else None)
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

    id_to_name, meta = load_process_map(process_map_path)
    if meta:
        print(f"Reading process map: {process_map_path}")
        print(f"  Stage 2 weighting: {meta.get('weighting')}"
              f"   neg weights: {meta.get('neg_weights')}")
    procs = background_processes(test, id_to_name)
    if procs:
        print(f"  Background processes in trees: "
              f"{', '.join(n for _, n, _ in procs)}")
    elif SPECTATOR_PROCESS not in test:
        print(f"  (no '{SPECTATOR_PROCESS}' spectator in the trees; "
              f"per-process plots will be skipped)")

    # Variable importance (optional - degrade gracefully if XML missing).
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

    composition = None
    rejection = None
    if len(procs) >= 2:
        composition = plot_response_stack(
            test, procs, os.path.join(plot_dir, "bdt_response_stack.png"),
            weighting=(meta or {}).get("weighting"))
        print("  bdt_response_stack.png")
        rejection = plot_per_process_rejection(
            test, procs, os.path.join(plot_dir, "per_process_rejection.png"))
        print("  per_process_rejection.png")
    elif procs:
        print("  (only one background process; per-process plots skipped)")

    if ranking:
        print("\n  Top variables by importance:")
        for name, imp in ranking[:args.top_n]:
            print(f"    {name:<16} {imp:.4f}")

    if composition:
        total = sum(s for _, s in composition)
        print("\n  Background composition in the test sample "
              "(weighted, at the Stage 2 weights):")
        for name, s in composition:
            pct = 100.0 * s / total if total else 0.0
            print(f"    {name:<18} {s:>14.6g}  {pct:>6.2f}%")

    if rejection:
        targets = sorted(next(iter(rejection.values())).keys())
        print("\n  Background efficiency at fixed signal efficiency:")
        head = "    " + f"{'process':<18}" + "".join(
            f"{'eps_S=' + format(t, '.2f'):>14}" for t in targets)
        print(head)
        for name in sorted(rejection, key=lambda n: n == "all background"):
            row = "".join(f"{rejection[name][t]:>14.3g}" for t in targets)
            print(f"    {name:<18}{row}")

    print("\nDone.")


if __name__ == "__main__":
    main()
