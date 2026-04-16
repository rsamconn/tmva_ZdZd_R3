#!/usr/bin/env python3
"""
analyse_data_for_bdt.py
=======================
Examines the ZdZd13TeV and ZdZdPostProcessing example ROOT files to determine
how to format the data for TMVA BDT training.

Outputs a summary to stdout covering:
  - File/tree structure for both data formats
  - Branch inventory (event-level scalars vs. per-object jagged arrays)
  - Key distributions (lepton multiplicities, dilepton masses, 4-lepton mass, etc.)
  - How the cutflow cuts in ZdZdPP_alg.cxx map to available branches
  - A recommended strategy for flattening the data to fixed-size BDT feature vectors

Usage:
    python3 analyse_data_for_bdt.py

Requirements:
    pip install uproot awkward numpy --break-system-packages
"""

import sys
import os
import uproot
import awkward as ak
import numpy as np
from collections import defaultdict

# Test function to check version.
def test_func():
    print("analyse_data_for_bdt version: 1.0")

# ---------------------------------------------------------------------------
# Paths (relative to script location or current working directory)
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_13TEV = {
    "mc23a_mZd30": os.path.join(BASE, "../data/example_data/ZdZd13TeV/mc23a_mZd30/my.output.root"),
    "mc23a_mZd60": os.path.join(BASE, "../data/example_data/ZdZd13TeV/mc23a_mZd60/my.output.root"),
}
DATA_PP = {
    "mc23a_mZd30": os.path.join(BASE, "../data/example_data/ZdZdPostProcessing/mc23a_mZd30/myfile.root"),
    "mc23a_mZd60": os.path.join(BASE, "../data/example_data/ZdZdPostProcessing/mc23a_mZd60/myfile.root"),
}

SEP = "=" * 70
SEP2 = "-" * 70


# ---------------------------------------------------------------------------
# Helper: simple text histogram
# ---------------------------------------------------------------------------
def text_hist(values, bins=10, lo=None, hi=None, width=40, label=""):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print("  (no finite values)")
        return
    lo = lo if lo is not None else float(values.min())
    hi = hi if hi is not None else float(values.max())
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    cmax = counts.max() if counts.max() > 0 else 1
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / cmax * width))
        print(f"  [{edges[i]:8.2f}, {edges[i+1]:8.2f}) | {bar:<{width}} {c}")


# ---------------------------------------------------------------------------
# Helper: print basic stats
# ---------------------------------------------------------------------------
def stats(values, name, unit="", scale=1.0):
    arr = np.asarray(values, dtype=float) * scale
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        print(f"  {name}: (empty)")
        return
    print(
        f"  {name}: n={len(arr)}, min={arr.min():.3g}, "
        f"mean={arr.mean():.3g}, median={np.median(arr):.3g}, "
        f"max={arr.max():.3g} {unit}"
    )


# ===========================================================================
# Section 1 – ZdZd13TeV tree structure
# ===========================================================================
def section_13tev_structure():
    print()
    print(SEP)
    print("SECTION 1: ZdZd13TeV output – file/tree structure")
    print(SEP)

    for label, path in DATA_13TEV.items():
        print(f"\n[{label}]  {path}")
        with uproot.open(path) as f:
            all_keys = f.keys()
            # Identify top-level directories (systematic variations)
            dirs = sorted({k.split("/")[0].rstrip(";1") for k in all_keys if "/" in k})
            trees_at_top = [k for k in all_keys if "/" not in k and "Tree" in k]
            other_top = [k for k in all_keys if "/" not in k and "Tree" not in k]

            print(f"  Top-level directories (systematic variants): {len(dirs)}")
            print(f"    Nominal  +  {len(dirs)-1} systematic variations")
            print(f"    Systematics: {', '.join(d for d in dirs if d != 'Nominal')[:120]}...")
            print(f"  Other top-level objects: {[k.rstrip(';1') for k in other_top]}")

            # Inspect Nominal tree in detail
            tree = f["Nominal/llllTree"]
            keys = tree.keys()
            scalar = [k for k in keys if not tree[k].interpretation.__class__.__name__.startswith("AsJagged")]
            jagged = [k for k in keys if tree[k].interpretation.__class__.__name__.startswith("AsJagged")]

            print(f"\n  Nominal/llllTree:")
            print(f"    Total entries (events): {tree.num_entries}")
            print(f"    Total branches:         {len(keys)}")
            print(f"    Scalar (event-level):   {len(scalar)}")
            print(f"    Jagged (per-object):    {len(jagged)}")

            # Classify jagged branches by prefix
            prefixes = ["truth_l_", "truth_ll_", "truth_llll_",
                        "hard_l_", "hard_ll_",
                        "l_", "ll_", "llll_", "j_"]
            prefix_counts = defaultdict(list)
            unclassified = []
            for b in jagged:
                matched = False
                for p in prefixes:
                    if b.startswith(p):
                        prefix_counts[p].append(b)
                        matched = True
                        break
                if not matched:
                    unclassified.append(b)

            print("\n  Jagged branches by collection:")
            for p, blist in prefix_counts.items():
                print(f"    {p:<20} {len(blist):3d} branches")
                # Show branch names without prefix, abbreviated
                names = [b[len(p):] for b in blist]
                chunk = ", ".join(names[:8])
                if len(names) > 8:
                    chunk += f", ... (+{len(names)-8} more)"
                print(f"      fields: {chunk}")
            if unclassified:
                print(f"    (unclassified): {unclassified}")

            # Systematic trees have fewer branches
            syst_tree_key = [k for k in all_keys if "EG_RESOLUTION" in k and "llllTree" in k][0]
            syst_tree = f[syst_tree_key]
            syst_keys = syst_tree.keys()
            print(f"\n  Example systematic tree ({syst_tree_key.rstrip(';1')}):")
            print(f"    Branches: {len(syst_keys)}")
            print(f"    (Systematic trees only carry kinematic branches, not full metadata)")

            break  # structure is the same for both samples, show once


# ===========================================================================
# Section 2 – ZdZdPostProcessing structure
# ===========================================================================
def section_pp_structure():
    print()
    print(SEP)
    print("SECTION 2: ZdZdPostProcessing output – file structure")
    print(SEP)

    for label, path in DATA_PP.items():
        print(f"\n[{label}]  {path}")
        with uproot.open(path) as f:
            all_keys = f.keys()
            # The PP output contains only histograms organised as:
            # avgMll / REGION / [SYST /] CHANNEL_NUMBER
            regions = set()
            systs = set()
            channels = set()
            for k in all_keys:
                parts = k.rstrip(";1").split("/")
                if len(parts) >= 2:
                    regions.add(parts[1])
                if len(parts) == 4:
                    systs.add(parts[2])
                if len(parts) >= 3 and parts[-1].startswith("_"):
                    channels.add(parts[-1])

            print(f"  Object type: TH1 histograms only (no TTrees)")
            print(f"  Histogram variable: avgMll = (m12 + m34) / 2")
            print(f"  Regions found:  {sorted(regions)}")
            print(f"  Systematics found ({len(systs)}): {', '.join(sorted(systs))[:100]}...")
            print(f"  MC channel numbers (datasets): {sorted(channels)}")

            # Read one histogram and show its properties
            # Nominal histograms are at depth 3: avgMll/REGION/CHANNEL_NUMBER
            sr_key = [k for k in all_keys if "SR_4m" in k and k.count("/") == 2
                      and "up" not in k and "down" not in k][0]
            h = f[sr_key]
            vals, edges = h.to_numpy()
            print(f"\n  Example histogram: {sr_key.rstrip(';1')}")
            print(f"    Bins: {len(vals)}, range: [{edges[0]:.1f}, {edges[-1]:.1f}] GeV")
            print(f"    Integral (sum of bin contents): {vals.sum():.4f}")

        break  # structure is identical for both


# ===========================================================================
# Section 3 – Key distributions in ZdZd13TeV Nominal trees
# ===========================================================================
def section_distributions():
    print()
    print(SEP)
    print("SECTION 3: Key distributions in ZdZd13TeV Nominal trees")
    print(SEP)

    for label, path in DATA_13TEV.items():
        print(f"\n{'='*50}")
        print(f"Sample: {label}")
        print(f"{'='*50}")

        with uproot.open(path) as f:
            tree = f["Nominal/llllTree"]
            N = tree.num_entries
            print(f"Total events in file: {N}")

            # ---- Load branches ----
            branches_to_load = [
                # Event-level
                "passCleaning", "passNPV", "passTriggers", "llll_n",
                "evtWeight", "PileupWeight", "mc_channel_number",
                "truth_zdzd_avgM",
                # Per-lepton
                "l_tlv_pt", "l_tlv_eta", "l_pdgId", "l_quality",
                "l_overlaps",
                # Per-dilepton
                "ll_tlv_m",
                # Per-quadruplet
                "llll_tlv_m", "llll_avgM", "llll_dM",
                "llll_pdgIdSum", "llll_charge", "llll_dCharge",
                "llll_overlaps", "llll_nCTorSA",
                "llll_triggerMatched",
                "llll_min_sf_dR", "llll_min_of_dR",
                "llll_max_el_d0Sig", "llll_max_mu_d0Sig",
                "llll_l_isIsolCloseBy",
                "llll_ll1", "llll_ll2", "llll_alt_ll1", "llll_alt_ll2",
                "llll_vtx_reduced_chi2",
                "llll_l1", "llll_l2", "llll_l3", "llll_l4",
                "l_tlv_pt",
            ]
            data = tree.arrays(branches_to_load)

            # ----------------------------------------------------------------
            # 3a: Event-level preselection
            # ----------------------------------------------------------------
            print("\n--- 3a. Event-level preselection ---")
            n_clean  = int(ak.sum(data["passCleaning"]))
            n_npv    = int(ak.sum(data["passNPV"] & data["passCleaning"]))
            n_trig   = int(ak.sum((data["passTriggers"] != 0) & data["passNPV"] & data["passCleaning"]))
            n_quad   = int(ak.sum(data["llll_n"] > 0))

            print(f"  passCleaning:          {n_clean:6d} / {N} ({100*n_clean/N:.1f}%)")
            print(f"  + passNPV:             {n_npv:6d} / {N} ({100*n_npv/N:.1f}%)")
            print(f"  + passTriggers != 0:   {n_trig:6d} / {N} ({100*n_trig/N:.1f}%)")
            print(f"  events with llll_n>0:  {n_quad:6d} / {N} ({100*n_quad/N:.1f}%)")

            # ----------------------------------------------------------------
            # 3b: Lepton multiplicity
            # ----------------------------------------------------------------
            print("\n--- 3b. Object multiplicities ---")
            n_l    = ak.to_numpy(ak.num(data["l_tlv_pt"]))
            n_ll   = ak.to_numpy(ak.num(data["ll_tlv_m"]))
            n_llll = ak.to_numpy(data["llll_n"])

            for arr, name in [(n_l, "leptons (l)"), (n_ll, "dileptons (ll)"),
                               (n_llll, "quadruplets (llll)")]:
                unique, counts = np.unique(arr, return_counts=True)
                dist = ", ".join(f"n={u}: {c}" for u, c in zip(unique, counts))
                print(f"  {name:<25}: {dist}")

            # ----------------------------------------------------------------
            # 3c: Lepton kinematics
            # ----------------------------------------------------------------
            print("\n--- 3c. Lepton kinematics (all reco leptons, MeV -> GeV) ---")
            # Flatten jagged arrays
            l_pt  = ak.to_numpy(ak.flatten(data["l_tlv_pt"])) / 1000.0
            l_eta = ak.to_numpy(ak.flatten(data["l_tlv_eta"]))
            l_id  = ak.to_numpy(ak.flatten(data["l_pdgId"]))
            l_qual = ak.to_numpy(ak.flatten(data["l_quality"]))

            n_el = int(np.sum(np.abs(l_id) == 11))
            n_mu = int(np.sum(np.abs(l_id) == 13))
            print(f"  Total reco leptons: {len(l_pt)} (electrons: {n_el}, muons: {n_mu})")
            stats(l_pt, "pT", "GeV")
            stats(l_eta, "|η|", "")
            unique_qual, qual_counts = np.unique(l_qual, return_counts=True)
            print(f"  quality values: {dict(zip(unique_qual.tolist(), qual_counts.tolist()))}")

            # ----------------------------------------------------------------
            # 3d: Dilepton masses
            # ----------------------------------------------------------------
            print("\n--- 3d. Dilepton (ll) masses ---")
            ll_m = ak.to_numpy(ak.flatten(data["ll_tlv_m"])) / 1000.0  # GeV
            stats(ll_m, "m_ll", "GeV")
            print("  Distribution (GeV):")
            text_hist(ll_m, bins=12, lo=0, hi=130, width=35)

            # ----------------------------------------------------------------
            # 3e: Quadruplet properties (first/best quadruplet)
            # ----------------------------------------------------------------
            print("\n--- 3e. Selected quadruplet (llll[0]) properties ---")
            mask_has_quad = data["llll_n"] > 0
            n_has_quad = int(ak.sum(mask_has_quad))
            print(f"  Events with ≥1 quadruplet: {n_has_quad}")

            # Extract [0]-th quadruplet for each event that has one
            def quad1(arr):
                """Extract the first element of jagged array for events with ≥1 quad."""
                return ak.to_numpy(arr[mask_has_quad][:, 0])

            q_m4l  = quad1(data["llll_tlv_m"]) / 1000.0   # GeV
            q_avgM = quad1(data["llll_avgM"]) / 1000.0     # GeV
            q_dM   = quad1(data["llll_dM"]) / 1000.0       # GeV
            q_pdg  = quad1(data["llll_pdgIdSum"])
            q_chr  = quad1(data["llll_charge"])
            q_dchr = quad1(data["llll_dCharge"])
            q_ovlp = quad1(data["llll_overlaps"])
            q_ncto = quad1(data["llll_nCTorSA"])
            q_trig = quad1(data["llll_triggerMatched"])
            q_sfdr = quad1(data["llll_min_sf_dR"])
            q_ofdr = quad1(data["llll_min_of_dR"])
            q_eld0 = quad1(data["llll_max_el_d0Sig"])
            q_mud0 = quad1(data["llll_max_mu_d0Sig"])
            q_isol = quad1(data["llll_l_isIsolCloseBy"])
            q_chi2 = quad1(data["llll_vtx_reduced_chi2"])

            # Dilepton masses for the selected quadruplet (via index lookup)
            ll_m_all = data["ll_tlv_m"][mask_has_quad]
            q_ll1 = ak.to_numpy(data["llll_ll1"][mask_has_quad][:, 0])
            q_ll2 = ak.to_numpy(data["llll_ll2"][mask_has_quad][:, 0])
            q_alt1 = ak.to_numpy(data["llll_alt_ll1"][mask_has_quad][:, 0])
            q_alt2 = ak.to_numpy(data["llll_alt_ll2"][mask_has_quad][:, 0])

            # Index into jagged ll array to get the four dilepton masses
            m12 = np.array([float(ll_m_all[i][q_ll1[i]]) for i in range(len(q_ll1))]) / 1000.0
            m34 = np.array([float(ll_m_all[i][q_ll2[i]]) for i in range(len(q_ll2))]) / 1000.0
            m14 = np.array([float(ll_m_all[i][q_alt1[i]]) for i in range(len(q_alt1))]) / 1000.0
            m32 = np.array([float(ll_m_all[i][q_alt2[i]]) for i in range(len(q_alt2))]) / 1000.0

            # Sorted by convention: mab >= mcd  (ZdZd13TeV_alg.py make_masses)
            mab = np.maximum(m12, m34)
            mcd = np.minimum(m12, m34)

            stats(q_m4l,  "m_4l",   "GeV")
            stats(mab,    "mab (leading dilepton)",   "GeV")
            stats(mcd,    "mcd (subleading dilepton)", "GeV")
            stats(q_avgM, "avgM = (mab+mcd)/2",       "GeV")
            stats(q_dM,   "ΔM   = mab-mcd",           "GeV")

            print("\n  m_4l distribution (GeV):")
            text_hist(q_m4l, bins=12, lo=100, hi=140, width=35)
            print("\n  mab distribution (GeV):")
            text_hist(mab, bins=12, lo=0, hi=70, width=35)
            print("\n  mcd distribution (GeV):")
            text_hist(mcd, bins=12, lo=0, hi=70, width=35)

            # ----------------------------------------------------------------
            # 3f: Cutflow-related quantities
            # ----------------------------------------------------------------
            print("\n--- 3f. Cutflow-relevant flags for selected quadruplet ---")

            # pdgIdSum: 44=4e, 48=2e2mu, 52=4mu
            pdg_labels = {44: "4e", 48: "2e2mu", 52: "4mu"}
            for pdg, lbl in pdg_labels.items():
                frac = np.mean(q_pdg == pdg)
                print(f"  pdgIdSum={pdg} ({lbl:<5}): {frac*100:.1f}%")

            print(f"\n  charge==0 and dCharge==0 (SFOS):  {np.mean((q_chr==0)&(q_dchr==0))*100:.1f}%")
            print(f"  overlaps & 0x36 == 0 (no OR):     {np.mean((q_ovlp & 54)==0)*100:.1f}%")
            print(f"  triggerMatched != 0:               {np.mean(q_trig!=0)*100:.1f}%")
            print(f"  l_isIsolCloseBy == 15 (all isol):  {np.mean(q_isol==15)*100:.1f}%")
            print(f"  nCTorSA < 2 (muon type ok):        {np.mean(q_ncto<2)*100:.1f}%")
            print(f"  min_sf_dR > 0.1:                   {np.mean(q_sfdr>0.1)*100:.1f}%")
            print(f"  min_of_dR > 0.2:                   {np.mean(q_ofdr>0.2)*100:.1f}%")
            print(f"  max_el_d0Sig < 5:                  {np.mean(q_eld0<5)*100:.1f}%")
            print(f"  max_mu_d0Sig < 3:                  {np.mean(q_mud0<3)*100:.1f}%")

            # ----------------------------------------------------------------
            # 3g: Leading lepton pTs in selected quadruplet
            # ----------------------------------------------------------------
            print("\n--- 3g. Lepton pTs inside selected quadruplet ---")
            l_pt_all = data["l_tlv_pt"][mask_has_quad]
            q_l1 = ak.to_numpy(data["llll_l1"][mask_has_quad][:, 0])
            q_l2 = ak.to_numpy(data["llll_l2"][mask_has_quad][:, 0])
            q_l3 = ak.to_numpy(data["llll_l3"][mask_has_quad][:, 0])
            q_l4 = ak.to_numpy(data["llll_l4"][mask_has_quad][:, 0])

            pt1 = np.array([float(l_pt_all[i][q_l1[i]]) for i in range(len(q_l1))]) / 1000.0
            pt2 = np.array([float(l_pt_all[i][q_l2[i]]) for i in range(len(q_l2))]) / 1000.0
            pt3 = np.array([float(l_pt_all[i][q_l3[i]]) for i in range(len(q_l3))]) / 1000.0
            pt4 = np.array([float(l_pt_all[i][q_l4[i]]) for i in range(len(q_l4))]) / 1000.0

            # Sort descending within each event
            pts_all = np.column_stack([pt1, pt2, pt3, pt4])
            pts_sorted = np.sort(pts_all, axis=1)[:, ::-1]

            for i, rank in enumerate(["leading", "subleading", "subsubleading", "trailing"]):
                stats(pts_sorted[:, i], f"  pT({rank:<15})", "GeV")

            # ----------------------------------------------------------------
            # 3h: Vertex chi2 and other continuous discriminants
            # ----------------------------------------------------------------
            print("\n--- 3h. Potential BDT discriminant variables ---")
            chi2_real = q_chi2[q_chi2 > -999]
            chi2_sentinel_frac = np.mean(q_chi2 <= -999)
            print(f"  vtx_reduced_chi2: sentinel (-999) fraction = {chi2_sentinel_frac*100:.1f}%")
            stats(chi2_real, "  vtx_reduced_chi2 (non-sentinel)", "")
            stats(q_sfdr, "  min_sf_dR (same-flavour ΔR)", "rad")
            # min_of_dR uses sentinel 9999999 for same-flavour (4e/4mu) quads
            ofdr_real = q_ofdr[q_ofdr < 1e6]
            ofdr_sentinel_frac = np.mean(q_ofdr >= 1e6)
            print(f"  min_of_dR: sentinel (4e/4mu, no OF pairs) = {ofdr_sentinel_frac*100:.1f}% of quads")
            stats(ofdr_real, "  min_of_dR (2e2mu quads only)", "rad")
            # Ratio m34/m12 (used in MediumSR cut)
            good = (mab > 0)
            ratio = mcd[good] / mab[good]
            stats(ratio, "  mcd/mab ratio", "")

            # Truth info
            t_avg = ak.to_numpy(data["truth_zdzd_avgM"]) / 1000.0
            t_avg = t_avg[t_avg > 0]
            if len(t_avg) > 0:
                stats(t_avg, "  truth_zdzd_avgM", "GeV")
                print(f"    (confirms mZd ~ {np.mean(t_avg):.1f} ± {np.std(t_avg):.2f} GeV)")


# ===========================================================================
# Section 4 – PostProcessing histogram yields
# ===========================================================================
def section_pp_yields():
    print()
    print(SEP)
    print("SECTION 4: ZdZdPostProcessing histogram yields (events passing full cutflow)")
    print(SEP)

    for label, path in DATA_PP.items():
        print(f"\n[{label}]")
        with uproot.open(path) as f:
            all_keys = f.keys()
            # Nominal SR histograms are at depth 3: avgMll/REGION/CHANNEL_NUMBER
            sr_keys = [k for k in all_keys
                       if "SR" in k and k.count("/") == 2
                       and "down" not in k and "up" not in k]
            for k in sorted(sr_keys):
                h = f[k]
                vals, edges = h.to_numpy()
                integral = vals.sum()
                n_bins = len(vals)
                print(f"  {k.rstrip(';1'):<45}  integral={integral:.4f}  bins={n_bins}")


# ===========================================================================
# Section 5 – BDT formatting strategy
# ===========================================================================
def section_bdt_strategy():
    print()
    print(SEP)
    print("SECTION 5: Recommended strategy for formatting data for BDT training")
    print(SEP)

    strategy = """
SUMMARY OF DATA FORMATS
========================

Format A – ZdZd13TeV output (Nominal/llllTree TTree, *.root):
  • One entry per event; branches are a mix of event-level scalars and
    jagged (variable-length) arrays of per-lepton, per-dilepton, and
    per-quadruplet objects.
  • Rich kinematic information: all four leptons, all dileptons, all
    quadruplet candidates, plus truth matching and systematic weights.
  • Well-suited as the primary BDT training input after flattening.

Format B – ZdZdPostProcessing output (TH1 histograms only, *.root):
  • Contains only the final avgMll distribution in the SR and validation
    regions, after all cutflow selections have been applied.
  • No event-level information survives; cannot be used for BDT training.
  → Format B is UNSUITABLE for BDT training.

RECOMMENDED APPROACH: use Format A (ZdZd13TeV output)
======================================================

Step 1 – Event preselection
  Apply the same preselection as ZdZdPP_alg.cxx before the quadruplet loop:
    passCleaning == True
    passNPV      == True
    passTriggers != 0

Step 2 – Quadruplet selection
  The ZdZd13TeV output stores multiple quadruplet candidates per event
  (variable-length jagged llll_* arrays).  The ZdZdPP_alg.cxx takes the
  first quadruplet that passes all cuts.  Replicate the same logic:
    (a) Keep quadruplets with llll_charge==0 AND llll_dCharge==0 (SFOS)
    (b) (llll_overlaps & 0x36) == 0  (no lepton-lepton overlap removal)
    (c) Kinematic cuts on leading/sub/subsub lepton pT
    (d) llll_triggerMatched != 0
    (e) llll_min_sf_dR > 0.1 AND llll_min_of_dR > 0.2
    (f) llll_nCTorSA < 2
    Take llll[0] after sorting (already ranked in the tree).

  ⚠ For training purposes you may want to apply some (but not all) of these
  cuts, since the BDT can learn to separate signal/background using some of
  the same variables that the cuts are based on.  Consider at minimum
  requiring SFOS and the kinematic pT cuts, and leaving the remaining
  quantities as BDT input features.

Step 3 – Flatten to a fixed-size feature vector (one row per event)
  Extract scalar properties of the selected quadruplet and its four leptons.
  Suggested feature vector:

  QUADRUPLET-LEVEL (directly available as scalar after index selection):
    m_4l                — llll_tlv_m          (four-lepton invariant mass)
    avgM = (mab+mcd)/2  — llll_avgM           (avg dilepton mass; key SR var)
    dM   = mab−mcd      — llll_dM             (dilepton mass asymmetry)
    mab                 — ll_tlv_m[llll_ll1]  (leading dilepton mass)
    mcd                 — ll_tlv_m[llll_ll2]  (subleading dilepton mass)
    mad                 — ll_tlv_m[llll_alt_ll1]
    mbc                 — ll_tlv_m[llll_alt_ll2]
    mcd/mab             — derived ratio (used in MediumSR cut)
    min_sf_dR           — llll_min_sf_dR
    min_of_dR           — llll_min_of_dR
    vtx_reduced_chi2    — llll_vtx_reduced_chi2
    max_el_d0Sig        — llll_max_el_d0Sig
    max_mu_d0Sig        — llll_max_mu_d0Sig
    pdgIdSum            — llll_pdgIdSum        (44=4e / 48=2e2mu / 52=4mu;
                                                use one-hot or as integer)
    nCTorSA             — llll_nCTorSA

  LEPTON-LEVEL (look up via llll_l1..l4 indices):
    pT_1, pT_2, pT_3, pT_4   — sorted descending (lepton pTs in GeV)
    eta_1, ..., eta_4         — lepton pseudorapidities
    (optionally) quality_1..4, ptvarcone, topoetcone for isolation

  EVENT-LEVEL:
    averageInteractionsPerCrossing  (pile-up; can affect performance)

  LABELS / WEIGHTS (not BDT inputs, but needed for training):
    label      = 1 for signal (H→ZdZd→4ℓ), 0 for background (ZZ*→4ℓ)
    evtWeight  = evtWeight × PileupWeight × scaleFactor
    truth_zdzd_avgM  (signal only, to split by mZd for mZd-inclusive training)

Step 4 – Handle multiple mZd signal samples
  The signal MC is split by mZd (5–60 GeV) and MC campaign (mc20a, mc23d…).
  Options:
    (a) Train one mZd-inclusive BDT using all signal samples combined.
        Add truth_zdzd_avgM as an input feature so the BDT can learn
        the mass dependence.
    (b) Train a separate BDT per mZd point and interpolate.
    (c) Train a parameterised BDT with mZd as an explicit input (recommended
        for continuous mass scans; see approach in ATLAS parameterised ML).

Step 5 – Signal/background labelling
  signal    = H→ZdZd→4ℓ samples (mc_channel_number > 0 and is_signal flag)
  background = ZZ*→4ℓ and other relevant backgrounds
              (mc_channel_number > 0 and NOT signal)

CONCERNING SYSTEMATIC VARIATIONS
==================================
  The ZdZd13TeV file contains ~60+ systematic trees (EG_RESOLUTION_ALL1up,
  MUONS_CB1up, etc.).  For BDT training use only the Nominal tree.
  Systematics are applied at the evaluation/statistical analysis stage.

VARIABLE COMPLETENESS CHECK
============================
  All quantities used in ZdZdPP_alg.cxx cutflow are present in the
  ZdZd13TeV Nominal/llllTree:
    passCleaning, passNPV, passTriggers       ✓ (event-level scalars)
    llll_charge, llll_dCharge                 ✓ (per-quad jagged)
    llll_overlaps                             ✓
    l_tlv_pt (via llll_l1..l4)               ✓
    llll_triggerMatched                       ✓
    llll_min_sf_dR, llll_min_of_dR           ✓
    llll_nCTorSA                              ✓
    llll_l_isIsolCloseBy                      ✓
    l_quality (via llll_l1..l4)              ✓
    llll_max_el_d0Sig, llll_max_mu_d0Sig     ✓
    ll_tlv_m (m12, m34, m14, m32)            ✓ (via ll index lookup)
    llll_tlv_m (m4l)                          ✓
    evtWeight, PileupWeight, scaleFactor      ✓

SENTINEL VALUES TO BE AWARE OF
================================
  Several branches use sentinel values that must be handled before training:
  • llll_min_of_dR = 9999999  for 4e and 4mu quadruplets (no opposite-flavour
    pairs exist, so the minimum ΔR is undefined).  Options:
      - Use a flag feature: is_mixed_flavour = (pdgIdSum==48)
      - Replace with a fixed value (e.g. 999) and let the BDT treat it as
        a category; or drop the variable and use the flag instead.
  • llll_vtx_reduced_chi2 = -999  for events where no vertex fit converged.
    Replace with -1 or a fixed imputation value, or use a binary flag.
  • llll_max_el_d0Sig / llll_max_mu_d0Sig may be -999 if no electrons/muons
    are present in the quadruplet (e.g. max_el_d0Sig for 4mu quads).

PRACTICAL IMPLEMENTATION NOTES
================================
  • Use uproot + awkward-array to read the files in Python.
  • After selecting the best quadruplet per event, convert to a pandas
    DataFrame or numpy array for TMVA input via ROOT's PyROOT or by writing
    a new flat TTree.
  • TMVA expects a flat TTree with one row per event; write signal and
    background to separate trees (or use TMVA's built-in class labelling).
  • Keep m_4l and avgM as BDT inputs since both are key discriminants.
    The ZVeto cut (m12<64 GeV, m34<64 GeV) and MediumSR cut (mcd/mab ratio)
    can also be included as input features for the BDT to learn.
  • Consider NOT applying the HWindow (115–130 GeV) cut during training so
    the BDT has access to sideband events too.
"""
    print(strategy)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print()
    print(SEP)
    print("  ZdZd Run-3 BDT Data Format Analysis")
    print(SEP)
    print("\nExamining data in:")
    for label, path in DATA_13TEV.items():
        exists = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  ZdZd13TeV     [{label}]  {exists}")
    for label, path in DATA_PP.items():
        exists = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  PostProcessing [{label}]  {exists}")

    section_13tev_structure()
    section_distributions()
    section_pp_structure()
    section_pp_yields()
    section_bdt_strategy()

    print()
    print(SEP)
    print("Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    main()
