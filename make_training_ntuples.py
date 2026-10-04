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

INPUT MODES
-----------
Manifest mode (preferred, multi-process):
    --manifest data/training_ntuples/manifest_mc23a.csv
A manifest built by make_sample_manifest.py.  Each row names one Ntuple, its
physics process, its signal/background class and its normalisation factor
`scale_d`.  Process identity and `scale_d` are carried through to the Parquet
so Stage 2 can reproduce the physical background composition.

Legacy mode (single background class, kept for the two-class demo):
    --signal FILE [FILE ...] --background FILE [FILE ...]
Every --background file is labelled as one undifferentiated background process
("background") with scale_d = 1.0.

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
    (l1-l4 are stored in pT-descending order within each quadruplet candidate,
    matching the convention in ZdZdPP_alg.cxx.)

No further cuts are applied so that the remaining selection quantities
(isolation, dR, d0, trigger match, ...) survive as BDT input features.

EVENT WEIGHTS
-------------
Controlled by --signal-weight-mode, which follows the cf_v2 cutflow convention
(see claude/cf_v2_weighting.md):

    background : evtWeight * PileupWeight * llll_scaleFactor
    signal     : PileupWeight * llll_scaleFactor          (mode "cf_v2", default)
                 evtWeight * PileupWeight * llll_scaleFactor  (mode "legacy")

`evtWeight` MUST NOT be applied to the ZdZd signal.  The signal samples carry
generator weights of order 1e-18 and the MC-Request team advised ignoring them;
cf_v2 does.  Applying them makes the summed signal training weight ~1e-13
against ~1e+07 for background, i.e. it silently removes the signal from any
weighted training.  Mode "legacy" exists only to reproduce the pre-2026-10 runs.

SUBSAMPLING
-----------
--target-selected-per-process N stops reading a process once N events have been
selected from it, so a "small sample of each background" is possible without
reading tens of millions of entries.  0 means no cap.  Reading is chunked
(--chunk-size) and can be strided (--chunk-stride) so the sample is spread
through the merged file rather than taken from its first grid job alone.

--process-cap NAME=N overrides the cap for one process, which is how a run can
read some processes in full while keeping an expensive one partial.  For
example, every mc23a entry of signal, H_ZZ_4l, Tribosons and ttbarZ, with
ZZ_4l (49.7M entries in 701040 alone) still capped:

    --target-selected-per-process 0 --process-cap ZZ_4l=150000

Each sample records the fraction of its entries actually read as
`sampling_fraction`, which Stage 2 divides out.  Without that division a
lightly-sampled process is under-represented by exactly the factor it was
sampled by.  By default the fraction is the entry fraction
(entries_read / entries_total), which assumes the entries read are
representative of the whole file; --exact-sampling-fraction instead measures it
as sum(evtWeight) over entries read divided by the same sum over all entries,
at the cost of one extra single-branch pass over the file.

OUTPUT COLUMNS
--------------
Metadata / labels (not BDT inputs):
    label              int    1 = signal, 0 = background
    process            str    physics process (e.g. "ZZ_4l", "signal_mZd30")
    process_id         int    small integer code for `process` (0 = signal)
    mc_channel_number  int    MC dataset number
    eventNumber        int    event number (for debugging / cross-checks)
    truth_zdzd_avgM    float  MeV  truth avg Zd mass (signal only; 0 for bkg)
    evtWeight_total    float  per-event weight, see EVENT WEIGHTS above
    scale_d            float  L * sigma_eff / SumW_total for this sample
    sampling_fraction  float  fraction of the sample's entries read (0,1]

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
    min_sf_dR          float  rad  min dR between same-flavour leptons
    min_of_dR          float  rad  min dR between opp.-flavour leptons
                               *** SENTINEL: 9999999 for 4e / 4mu quads ***
    vtx_reduced_chi2   float  quadruplet vertex fit reduced chi2
                               *** SENTINEL: -999 when fit did not converge ***
    max_el_d0Sig       float  max |d0/sigma(d0)| for electrons in quadruplet
                               *** SENTINEL: 0.0 for 4mu quads (no electrons) ***
    max_mu_d0Sig       float  max |d0/sigma(d0)| for muons in quadruplet
                               *** SENTINEL: 0.0 for 4e quads (no muons) ***
    nCTorSA            int    number of CT or SA muons in quadruplet
    l_isIsolCloseBy    int    isolation bitmask (15 = all four leptons isolated)
    triggerMatched     int    trigger-matching bitmask (non-zero = matched)
    is_4e              int    1 if pdgIdSum == 44 (eeee),  else 0
    is_2e2mu           int    1 if pdgIdSum == 48 (eemumu), else 0
    is_4mu             int    1 if pdgIdSum == 52 (mumumumu), else 0
    pT_l1              float  MeV  leading lepton pT
    pT_l2              float  MeV  subleading lepton pT
    pT_l3              float  MeV  subsubleading lepton pT
    pT_l4              float  MeV  trailing lepton pT
    eta_l1             float  leading lepton eta   (ordered same as pT_l*)
    eta_l2             float  subleading lepton eta
    eta_l3             float  subsubleading lepton eta
    eta_l4             float  trailing lepton eta

Columns marked SENTINEL contain placeholder values for certain quadruplet
flavours and must be handled (excluded or imputed) in the training script
before being passed to TMVA.

A provenance sidecar <output>.samples.json records, per sample, the path,
process, class, scale_d, entries total/read, events selected, the sampling
fraction and how it was measured.

USAGE
-----
Manifest mode, 150k selected events per background process:
    python3 make_training_ntuples.py \\
        --manifest data/training_ntuples/manifest_mc23a.csv \\
        --target-selected-per-process 150000 --chunk-stride 3 \\
        --output   data/training_ntuples/sig3_irreducible_mc23a.parquet

Legacy two-class mode:
    python3 make_training_ntuples.py \\
        --signal     path/to/sig_mZd30.root \\
        --background path/to/ZZstar_bkg.root \\
        --output     data/training_ntuples/training.parquet

REQUIREMENTS
------------
    pip install uproot awkward numpy pandas pyarrow
"""

import argparse
import csv
import glob
import json
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
PDG_2E2MU = 48   # e e mu mu
PDG_4MU   = 52   # mu mu mu mu

# Default entries per read chunk.  Large enough to amortise uproot's basket
# decompression, small enough that a capped sample stops promptly.
DEFAULT_CHUNK_SIZE = 250_000

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

# Columns whose values contain known sentinels - flagged for the training script.
# Confirmed by inspecting both signal and background samples with uproot.
SENTINEL_NOTES = {
    "min_of_dR":        "9999999 for 4e/4mu quads (no opposite-flavour pairs)",
    "vtx_reduced_chi2": "-999 when vertex fit did not converge (~1-4% of events)",
    "max_el_d0Sig":     "0.0 for 4mu quads (no electrons present; computed via std::max with 0.0 fill)",
    "max_mu_d0Sig":     "0.0 for 4e quads (no muons present; computed via std::max with 0.0 fill)",
}

MANIFEST_REQUIRED_COLUMNS = ["path", "process", "sample_class", "scale_d"]

# Column dtypes for the per-chunk DataFrames.  Rows are converted chunk by
# chunk rather than accumulated as one list of dicts: a list of dicts costs
# roughly an order of magnitude more memory per row than a typed frame, and an
# uncapped run over the full mc23a set is ~2M rows.  Weight and bookkeeping
# columns stay float64 because scale_d spans ~1e-8 to ~4e-2 and the sums that
# Stage 2 takes over them need the headroom; the BDT features go to float32,
# which is what Stage 2 writes into the TTrees anyway.
INT_COLUMNS = [
    "label", "process_id", "mc_channel_number", "eventNumber",
    "nCTorSA", "l_isIsolCloseBy", "triggerMatched",
    "is_4e", "is_2e2mu", "is_4mu",
]
FLOAT64_COLUMNS = [
    "evtWeight_total", "scale_d", "sampling_fraction", "truth_zdzd_avgM",
]


# ---------------------------------------------------------------------------
# Sample description
# ---------------------------------------------------------------------------

class Sample:
    """One manifest row, resolved to concrete files."""

    def __init__(self, process, sample_class, paths, scale_d=1.0, dsid=None,
                 campaign=None, notes=""):
        self.process = process
        self.sample_class = sample_class          # "signal" | "background"
        self.paths = list(paths)
        self.scale_d = float(scale_d)
        self.dsid = dsid
        self.campaign = campaign
        self.notes = notes
        # Filled in by processing
        self.entries_total = 0
        self.entries_read = 0
        self.n_selected = 0
        self.sumw_read = 0.0
        self.sumw_all = None
        self.sampling_fraction = 1.0
        self.fraction_method = "entries"

    @property
    def label(self):
        return 1 if self.sample_class == "signal" else 0

    def __repr__(self):
        return (f"Sample({self.process!r}, {self.sample_class!r}, "
                f"{len(self.paths)} file(s), scale_d={self.scale_d:.4g})")


def resolve_paths(spec):
    """Expand a manifest path, which may be a glob (signal filenames carry a
    per-job ID so they cannot be written out literally).

    Background manifest rows hold exact merged-file paths; a glob there would
    risk double counting against a per-merged-file SumW_total, so only expand
    patterns that actually contain wildcards.
    """
    if any(ch in spec for ch in "*?["):
        matches = sorted(glob.glob(spec))
        if not matches:
            raise FileNotFoundError(f"No files match pattern: {spec}")
        return matches
    if not os.path.exists(spec):
        raise FileNotFoundError(f"ROOT file not found: {spec}")
    return [spec]


def read_manifest(path):
    """Read a make_sample_manifest.py CSV into a list of Samples."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFEST_REQUIRED_COLUMNS
                   if c not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"Manifest {path} is missing column(s): {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest {path} contains no rows")

    samples = []
    for row in rows:
        sample_class = row["sample_class"].strip().lower()
        if sample_class not in ("signal", "background"):
            raise ValueError(
                f"Manifest {path}: sample_class must be 'signal' or "
                f"'background', got {row['sample_class']!r}")
        scale_d = float(row["scale_d"]) if row["scale_d"].strip() else 1.0
        if scale_d <= 0:
            raise ValueError(
                f"Manifest {path}: non-positive scale_d for {row['path']}")
        samples.append(Sample(
            process=row["process"].strip(),
            sample_class=sample_class,
            paths=resolve_paths(row["path"].strip()),
            scale_d=scale_d,
            dsid=int(row["dsid"]) if row.get("dsid", "").strip() else None,
            campaign=row.get("campaign", "").strip() or None,
            notes=row.get("notes", "").strip(),
        ))
    return samples


def samples_from_legacy_args(signal_files, background_files):
    """Build Samples from the old --signal / --background file lists."""
    samples = []
    for path in signal_files or []:
        samples.append(Sample("signal", "signal", resolve_paths(path)))
    if background_files:
        paths = []
        for path in background_files:
            paths.extend(resolve_paths(path))
        samples.append(Sample("background", "background", paths))
    return samples


def assign_process_ids(samples):
    """Map process names to small integers, signal first (0, 1, ... then bkg).

    Written into the Parquet as `process_id` and carried into TMVA as a
    spectator variable, so the per-process breakdown survives into the
    TrainTree/TestTree that Stage 3 reads.
    """
    sig = sorted({s.process for s in samples if s.sample_class == "signal"})
    bkg = sorted({s.process for s in samples if s.sample_class == "background"})
    return {name: i for i, name in enumerate(sig + bkg)}


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
# Per-chunk processing
# ---------------------------------------------------------------------------

def rows_from_chunk(data, sample, process_id, use_evt_weight_for_signal):
    """Apply preselection + quadruplet selection to one chunk of entries.

    Returns (rows, n_presel, n_no_cands, n_no_pass).  `data` is the awkward
    record array returned by tree.arrays() for an entry range.
    """
    label = sample.label

    # -------------------------------------------------------------------------
    # Event-level preselection
    # -------------------------------------------------------------------------
    evt_mask = (
        data["passCleaning"]
        & data["passNPV"]
        & (data["passTriggers"] != 0)
    )
    n_presel = int(ak.sum(evt_mask))
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
    # A Python loop is used for clarity; see README "Known limitations" for the
    # vectorisation that full-statistics training will need.

    rows       = []
    n_no_cands = 0
    n_no_pass  = 0

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

        # Total event weight.  The ZdZd signal generator weight (~1e-18) is
        # dropped unless the legacy mode was asked for -- see EVENT WEIGHTS.
        if label == 1 and not use_evt_weight_for_signal:
            weight = pu_wts[i] * sfs[i][q]
        else:
            weight = evt_wts[i] * pu_wts[i] * sfs[i][q]

        pdg_sum = pdgsums[i][q]

        rows.append({
            # --- Metadata / labels ---
            "label":             label,
            "process":           sample.process,
            "process_id":        process_id,
            "mc_channel_number": mc_chans[i],
            "eventNumber":       evt_nums[i],
            "truth_zdzd_avgM":   truth_ms[i],
            "evtWeight_total":   weight,
            "scale_d":           sample.scale_d,
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
            # --- Lepton kinematics (MeV; l1=leading -> l4=trailing) ---
            "pT_l1":             lpt_i[l1],
            "pT_l2":             lpt_i[l2],
            "pT_l3":             lpt_i[l3],
            "pT_l4":             lpt_i[l4],
            "eta_l1":            leta_i[l1],
            "eta_l2":            leta_i[l2],
            "eta_l3":            leta_i[l3],
            "eta_l4":            leta_i[l4],
        })

    return rows, n_presel, n_no_cands, n_no_pass


# ---------------------------------------------------------------------------
# Chunk planning and per-sample processing
# ---------------------------------------------------------------------------

def rows_to_frame(rows):
    """Convert a chunk's row dicts to a typed DataFrame (see the dtype note)."""
    df = pd.DataFrame(rows)
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in df.columns:
        if col in INT_COLUMNS or col in FLOAT64_COLUMNS or col == "process":
            continue
        df[col] = df[col].astype("float32")
    for col in FLOAT64_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    if "process" in df.columns:
        df["process"] = df["process"].astype("category")
    return df


def parse_process_caps(specs):
    """Parse --process-cap NAME=N into {process: cap}."""
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"ERROR: bad --process-cap '{spec}'; expected PROCESS=N")
        name, value = spec.split("=", 1)
        try:
            cap = int(value)
        except ValueError:
            raise SystemExit(
                f"ERROR: bad --process-cap '{spec}'; N must be an integer")
        if cap < 0:
            raise SystemExit(f"ERROR: --process-cap '{spec}' must be >= 0")
        out[name.strip()] = cap
    return out


def plan_chunks(n_entries, chunk_size, stride):
    """Return the (start, stop) entry ranges to read, in read order.

    With stride > 1 the chunks are spread evenly through the file: the merged
    Ntuples are hadd-ed from grid jobs in file order, so the first N entries are
    not a random sample of the production.  Striding visits every `stride`-th
    chunk first, then fills in the ones it skipped, so stopping early still
    leaves a sample drawn from across the whole file while reading every entry
    exactly once if the cap is never reached.
    """
    starts = list(range(0, n_entries, chunk_size))
    if stride > 1:
        ordered = []
        for offset in range(stride):
            ordered.extend(starts[offset::stride])
        starts = ordered
    return [(a, min(a + chunk_size, n_entries)) for a in starts]


def sum_evt_weight(tree, chunk_size):
    """Sum evtWeight over every entry of a tree, reading only that branch."""
    total = 0.0
    for a in range(0, tree.num_entries, chunk_size):
        arr = tree["evtWeight"].array(
            library="np", entry_start=a,
            entry_stop=min(a + chunk_size, tree.num_entries))
        total += float(np.sum(arr))
    return total


def process_sample(sample, process_id, tree_path, target_selected, chunk_size,
                   chunk_stride, use_evt_weight_for_signal, exact_fraction):
    """Read one Sample (possibly several files) and return its selected rows."""
    print(f"\n  Sample : {sample.process}  [{sample.sample_class}]"
          f"  scale_d = {sample.scale_d:.6g}")
    if sample.notes:
        print(f"  Note   : {sample.notes}")

    frames = []
    n_rows = 0
    n_presel = n_no_cands = n_no_pass = 0

    for path in sample.paths:
        if target_selected and n_rows >= target_selected:
            print(f"  File   : {os.path.basename(path)}  (skipped, cap reached)")
            continue

        print(f"  File   : {path}")
        with uproot.open(path) as f:
            if tree_path not in f:
                raise KeyError(
                    f"Tree '{tree_path}' not found in {path}.\n"
                    f"Available keys: {[k for k in f.keys() if 'Tree' in k]}"
                )
            tree = f[tree_path]
            n_total = tree.num_entries
            sample.entries_total += n_total
            print(f"    Entries in tree: {n_total:,}")

            # Check for missing required branches
            available = set(tree.keys())
            missing_req = [b for b in BRANCHES_REQUIRED if b not in available]
            if missing_req:
                raise KeyError(
                    f"Required branches not found in {path}: {missing_req}")

            branches_to_load = BRANCHES_REQUIRED + [
                b for b in BRANCHES_OPTIONAL if b in available
            ]
            missing_opt = [b for b in BRANCHES_OPTIONAL if b not in available]
            if missing_opt:
                print(f"    Note: optional branches absent, defaulting to 0: "
                      f"{missing_opt}")

            if exact_fraction:
                w_all = sum_evt_weight(tree, chunk_size)
                sample.sumw_all = (sample.sumw_all or 0.0) + w_all
                print(f"    sum(evtWeight) over all entries: {w_all:.6g}")

            n_read_file = 0
            for a, b in plan_chunks(n_total, chunk_size, chunk_stride):
                data = tree.arrays(branches_to_load, entry_start=a, entry_stop=b)
                sample.sumw_read += float(np.sum(ak.to_numpy(data["evtWeight"])))
                n_read_file += (b - a)

                chunk_rows, npre, nc, npass = rows_from_chunk(
                    data, sample, process_id, use_evt_weight_for_signal)
                n_presel += npre
                n_no_cands += nc
                n_no_pass += npass
                if chunk_rows:
                    frames.append(rows_to_frame(chunk_rows))
                    n_rows += len(chunk_rows)
                del data, chunk_rows

                if target_selected and n_rows >= target_selected:
                    print(f"    Reached cap of {target_selected:,} selected "
                          f"events after {n_read_file:,} entries")
                    break

            sample.entries_read += n_read_file

    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()

    # Trim to the cap so the sampling fraction and the row count agree.
    if target_selected and len(df) > target_selected:
        df = df.iloc[:target_selected].reset_index(drop=True)

    sample.n_selected = len(df)

    if sample.entries_total == 0:
        raise ValueError(f"Sample {sample.process} has no entries")

    if exact_fraction and sample.sumw_all not in (None, 0.0):
        sample.sampling_fraction = sample.sumw_read / sample.sumw_all
        sample.fraction_method = "sum(evtWeight)"
    else:
        sample.sampling_fraction = sample.entries_read / sample.entries_total
        sample.fraction_method = "entries"

    if not 0.0 < sample.sampling_fraction <= 1.0 + 1e-9:
        print(f"    WARNING: sampling fraction {sample.sampling_fraction:.6g} "
              f"outside (0, 1]; falling back to the entry fraction")
        sample.sampling_fraction = sample.entries_read / sample.entries_total
        sample.fraction_method = "entries (fallback)"
    sample.sampling_fraction = min(sample.sampling_fraction, 1.0)

    if len(df):
        df["sampling_fraction"] = np.float64(sample.sampling_fraction)

    print(f"    Entries read            : {sample.entries_read:,}"
          f" / {sample.entries_total:,}")
    print(f"    Pass preselection       : {n_presel:,}"
          f"  (clean + NPV + trigger)")
    print(f"    No quadruplet candidates: {n_no_cands:,}")
    print(f"    Candidates fail SFOS/pT : {n_no_pass:,}")
    print(f"    Selected events         : {sample.n_selected:,}")
    print(f"    Sampling fraction       : {sample.sampling_fraction:.6g}"
          f"  ({sample.fraction_method})")

    return df


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


def print_process_table(df):
    """Per-process event counts, raw and normalised weight sums, and N_eff."""
    sep = "=" * 92
    print(f"\n{sep}")
    print("Per-process summary")
    print(sep)
    header = (f"  {'process':<18}{'class':<11}{'events':>10}{'sum w':>13}"
              f"{'expected yield':>16}{'N_eff':>10}{'neg w':>9}")
    print(header)

    for proc, g in df.groupby("process", sort=True, observed=True):
        cls = "signal" if int(g["label"].iloc[0]) == 1 else "background"
        w = g["evtWeight_total"].to_numpy(dtype=float)
        norm = w * g["scale_d"].to_numpy(dtype=float) \
                 / g["sampling_fraction"].to_numpy(dtype=float)
        sw, sw2 = norm.sum(), np.square(norm).sum()
        n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
        n_neg = int((w < 0).sum())
        print(f"  {proc:<18}{cls:<11}{len(g):>10,}{w.sum():>13.4g}"
              f"{sw:>16.6g}{n_eff:>10.1f}"
              f"{100.0 * n_neg / len(g):>8.2f}%")

    print("\n  'expected yield' = sum(w * scale_d / sampling_fraction), i.e. the")
    print("  weight Stage 2 trains with.  For background it is the yield at the")
    print("  campaign luminosity; for signal scale_d = 1 and it is not a yield.")


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


def write_sidecar(path, samples, process_ids, args):
    """Write per-sample provenance next to the Parquet."""
    payload = {
        "output": os.path.abspath(args.output),
        "tree": args.tree,
        "signal_weight_mode": args.signal_weight_mode,
        "target_selected_per_process": args.target_selected_per_process,
        "process_cap": parse_process_caps(args.process_cap),
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "exact_sampling_fraction": bool(args.exact_sampling_fraction),
        "manifest": os.path.abspath(args.manifest) if args.manifest else None,
        "process_ids": process_ids,
        "samples": [
            {
                "process": s.process,
                "process_id": process_ids[s.process],
                "sample_class": s.sample_class,
                "dsid": s.dsid,
                "campaign": s.campaign,
                "paths": s.paths,
                "scale_d": s.scale_d,
                "entries_total": s.entries_total,
                "entries_read": s.entries_read,
                "n_selected": s.n_selected,
                "sumw_read": s.sumw_read,
                "sumw_all": s.sumw_all,
                "sampling_fraction": s.sampling_fraction,
                "fraction_method": s.fraction_method,
                "notes": s.notes,
            }
            for s in samples
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Written: {path}  (per-sample provenance)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Stage 1 of the TMVA pipeline: convert ZdZd13TeV ROOT ntuples "
            "to a flat Parquet file for BDT training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest", metavar="CSV", default=None,
        help="Sample manifest from make_sample_manifest.py (preferred). "
             "Mutually exclusive with --signal/--background.",
    )
    p.add_argument(
        "--signal", nargs="+", metavar="FILE",
        help="Legacy mode: one or more signal ROOT files (H->ZdZd->4l MC).",
    )
    p.add_argument(
        "--background", nargs="+", metavar="FILE", default=[],
        help="Legacy mode: one or more background ROOT files, all labelled as "
             "a single undifferentiated background process.",
    )
    p.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output Parquet file path.",
    )
    p.add_argument(
        "--tree", default=TREE_PATH,
        help=f"TTree path within the ROOT file (default: {TREE_PATH}).",
    )
    p.add_argument(
        "--signal-weight-mode", choices=["cf_v2", "legacy"], default="cf_v2",
        help="cf_v2 (default): signal weight = PileupWeight * llll_scaleFactor, "
             "dropping the ~1e-18 ZdZd generator weight, as cf_v2 does. "
             "legacy: also multiply by evtWeight (reproduces pre-2026-10 runs; "
             "sinks the signal in any weighted training).",
    )
    p.add_argument(
        "--target-selected-per-process", type=int, default=0, metavar="N",
        help="Stop reading a process once N events have been selected from it. "
             "0 (default) reads every entry.",
    )
    p.add_argument(
        "--process-cap", nargs="*", default=[], metavar="PROC=N",
        help="Override --target-selected-per-process for one process, e.g. "
             "'ZZ_4l=150000'. Use 0 for no cap. Repeatable; lets a run read "
             "some processes in full while keeping an expensive one partial.",
    )
    p.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, metavar="N",
        help=f"Entries per read chunk (default {DEFAULT_CHUNK_SIZE:,}).",
    )
    p.add_argument(
        "--chunk-stride", type=int, default=1, metavar="K",
        help="Visit every K-th chunk first, so a capped sample is drawn from "
             "across the merged file rather than from its first grid job. "
             "Default 1 (sequential).",
    )
    p.add_argument(
        "--exact-sampling-fraction", action="store_true",
        help="Measure the sampling fraction as sum(evtWeight) read / "
             "sum(evtWeight) total instead of the entry fraction. Costs one "
             "extra single-branch pass over each file.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.manifest and (args.signal or args.background):
        sys.exit("ERROR: --manifest cannot be combined with "
                 "--signal/--background.")
    if not args.manifest and not args.signal and not args.background:
        sys.exit("ERROR: provide --manifest, or at least one --signal or "
                 "--background file.")
    if args.chunk_size < 1:
        sys.exit("ERROR: --chunk-size must be >= 1.")
    if args.chunk_stride < 1:
        sys.exit("ERROR: --chunk-stride must be >= 1.")

    if args.manifest:
        if not os.path.exists(args.manifest):
            sys.exit(f"ERROR: manifest not found: {args.manifest}")
        samples = read_manifest(args.manifest)
        print(f"Manifest: {args.manifest}  ({len(samples)} samples)")
    else:
        samples = samples_from_legacy_args(args.signal, args.background)
        print(f"Legacy mode: {len(samples)} sample group(s)")

    process_ids = assign_process_ids(samples)
    print(f"Process IDs: {process_ids}")
    print(f"Signal weight mode: {args.signal_weight_mode}")
    if args.signal_weight_mode == "legacy":
        print("  WARNING: legacy mode applies the ~1e-18 ZdZd generator weight "
              "to signal; see EVENT WEIGHTS in this script's docstring.")

    # The per-process cap is a cap per PROCESS, not per file, so group the
    # manifest rows by process before reading.
    by_process = {}
    for s in samples:
        by_process.setdefault(s.process, []).append(s)

    process_caps = parse_process_caps(args.process_cap)
    unknown_caps = [p for p in process_caps if p not in by_process]
    if unknown_caps:
        print(f"  WARNING: --process-cap names not in the manifest: "
              f"{unknown_caps}  (known: {sorted(by_process)})")

    frames = []
    use_evt_w_sig = (args.signal_weight_mode == "legacy")

    for proc in sorted(by_process, key=lambda p: process_ids[p]):
        proc_samples = by_process[proc]
        proc_cap = process_caps.get(proc, args.target_selected_per_process)
        if proc in process_caps:
            print(f"\n  [{proc}] cap overridden: "
                  f"{proc_cap:,} selected events"
                  if proc_cap else f"\n  [{proc}] cap overridden: no cap")
        # The cap is per PROCESS but it is spent per SAMPLE, because each DSID
        # of a process has its own scale_d and its own sampling fraction.
        # Splitting it evenly (and rolling unused quota forward) keeps every
        # DSID represented; filling the cap from the first DSID alone would
        # silently drop the others' cross-sections from the background mix.
        remaining = proc_cap
        n_left = len(proc_samples)
        for s in proc_samples:
            if proc_cap:
                quota = max(1, -(-remaining // n_left)) if remaining > 0 else 0
            else:
                quota = 0
            if proc_cap and quota == 0:
                print(f"\n  Sample : {s.process} [{s.sample_class}] "
                      f"- skipped, process cap already filled")
                n_left -= 1
                continue
            sample_df = process_sample(
                s, process_ids[proc], args.tree, quota,
                args.chunk_size, args.chunk_stride, use_evt_w_sig,
                args.exact_sampling_fraction,
            )
            if len(sample_df):
                frames.append(sample_df)
            if proc_cap:
                remaining -= len(sample_df)
            n_left -= 1

    if not frames:
        print("ERROR: no events survived selection. "
              "Check input files and selection cuts.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    del frames
    if "process" in df.columns:
        df["process"] = df["process"].astype("category")
    print_summary(df)
    print_process_table(df)

    # Write output, creating intermediate directories if needed
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output, index=False)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nWritten: {args.output}  ({size_kb:.0f} kB, {len(df):,} rows, "
          f"{len(df.columns)} columns)")

    write_sidecar(args.output + ".samples.json", samples, process_ids, args)


if __name__ == "__main__":
    main()
