#!/usr/bin/env bash
#
# run_bdt_irreducible_lxplus.sh
# =============================
# Train the tmva_ZdZd_R3 BDT on signal plus the irreducible backgrounds, using
# the merged Run 3 Ntuples on eos.  Four stages, selectable with STAGES=:
#
#   Stage 0  make_sample_manifest.py     production registry -> manifest + scale_d
#   Stage 1  make_training_ntuples.py    ROOT (eos)          -> Parquet
#   Stage 2  train_bdt.py                Parquet             -> TMVA BDT
#   Stage 3  plot_bdt_output.py          TMVA output         -> PNG plots
#
# Stages communicate through files, not shell state, so any subset can be run:
#
#   STAGES="0 1"   build the manifest and the Parquet
#   STAGES="2"     train from an existing Parquet
#   STAGES="3"     re-plot from an existing TMVA output
#   STAGES="2 3"   train and plot
#
# Stage 2 needs Stage 1's Parquet on disk; Stage 3 needs Stage 2's
# TMVAClassification.root, weights XML and process_map.json.  Neither needs the
# earlier stage to have run in the same invocation.
#
# ENVIRONMENT
#   This script sets up NOTHING.  Run `setupATLAS` (and whatever it brings in)
#   BEFORE invoking it; the script only checks that what it needs is on the
#   PATH and importable, and reports clearly if it is not.  Required: python3
#   with uproot, awkward, numpy, pandas, pyarrow and matplotlib for Stages 0/1/3,
#   and PyROOT with TMVA for Stage 2.
#
# RUNS
#   RUN_NAME names the run and keeps its outputs separate, so an earlier run is
#   not overwritten:
#     <project>/data/training_ntuples/manifest_<RUN_NAME>_<CAMPAIGN>.csv
#     <project>/data/training_ntuples/<RUN_NAME>_<CAMPAIGN>.parquet
#     <repo>/runs/<RUN_NAME>/            TMVA output, weights, process map, plots
#
# DEFAULT RUN: partial_ZZ_4l  (mc23a)
#   Signal     : the HIGH MASS grid, 10 DSIDs 561508-561517
#                (mZd 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 GeV), read in full.
#                mZd < 15 GeV is the Low Mass region and is excluded: its signal
#                kinematics differ enough (strongly collimated dilepton pairs)
#                that training across the split blurs the classifier.
#   Background : H_ZZ_4l (8 samples), Tribosons (4) and ttbarZ (1) read in FULL,
#                with no event cap; ZZ_4l kept PARTIAL at 150k selected events,
#                since 701040 alone is 49.7M entries.  Hence the run name.
#                701185 and 701190 remain excluded on m4l-overlap grounds.
#
# NORMALISATION
#   Stage 0 computes scale_d = L * sigma * k * eps_filt / SumW_total per sample
#   and Stage 2 applies it, so the background processes enter in proportion to
#   their expected yield rather than to their MC statistics.  603293 mc23a has no
#   measured SumW_total and uses the accepted estimate (~18767); the manifest
#   flags it and Stage 0 prints a warning.
#
# Usage:
#   setupATLAS
#   bash run_bdt_irreducible_lxplus.sh
# Optional overrides (environment variables):
#   STAGES="0 1 2 3"     which stages to run (default: all four)
#   RUN_NAME=...         run label         (default: partial_ZZ_4l)
#   PROJECT_DIR=...      root of run3_ZdZd_project (default: parent of this script)
#   INPUT_DIR=...        normalisation CSVs (default: <repo>/normalisation_inputs)
#   CAMPAIGN=mc23a       MC campaign
#   SIGNAL_DSIDS="all"   signal DSIDs, or "all" for the whole mZd grid
#   SIGNAL_MIN_MZD=15    drop signal samples below this mZd in GeV ("" for none)
#   SIGNAL_MAX_MZD=      drop signal samples above this mZd in GeV ("" for none)
#   TARGET_PER_PROCESS=  default cap on selected events per process (0 = no cap)
#   PROCESS_CAP=         per-process overrides, e.g. "ZZ_4l=150000"
#   CHUNK_STRIDE=        spread a capped sample across each merged file (default 4)
#   WEIGHTING=           normalised | equal-process | raw  (default normalised)
#   SKIP_ENV_CHECK=1     skip the dependency check below
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TMVA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${TMVA_DIR}/.." && pwd)}"
INPUT_DIR="${INPUT_DIR:-${TMVA_DIR}/normalisation_inputs}"
STAGES="${STAGES:-0 1 2 3}"
RUN_NAME="${RUN_NAME:-partial_ZZ_4l}"
CAMPAIGN="${CAMPAIGN:-mc23a}"
TARGET_PER_PROCESS="${TARGET_PER_PROCESS:-0}"
PROCESS_CAP="${PROCESS_CAP:-ZZ_4l=150000}"
CHUNK_STRIDE="${CHUNK_STRIDE:-4}"
WEIGHTING="${WEIGHTING:-normalised}"

MANIFEST="${PROJECT_DIR}/data/training_ntuples/manifest_${RUN_NAME}_${CAMPAIGN}.csv"
NTUPLE_OUT="${PROJECT_DIR}/data/training_ntuples/${RUN_NAME}_${CAMPAIGN}.parquet"
RUN_DIR="${TMVA_DIR}/runs/${RUN_NAME}"

# eos Ntuple areas
EOS_BASE="/eos/atlas/atlascerngroupdisk/phys-hmbs/hlrs/ZdZd/ZdZd13TeV_Ntuples"
SIG_DIR="${SIG_DIR:-${EOS_BASE}/signal_Ntuples/${CAMPAIGN}_p6697_noSyst}"

# Signal DSIDs.  "all" is the full mZd grid, 561504-561517; the mass cut below
# then keeps the High Mass side of the analysis' 15 GeV Low/High Mass split.
SIGNAL_DSIDS="${SIGNAL_DSIDS:-all}"
SIGNAL_MIN_MZD="${SIGNAL_MIN_MZD-15}"
SIGNAL_MAX_MZD="${SIGNAL_MAX_MZD-}"

stage_enabled () { [[ " ${STAGES} " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# Pre-flight (no setup is performed here - run setupATLAS first)
# ---------------------------------------------------------------------------
echo "=== Pre-flight ==="
echo "  run         : ${RUN_NAME}  (${CAMPAIGN})"
echo "  stages      : ${STAGES}"
echo "  repo        : ${TMVA_DIR}"
echo "  project dir : ${PROJECT_DIR}"
echo "  inputs      : ${INPUT_DIR}"
echo "  run dir     : ${RUN_DIR}"
echo "  python3     : $(command -v python3 || echo 'NOT FOUND')"
echo "  root        : $(command -v root || echo 'not found (Stage 2 uses PyROOT)')"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: no python3 on PATH. Run setupATLAS first." >&2
    exit 1
fi

if [[ -z "${STAGES// /}" ]]; then
    echo "ERROR: STAGES is empty; nothing to do." >&2
    exit 1
fi
for s in ${STAGES}; do
    case "${s}" in
        0|1|2|3) ;;
        *) echo "ERROR: unknown stage '${s}' in STAGES; valid stages are 0 1 2 3." >&2
           exit 1 ;;
    esac
done

if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
    python3 - <<'PY' || { echo "       Run setupATLAS (and any lsetup it implies) before this script," >&2; echo "       or set SKIP_ENV_CHECK=1 to proceed anyway." >&2; exit 1; }
import importlib, sys
needed = {
    "numpy": "Stages 0-3", "pandas": "Stages 1-2", "uproot": "Stages 1, 3",
    "awkward": "Stage 1", "pyarrow": "Stage 1 (Parquet)",
    "matplotlib": "Stage 3", "ROOT": "Stage 2 (PyROOT + TMVA)",
}
missing = []
for mod, where in needed.items():
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<12} {getattr(m, '__version__', 'ok'):<12} ({where})", flush=True)
    except Exception:
        missing.append((mod, where))
if missing:
    print("\nERROR: missing Python module(s):", file=sys.stderr)
    for mod, where in missing:
        print(f"  {mod:<12} needed for {where}", file=sys.stderr)
    sys.exit(1)
PY
fi

if stage_enabled 0; then
    for f in cutflow_inputs.csv crossSections_run3.csv sumw_total_p7266.csv; do
        if [[ ! -f "${INPUT_DIR}/${f}" ]]; then
            echo "ERROR: ${f} not found under ${INPUT_DIR}" >&2
            echo "       See normalisation_inputs/README.md." >&2
            exit 1
        fi
    done
    if [[ ! -d "${SIG_DIR}" ]]; then
        echo "ERROR: signal Ntuple directory not found: ${SIG_DIR}" >&2
        exit 1
    fi
fi

if stage_enabled 1 && [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Stage 1 needs the manifest, which does not exist:" >&2
    echo "       ${MANIFEST}" >&2
    echo "       Run Stage 0 first (STAGES=\"0 1 ...\")." >&2
    exit 1
fi
if stage_enabled 2 && [[ ! -f "${NTUPLE_OUT}" ]]; then
    echo "ERROR: Stage 2 needs the Parquet, which does not exist:" >&2
    echo "       ${NTUPLE_OUT}" >&2
    echo "       Run Stage 1 first (STAGES=\"1 2 ...\")." >&2
    exit 1
fi
if stage_enabled 3 && [[ ! -f "${RUN_DIR}/TMVAClassification.root" ]]; then
    echo "ERROR: Stage 3 needs Stage 2's output, which does not exist:" >&2
    echo "       ${RUN_DIR}/TMVAClassification.root" >&2
    echo "       Run Stage 2 first (STAGES=\"2 3\")." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 0 - build the sample manifest
# ---------------------------------------------------------------------------
if stage_enabled 0; then
    echo
    echo "=== Stage 0: make_sample_manifest.py ==="
    mkdir -p "$(dirname "${MANIFEST}")"
    python3 "${TMVA_DIR}/make_sample_manifest.py" \
        --cutflow-inputs "${INPUT_DIR}/cutflow_inputs.csv" \
        --cross-sections "${INPUT_DIR}/crossSections_run3.csv" \
        --sumw           "${INPUT_DIR}/sumw_total_p7266.csv" \
        --campaign       "${CAMPAIGN}" \
        --signal-dir     "${SIG_DIR}" \
        --signal-dsids   ${SIGNAL_DSIDS} \
        ${SIGNAL_MIN_MZD:+--signal-min-mzd ${SIGNAL_MIN_MZD}} \
        ${SIGNAL_MAX_MZD:+--signal-max-mzd ${SIGNAL_MAX_MZD}} \
        --output         "${MANIFEST}"
fi

# ---------------------------------------------------------------------------
# Stage 1 - build the training Parquet
# ---------------------------------------------------------------------------
if stage_enabled 1; then
    echo
    echo "=== Stage 1: make_training_ntuples.py ==="
    echo "  default cap per process : ${TARGET_PER_PROCESS} (0 = no cap)"
    echo "  per-process overrides   : ${PROCESS_CAP:-none}"
    echo "  chunk stride            : ${CHUNK_STRIDE}"
    python3 "${TMVA_DIR}/make_training_ntuples.py" \
        --manifest "${MANIFEST}" \
        --target-selected-per-process "${TARGET_PER_PROCESS}" \
        ${PROCESS_CAP:+--process-cap ${PROCESS_CAP}} \
        --chunk-stride "${CHUNK_STRIDE}" \
        --output   "${NTUPLE_OUT}"
fi

# ---------------------------------------------------------------------------
# Stage 2 - train the BDT
# ---------------------------------------------------------------------------
if stage_enabled 2; then
    echo
    echo "=== Stage 2: train_bdt.py ==="
    mkdir -p "${RUN_DIR}"
    python3 "${TMVA_DIR}/train_bdt.py" \
        --input      "${NTUPLE_OUT}" \
        --output-dir "${RUN_DIR}" \
        --weighting  "${WEIGHTING}"
fi

# ---------------------------------------------------------------------------
# Stage 3 - plots
# ---------------------------------------------------------------------------
if stage_enabled 3; then
    echo
    echo "=== Stage 3: plot_bdt_output.py ==="
    python3 "${TMVA_DIR}/plot_bdt_output.py" \
        --output-dir "${RUN_DIR}" \
        --top-n 6
fi

echo
echo "=== Done (${RUN_NAME}, stages ${STAGES}). Outputs under ${RUN_DIR} ==="
