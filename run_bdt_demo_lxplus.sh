#!/usr/bin/env bash
#
# run_bdt_demo_lxplus.sh
# ======================
# End-to-end demonstration of the tmva_ZdZd_R3 BDT on lxplus, using the merged
# Run 3 Ntuples on eos.  Runs all three stages:
#
#   Stage 1  make_training_ntuples.py   ROOT (eos) -> Parquet
#   Stage 2  train_bdt.py               Parquet    -> TMVA BDT (+ ROOT output)
#   Stage 3  plot_bdt_output.py         TMVA output -> PNG demonstration plots
#
# DEMO SAMPLE (small subset, mc23a only)
#   Signal      : H->ZdZd->4l at mZd = 20, 30, 50 GeV  (DSIDs 561509/561511/561515)
#   Background  : SM H->ZZ*->4l   VBFH (601500) + ggH NNLOPS (604263)
#
# Each signal/background process is assumed to be a single merged Ntuple file.
# Signal filenames carry a unique job ID, so they are resolved by a DSID glob.
#
# ENVIRONMENT
#   This script sets up NOTHING.  Run `setupATLAS` (and whatever it brings in)
#   BEFORE invoking it.  Needs python3 with uproot/awkward/numpy/pandas/pyarrow
#   and matplotlib, plus PyROOT with TMVA for Stage 2.
#
# Usage:
#   setupATLAS
#   bash run_bdt_demo_lxplus.sh
# Optional overrides (environment variables):
#   PROJECT_DIR=/path/to/run3_ZdZd_project
#
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TMVA_DIR="${PROJECT_DIR}/tmva_ZdZd_R3"
NTUPLE_OUT="${PROJECT_DIR}/data/training_ntuples/demo_sig3_bkgHZZ.parquet"

# eos Ntuple areas
EOS_BASE="/eos/atlas/atlascerngroupdisk/phys-hmbs/hlrs/ZdZd/ZdZd13TeV_Ntuples"
SIG_DIR="${EOS_BASE}/signal_Ntuples/mc23a_p6697_noSyst"
BKG_DIR="${EOS_BASE}/bkg_Ntuples/H_ZZ_4l"

# Signal DSIDs to use (mc23a).  561509=mZd20, 561511=mZd30, 561515=mZd50.
SIGNAL_DSIDS=(561509 561511 561515)

# Background files (exact merged paths from the production CSVs, mc23a).
BACKGROUND_FILES=(
    "${BKG_DIR}/601500.PhPy8EG_PDF4LHC21_VBFH125_HZZ_4l_notau.mc23a.p7266.v1.root"
    "${BKG_DIR}/604263.PhPy8EG_Hto4l_NNLOPS_nnlo_30_ggH125_ZZ4l.mc23a.p7266.v1.root"
)


# ---------------------------------------------------------------------------
# 1. Pre-flight (no setup is performed here - run setupATLAS first)
# ---------------------------------------------------------------------------
echo "=== Pre-flight ==="
echo "  python : $(command -v python3 || echo 'NOT FOUND')"
echo "  root   : $(command -v root || echo 'not found (Stage 2 uses PyROOT)')"
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: no python3 on PATH. Run setupATLAS first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Resolve signal files (one file per DSID, via glob)
# ---------------------------------------------------------------------------
echo
echo "=== Resolving signal Ntuples in ${SIG_DIR} ==="
shopt -s nullglob
SIGNAL_FILES=()
for dsid in "${SIGNAL_DSIDS[@]}"; do
    matches=( "${SIG_DIR}/user.connell.${dsid}."*".my.output.root" )
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERROR: no signal file found for DSID ${dsid} in ${SIG_DIR}" >&2
        echo "       (expected user.connell.${dsid}.*.my.output.root)" >&2
        exit 1
    elif [[ ${#matches[@]} -gt 1 ]]; then
        echo "WARNING: ${#matches[@]} files match DSID ${dsid}; using the first." >&2
        printf '         %s\n' "${matches[@]}" >&2
    fi
    SIGNAL_FILES+=( "${matches[0]}" )
    echo "  DSID ${dsid} -> ${matches[0]}"
done
shopt -u nullglob

echo
echo "=== Background Ntuples ==="
for f in "${BACKGROUND_FILES[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: background file not found: ${f}" >&2
        exit 1
    fi
    echo "  ${f}"
done

# ---------------------------------------------------------------------------
# 3. Stage 1 — build the training Parquet
# ---------------------------------------------------------------------------
echo
echo "=== Stage 1: make_training_ntuples.py ==="
mkdir -p "$(dirname "${NTUPLE_OUT}")"
python3 "${TMVA_DIR}/make_training_ntuples.py" \
    --signal     "${SIGNAL_FILES[@]}" \
    --background "${BACKGROUND_FILES[@]}" \
    --output     "${NTUPLE_OUT}"

# ---------------------------------------------------------------------------
# 4. Stage 2 — train the BDT
# ---------------------------------------------------------------------------
echo
echo "=== Stage 2: train_bdt.py ==="
python3 "${TMVA_DIR}/train_bdt.py" \
    --input      "${NTUPLE_OUT}" \
    --output-dir "${TMVA_DIR}"

# ---------------------------------------------------------------------------
# 5. Stage 3 — demonstration plots
# ---------------------------------------------------------------------------
echo
echo "=== Stage 3: plot_bdt_output.py ==="
python3 "${TMVA_DIR}/plot_bdt_output.py" \
    --output-dir "${TMVA_DIR}" \
    --top-n 6

echo
echo "=== Done. Plots written to ${TMVA_DIR}/plots/ ==="
ls -1 "${TMVA_DIR}/plots/" 2>/dev/null || true
