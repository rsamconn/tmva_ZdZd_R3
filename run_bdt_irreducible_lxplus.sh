#!/usr/bin/env bash
#
# run_bdt_irreducible_lxplus.sh
# =============================
# Train the tmva_ZdZd_R3 BDT on signal plus a small sample of EACH irreducible
# background process, using the merged Run 3 Ntuples on eos.  Runs four stages:
#
#   Stage 0  make_sample_manifest.py     production registry -> manifest + scale_d
#   Stage 1  make_training_ntuples.py    ROOT (eos)          -> Parquet
#   Stage 2  train_bdt.py                Parquet             -> TMVA BDT
#   Stage 3  plot_bdt_output.py          TMVA output         -> PNG plots
#
# SAMPLES (mc23a)
#   Signal     : H->ZdZd->4l at mZd = 20, 30, 50 GeV (DSIDs 561509/561511/561515)
#   Background : the four irreducible processes - H_ZZ_4l, ZZ_4l, Tribosons,
#                ttbarZ - taken from cutflow_inputs.csv, with 701185 and 701190
#                excluded on m4l-overlap grounds with the inclusive 701040
#                Sh_llll sample.
#
# The background sample list is NOT hardcoded here: it is derived from the
# production registry by Stage 0, so it tracks the registry rather than drifting
# from it.  Only the signal DSIDs and the per-process event cap live in this file.
#
# NORMALISATION
#   Stage 0 computes scale_d = L * sigma * k * eps_filt / SumW_total per sample
#   and Stage 2 applies it, so the background processes enter in proportion to
#   their expected yield rather than to their MC statistics.  603293 mc23a has no
#   measured SumW_total and uses the accepted estimate (~18767); the manifest
#   flags it and Stage 0 prints a warning.
#
# Usage:
#   bash run_bdt_irreducible_lxplus.sh
# Optional overrides (environment variables):
#   PROJECT_DIR=...      root of run3_ZdZd_project   (default: parent of this script)
#   CUTFLOW_DIR=...      ZdZdPostProcessing/cutflow_automation
#   LCG_VIEW=...         LCG view to source
#   CAMPAIGN=mc23a       MC campaign
#   TARGET_PER_PROCESS=  selected events per background process (default 150000)
#   CHUNK_STRIDE=        spread the sample across each merged file (default 4)
#   WEIGHTING=           normalised | equal-process | raw  (default normalised)
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TMVA_DIR="${PROJECT_DIR}/tmva_ZdZd_R3"
CAMPAIGN="${CAMPAIGN:-mc23a}"
TARGET_PER_PROCESS="${TARGET_PER_PROCESS:-150000}"
CHUNK_STRIDE="${CHUNK_STRIDE:-4}"
WEIGHTING="${WEIGHTING:-normalised}"

CUTFLOW_DIR="${CUTFLOW_DIR:-${PROJECT_DIR}/background/current_code/ZdZdPostProcessing/cutflow_automation}"

MANIFEST="${PROJECT_DIR}/data/training_ntuples/manifest_${CAMPAIGN}.csv"
NTUPLE_OUT="${PROJECT_DIR}/data/training_ntuples/sig3_irreducible_${CAMPAIGN}.parquet"

# eos Ntuple areas
EOS_BASE="/eos/atlas/atlascerngroupdisk/phys-hmbs/hlrs/ZdZd/ZdZd13TeV_Ntuples"
SIG_DIR="${SIG_DIR:-${EOS_BASE}/signal_Ntuples/${CAMPAIGN}_p6697_noSyst}"

# Signal DSIDs.  561509=mZd20, 561511=mZd30, 561515=mZd50.
SIGNAL_DSIDS="${SIGNAL_DSIDS:-561509 561511 561515}"

# LCG software view (provides ROOT+PyROOT+TMVA, uproot, awkward, matplotlib).
LCG_VIEW="${LCG_VIEW:-LCG_105/x86_64-el9-gcc13-opt}"

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
echo "=== Setting up environment: ${LCG_VIEW} ==="
LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/${LCG_VIEW}/setup.sh"
if [[ ! -f "${LCG_SETUP}" ]]; then
    echo "ERROR: LCG view setup not found: ${LCG_SETUP}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${LCG_SETUP}"

PYTHON_BIN="$(command -v python3 || true)"
ROOT_BIN="$(command -v root || true)"
echo "  python : ${PYTHON_BIN:-not found}"
echo "  root   : ${ROOT_BIN:-not found}"

# The 2026-08-03 run sourced this view and still got /usr/bin/python3, which is
# the interpreter that segfaulted at TMVA teardown.  Fail loudly rather than
# silently training under the system Python.
if [[ "${PYTHON_BIN}" != /cvmfs/* ]]; then
    echo "ERROR: python3 resolves to '${PYTHON_BIN}', not the LCG view." >&2
    echo "       The view did not take effect. Check that \$LCG_VIEW exists and" >&2
    echo "       that no earlier setup pinned PATH, or set ALLOW_SYSTEM_PYTHON=1" >&2
    echo "       to proceed anyway." >&2
    [[ "${ALLOW_SYSTEM_PYTHON:-0}" == "1" ]] || exit 1
    echo "       ALLOW_SYSTEM_PYTHON=1 set; continuing with ${PYTHON_BIN}." >&2
fi

for f in cutflow_inputs.csv crossSections_run3.csv sumw_total_p7266.csv; do
    if [[ ! -f "${CUTFLOW_DIR}/${f}" ]]; then
        echo "ERROR: ${f} not found under ${CUTFLOW_DIR}" >&2
        echo "       Set CUTFLOW_DIR to ZdZdPostProcessing/cutflow_automation." >&2
        exit 1
    fi
done

if [[ ! -d "${SIG_DIR}" ]]; then
    echo "ERROR: signal Ntuple directory not found: ${SIG_DIR}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Stage 0 - build the sample manifest
# ---------------------------------------------------------------------------
echo
echo "=== Stage 0: make_sample_manifest.py ==="
mkdir -p "$(dirname "${MANIFEST}")"
python3 "${TMVA_DIR}/make_sample_manifest.py" \
    --cutflow-inputs "${CUTFLOW_DIR}/cutflow_inputs.csv" \
    --cross-sections "${CUTFLOW_DIR}/crossSections_run3.csv" \
    --sumw           "${CUTFLOW_DIR}/sumw_total_p7266.csv" \
    --campaign       "${CAMPAIGN}" \
    --signal-dir     "${SIG_DIR}" \
    --signal-dsids   ${SIGNAL_DSIDS} \
    --output         "${MANIFEST}"

# ---------------------------------------------------------------------------
# 3. Stage 1 - build the training Parquet
# ---------------------------------------------------------------------------
echo
echo "=== Stage 1: make_training_ntuples.py ==="
echo "  cap per background process : ${TARGET_PER_PROCESS}"
echo "  chunk stride               : ${CHUNK_STRIDE}"
python3 "${TMVA_DIR}/make_training_ntuples.py" \
    --manifest "${MANIFEST}" \
    --target-selected-per-process "${TARGET_PER_PROCESS}" \
    --chunk-stride "${CHUNK_STRIDE}" \
    --output   "${NTUPLE_OUT}"

# ---------------------------------------------------------------------------
# 4. Stage 2 - train the BDT
# ---------------------------------------------------------------------------
echo
echo "=== Stage 2: train_bdt.py ==="
python3 "${TMVA_DIR}/train_bdt.py" \
    --input      "${NTUPLE_OUT}" \
    --output-dir "${TMVA_DIR}" \
    --weighting  "${WEIGHTING}"

# ---------------------------------------------------------------------------
# 5. Stage 3 - plots
# ---------------------------------------------------------------------------
echo
echo "=== Stage 3: plot_bdt_output.py ==="
python3 "${TMVA_DIR}/plot_bdt_output.py" \
    --output-dir "${TMVA_DIR}" \
    --top-n 6

echo
echo "=== Done. Plots written to ${TMVA_DIR}/plots/ ==="
