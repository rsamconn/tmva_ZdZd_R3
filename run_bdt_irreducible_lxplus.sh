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
# ENVIRONMENT
#   This script sets up NOTHING.  Run `setupATLAS` (and whatever it brings in)
#   BEFORE invoking it; the script only checks that what it needs is on the
#   PATH and importable, and reports clearly if it is not.  Required: python3
#   with uproot, awkward, numpy, pandas, pyarrow and matplotlib for Stages 0/1/3,
#   and PyROOT with TMVA for Stage 2.
#
# SAMPLES (mc23a)
#   Signal     : H->ZdZd->4l at mZd = 20, 30, 50 GeV (DSIDs 561509/561511/561515)
#   Background : the four irreducible processes - H_ZZ_4l, ZZ_4l, Tribosons,
#                ttbarZ - taken from normalisation_inputs/cutflow_inputs.csv,
#                with 701185 and 701190 excluded on m4l-overlap grounds with the
#                inclusive 701040 Sh_llll sample.
#
# The background sample list is NOT hardcoded here: it is derived from the
# production registry by Stage 0.  Only the signal DSIDs and the per-process
# event cap live in this file.
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
#   PROJECT_DIR=...      root of run3_ZdZd_project   (default: parent of this script)
#   INPUT_DIR=...        normalisation CSVs          (default: <repo>/normalisation_inputs)
#   CAMPAIGN=mc23a       MC campaign
#   TARGET_PER_PROCESS=  selected events per background process (default 150000)
#   CHUNK_STRIDE=        spread the sample across each merged file (default 4)
#   WEIGHTING=           normalised | equal-process | raw  (default normalised)
#   SKIP_ENV_CHECK=1     skip the dependency check below
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
TMVA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${TMVA_DIR}/.." && pwd)}"
INPUT_DIR="${INPUT_DIR:-${TMVA_DIR}/normalisation_inputs}"
CAMPAIGN="${CAMPAIGN:-mc23a}"
TARGET_PER_PROCESS="${TARGET_PER_PROCESS:-150000}"
CHUNK_STRIDE="${CHUNK_STRIDE:-4}"
WEIGHTING="${WEIGHTING:-normalised}"

MANIFEST="${PROJECT_DIR}/data/training_ntuples/manifest_${CAMPAIGN}.csv"
NTUPLE_OUT="${PROJECT_DIR}/data/training_ntuples/sig3_irreducible_${CAMPAIGN}.parquet"

# eos Ntuple areas
EOS_BASE="/eos/atlas/atlascerngroupdisk/phys-hmbs/hlrs/ZdZd/ZdZd13TeV_Ntuples"
SIG_DIR="${SIG_DIR:-${EOS_BASE}/signal_Ntuples/${CAMPAIGN}_p6697_noSyst}"

# Signal DSIDs.  561509=mZd20, 561511=mZd30, 561515=mZd50.
SIGNAL_DSIDS="${SIGNAL_DSIDS:-561509 561511 561515}"

# ---------------------------------------------------------------------------
# 1. Pre-flight checks (no setup is performed here)
# ---------------------------------------------------------------------------
echo "=== Pre-flight ==="
echo "  repo        : ${TMVA_DIR}"
echo "  project dir : ${PROJECT_DIR}"
echo "  inputs      : ${INPUT_DIR}"
echo "  python3     : $(command -v python3 || echo 'NOT FOUND')"
echo "  root        : $(command -v root || echo 'not found (Stage 2 uses PyROOT)')"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: no python3 on PATH. Run setupATLAS first." >&2
    exit 1
fi

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

# ---------------------------------------------------------------------------
# 2. Stage 0 - build the sample manifest
# ---------------------------------------------------------------------------
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
