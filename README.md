# tmva_ZdZd_R3
Developing a TMVA for background reduction in the Run 3 H->ZdZd->4l analysis, with the help of Claude Cowork.

This repository is stored online here: [github](https://github.com/rsamconn/tmva_ZdZd_R3)

## Updating CLAUDE.md
The repo is currently just initialized with a README — as you add training scripts, config files, and workflows, the CLAUDE.md should be updated with:                                                                                        
- How to run TMVA training (e.g., root -l train.C or Python equivalent)
- Input ntuple locations / data format
- Key variables/features used in the MVA
- How to evaluate/apply the trained discriminant

---

## 1. Project Overview

The goal of this project is to train a Boosted Decision Tree (BDT) classifier
using the ROOT TMVA framework to separate signal events from background events
in a high-energy physics analysis. The BDT is intended as a multivariate
alternative to an existing cut-based analysis cutflow (e.g. cuts such as
requiring electron pT > 5 GeV). Replacing hard cuts with a single BDT score
should improve signal-to-background discrimination.

- **ML framework:** ROOT TMVA (Toolkit for Multivariate Analysis)
- **Method:** Boosted Decision Tree (BDT)
- **Interface:** Python / PyROOT
- **Primary language:** Python 3

---

## 2. Physics & Analysis Context

<!-- TO BE FILLED IN -->

- **Analysis name:** _Search for Higgs boson decays to four leptons via dark vector bosons_
- **Signal process:** _e.g. H → ZdZd → ℓℓℓℓ_
- **Background process(es):** _e.g. ZZ*, Zjets, ttbar_
- **Existing cutflow (implemented in `ZdZdPlottingAlg.cxx`):**
  A sequential set of selection cuts is used to separate signal from background.
  Key cuts, in order:
  - Event preselection: `passCleaning`, `passNPV`, trigger bitmask
  - Quadruplet SFOS: `charge == 0` and `dCharge == 0`
  - Overlap removal: `overlaps & 54 == 0` (removes lepton–lepton overlap-removed quads)
  - Kinematic pT: l1 ≥ 20 GeV, l2 ≥ 15 GeV, l3 ≥ 10 GeV (pT-ordered)
  - Trigger match: `triggerMatched != 0`
  - ΔR: `min_sf_dR > 0.1`, `min_of_dR > 0.2`
  - Muon type: `nCTorSA < 2`
  - Isolation: `l_isIsolCloseBy == 15` (all 4 leptons pass FixedCutLoose)
  - Electron quality: all leptons with `quality < 3`
  - Impact parameter: `max_el_d0Sig < 5`, `max_mu_d0Sig < 3`
  - Quarkonia vetoes: J/ψ window (|m − 3416| < 570 MeV), Υ window (|m − 9933| < 1172 MeV),
    applied to all four dilepton pairings (m12, m34, m32, m14)
  - H window: 115 GeV < m_4l < 130 GeV
  - Z veto: m12 < 64 GeV, m34 < 64 GeV, **m32 < 75 GeV, m14 < 75 GeV**
    (primary pairings at 64 GeV, alternative pairings at 75 GeV)
  - Loose SR: all four dilepton masses > 10 GeV
  - Medium SR: m34/m12 ≥ 0.85 − 0.1125 × f(m12), where f is a sigmoid-like
    modulation function centred at m12 ≈ 50 GeV (implemented in `SRModulation.cxx`)
- **Analysis framework:**
    - The custom AnalysisCam framework [gitlab](https://gitlab.cern.ch/atlas-phys/exot/ueh/EXOT-2016-22/AnalysisCam) is used for constructing multi-lepton objects.
    - Signal, background and ATLAS data are supplied as `.root` files in DAOD_PHYS format.
    - These are passed into the **ZdZd13TeV algorithm** (`ZdZdAnalysisAlg.cxx` + `ZdZdHelper.h`),
      which identifies events with viable lepton quadruplets, builds dilepton and quadruplet
      objects (pT-ordering leptons l1–l4, Z-distance ordering dileptons ll1/ll2), computes
      derived quantities (min_sf_dR, min_of_dR, d0Sig, scaleFactor, etc.), and writes output
      ntuples to `Nominal/llllTree` (plus ~56 systematic variation trees).
    - These output ntuples are passed through the **ZdZdPostProcessing algorithm**
      (`ZdZdPlottingAlg.cxx`), which applies the full cutflow (see above) and fills
      `avgMll` histograms for the final statistical analysis.
    - The TMVA algorithm sits between event preselection (cleaning + trigger + SFOS + kinematic
      pT) and the isolation/impact-parameter/quarkonia/ZVeto/LooseSR/MediumSR cuts. The goal
      is for a single BDT score to replace those later cuts with better signal efficiency.

---

## 3. Team & Users

<!-- TO BE FILLED IN -->

- **Developer (code author):** _Matt Connell _
- **Primary user(s):** _BNL-SA-Morocco and Bucharest-Yale groups_
- **Experiment:** _ATLAS_
- **Notes on expertise level:** _PhD level student, comfortable in Python and C++ with no formal training in software engineering and intermediate knowledge of machine learning in scikit-learn._

---

## 4. Data

<!-- TO BE FILLED IN -->

- **File format:** ROOT `.root` files (TTree-based), produced by the analysis code
- **Signal sample(s):** _File path(s), MC generator, process, number of events_
- **Background sample(s):** _File path(s), MC generator, process(es), number of events_
- **TTree name:** _To be confirmed by running the data exploration script_
- **Branch structure:** Possibly non-flat (vector branches / object collections
  such as dilepton pairs). **Must be confirmed before writing the training script.**
- **Signal vs. background labelling:** _Are signal and background in separate files,
  or in the same file with a label branch? To be confirmed._
- **Event weights:** _Does the TTree contain event weights (e.g. MC weight,
  luminosity weight)? If so, which branch?_
- **Intended input variables (BDT features):** _To be listed once data
  exploration is complete — e.g. lepton pT, MET, dilepton mass, …_

---

## 5. Environments
Local machine is used for small-scale testing only; the main workflow will take place via grid-submission on lxplus.

### 5a. Local machine (testing)
- **OS:** _macOS 26.3.1_
- **ROOT version:** 6.26.06
- **Python version:** _Python 3.11.3_

### 5b. CERN lxplus (primary development & testing)
- **Access:** `ssh <username>@lxplus.cern.ch`
- **ROOT:** 6.38.04
- **Python:** Python 3.9.25
- **Working directory:** _To be filled in (e.g. `/afs/cern.ch/user/...`)_

### 5c. Grid submission (large-scale jobs)
- **Grid system:** CERN/WLCG grid
- **Submission tool:** _e.g. HTCondor, Rucio — to be filled in_
- **Notes:** Large training jobs and application to full datasets will be
  submitted to the grid.

### 5d. Claude tools
- **Claude Cowork:** Used for writing, designing, and iterating on code;
  project planning; document management.
- **Claude Code:** Installed in local environment for running scripts, reading output, and interactive debugging.

---

## 6. Code Structure

<!-- TO BE FILLED IN once scripts exist -->

```
tmva_ZdZd_R3/
├── README.md                  ← this file
├── CLAUDE.md                  ← instructions for directing Claude when coding
├── explore_data.py            ← inspect ROOT file structure (step 1)
├── train_bdt.py               ← TMVA BDT training script (step 2)
├── evaluate_bdt.py            ← ROC curves, feature importance (step 3)
├── apply_bdt.py               ← apply trained weights to new data (step 4)
├── weights/                   ← TMVA output weights directory
│   └── TMVAClassification_BDT.weights.xml
└── plots/                     ← output figures
```

---

## 7. Development Workflow

The intended workflow is:

1. **Explore data** — run `explore_data.py` on lxplus to confirm TTree name,
   branch names, and data types.
2. **Define features** — decide which branches to use as BDT input variables
   based on the exploration output.
3. **Train** — run `train_bdt.py` on lxplus (or locally for small samples).
4. **Evaluate** — inspect ROC curves, overtraining checks, and feature
   importance produced by TMVA.
5. **Iterate** — tune BDT hyperparameters or add/remove features as needed.
6. **Apply** — run `apply_bdt.py` to score events in new data files.
7. **Grid submission** — scale up to full dataset via grid once the pipeline
   is validated on a small sample.

---

## 8. TMVA Configuration

<!-- TO BE FILLED IN as decisions are made -->

- **TMVA method:** `BDT`
- **Key hyperparameters (initial defaults):**
  - `NTrees`: _e.g. 850_
  - `MaxDepth`: _e.g. 3_
  - `AdaBoostBeta`: _e.g. 0.5_
  - `nCuts`: _e.g. 20_
  - _(to be tuned after initial training)_
- **Train/test split:** _e.g. 70% training / 30% test_
- **Input variable transformations:** _e.g. normalisation — to be decided_
- **Output weights file:** `weights/TMVAClassification_BDT.weights.xml`

---

## 9. Performance Targets & Validation

<!-- TO BE FILLED IN -->

- **Baseline to beat:** The existing cut-based analysis cutflow
- **Primary figure of merit:** _e.g. signal significance (S/√B), ROC AUC,
  or analysis-specific metric_
- **Overtraining check:** Kolmogorov–Smirnov test (provided by TMVA);
  train/test response distributions must agree
- **Other checks:** _e.g. agreement with cutflow on validation sample_

---

## 10. Assumptions and Confirmed Facts

1. ROOT 6.26.06 on the local machine is compatible with the PyROOT TMVA
   interface used in this project (TMVA 4.3.0, requires ROOT ≥ 6.12).
2. lxplus has ROOT available via CVMFS and scripts will be run with `python3`
   after sourcing the appropriate LCG release.
3. **CONFIRMED**: The output TTree is `Nominal/llllTree` inside the NTUP4L ROOT
   file. Quadruplet quantities are stored as `vector<>` branches (one entry per
   quadruplet candidate per event); lepton/dilepton quantities are also vectors
   indexed by l_index / ll_index. Stage 1 flattens these to one row per event
   (selecting the first passing quadruplet).
4. **CONFIRMED** (from `ZdZdHelper.h`): `l1/l2/l3/l4` indices in the quadruplet
   are pT-descending (set via `sort_Pt()`). The kinematic pT cuts in Stage 1 are
   applied directly on these indices without re-sorting, consistent with
   `ZdZdPlottingAlg.cxx`.
5. **CORRECTED** (from `ZdZdHelper.h`): `max_el_d0Sig` and `max_mu_d0Sig` store
   **0.0** (not −999) for quadruplet flavours with no electrons/muons respectively.
   This is computed as `std::max({pdgId!=11 ? 0.0 : |d0Sig|, ...})`. The 0.0
   value is physically meaningful and does not require special imputation before
   BDT training.

---

## 11. Ntuple Conversion to Training Data

The script `make_training_ntuples.py` (Stage 1 of the pipeline) converts
ZdZd13TeV ROOT ntuples into a flat Parquet file ready for BDT training.
The following decisions were made when designing this script.

### Input data source
- **Use the ZdZd13TeV `Nominal/llllTree` only.** The ZdZdPostProcessing output
  contains only final `avgMll` histograms after the full cutflow and carries no
  per-event information; it cannot be used for BDT training.
- **Use the Nominal tree only**, not the systematic variation trees
  (e.g. `EG_RESOLUTION_ALL1up/llllTree`). Systematics are applied at the
  evaluation/statistical analysis stage.

### Event preselection
All three of the following must pass (matching `ZdZdPP_alg.cxx`):
- `passCleaning == True`
- `passNPV == True`
- `passTriggers != 0`

### Quadruplet selection
For each event the first quadruplet candidate (in the order stored in the tree)
passing **both** of the following hard cuts is selected:

1. **SFOS**: `llll_charge == 0` AND `llll_dCharge == 0`
2. **Kinematic pT** (thresholds from `ZdZdPP_alg.cxx`, in MeV):
   - `pT(l1) >= 20 000` (leading lepton)
   - `pT(l2) >= 15 000` (subleading)
   - `pT(l3) >= 10 000` (subsubleading)
   - `l1–l4` are stored in pT-descending order within each quadruplet candidate.

All remaining selection quantities (isolation, dR, impact parameter, trigger
match, muon type, …) are **not applied as hard cuts** — they are retained as
BDT input features so the classifier can learn from them.

### Feature vector
All mass and momentum quantities are stored in **MeV** throughout, consistent
with the ZdZd13TeV tree convention.

| Column | Source | Notes |
|---|---|---|
| `m_4l` | `llll_tlv_m` | Four-lepton invariant mass |
| `avgM` | `llll_avgM` | (mab + mcd) / 2 |
| `dM` | `llll_dM` | mab − mcd |
| `mab` | `ll_tlv_m[llll_ll1 or ll2]` | Larger of the two primary dilepton masses |
| `mcd` | `ll_tlv_m[llll_ll1 or ll2]` | Smaller of the two primary dilepton masses |
| `mad` | `ll_tlv_m[llll_alt_ll1]` | Alt. pairing leading dilepton mass |
| `mbc` | `ll_tlv_m[llll_alt_ll2]` | Alt. pairing subleading dilepton mass |
| `mcd_over_mab` | derived | mcd / mab; MediumSR discriminant |
| `min_sf_dR` | `llll_min_sf_dR` | Min ΔR same-flavour lepton pair |
| `min_of_dR` | `llll_min_of_dR` | Min ΔR opp.-flavour pair (**sentinel** 9999999 for 4e/4mu) |
| `vtx_reduced_chi2` | `llll_vtx_reduced_chi2` | Quadruplet vertex χ² (**sentinel** −999) |
| `max_el_d0Sig` | `llll_max_el_d0Sig` | Max electron \|d0/σ\| (**sentinel** 0.0 for 4mu — no electrons present) |
| `max_mu_d0Sig` | `llll_max_mu_d0Sig` | Max muon \|d0/σ\| (**sentinel** 0.0 for 4e — no muons present) |
| `nCTorSA` | `llll_nCTorSA` | Number of CT or SA muons |
| `l_isIsolCloseBy` | `llll_l_isIsolCloseBy` | Isolation bitmask (15 = all isolated) |
| `triggerMatched` | `llll_triggerMatched` | Trigger-match bitmask |
| `is_4e`, `is_2e2mu`, `is_4mu` | one-hot from `llll_pdgIdSum` | Quadruplet flavour |
| `pT_l1`…`pT_l4` | `l_tlv_pt` via `llll_l1`…`l4` | Lepton pTs, pT-ordered |
| `eta_l1`…`eta_l4` | `l_tlv_eta` via `llll_l1`…`l4` | Lepton η, same ordering |
| `mu` | `averageInteractionsPerCrossing` | Pile-up |

Columns marked **sentinel** contain placeholder values for certain quadruplet
flavours and must be handled (excluded or imputed) in the training script
before being passed to TMVA.

**Sentinel value clarifications (confirmed from `ZdZdHelper.h`):**
- `min_of_dR`: stored as 9999999 when no opposite-flavour pairs exist (4e or 4μ events)
- `vtx_reduced_chi2`: stored as −999 when the vertex fit fails (~1% of events)
- `max_el_d0Sig`: stored as **0.0** (not −999) for 4μ events — computed via
  `std::max({pdgId!=11 ? 0.0 : |d0Sig|, ...})`, so the absence of electrons gives 0
- `max_mu_d0Sig`: stored as **0.0** (not −999) for 4e events — same logic

The 0.0 sentinel for d0Sig is physically interpretable (zero impact parameter
significance) and does not create a discontinuous spike in the BDT training
distributions, unlike −999 would have.

### Handling multiple mZd signal samples (option a)
A single mZd-inclusive BDT is trained using all signal samples combined.
`truth_zdzd_avgM` (MeV) is included in the output as a feature column so the
BDT can learn mass dependence. For background samples this column defaults to 0.

### Signal / background labelling
- `label = 1`: H→ZdZd→4ℓ signal MC (all mZd values, all MC campaigns)
- `label = 0`: background MC (ZZ\*→4ℓ and other relevant backgrounds)
- Signal and background files are processed separately and combined in the
  output Parquet. The training script feeds them to TMVA's separate signal and
  background TTrees.

### H window cut
The H window cut (115–130 GeV on m_4l) is **not applied** during conversion,
giving the BDT access to sideband events and avoiding bias in the m_4l feature.

### Output format
Stage 1 outputs a **Parquet file** (one per run, containing all input samples).
Stage 2 (training script) reads the Parquet, constructs flat ROOT TTrees for
TMVA, and runs training. This separation means data preparation can be run and
inspected without ROOT.

---

## 13. To Do / Open Questions

The following questions are open and must be answered before the corresponding
development steps can proceed. They will be addressed one by one.

### Data (required before writing training script)
- [x] What is the TTree name? → `Nominal/llllTree` inside the NTUP4L ROOT file
- [x] Are branches scalar or vectors? → Vectors (`std::vector<>` per candidate); Stage 1
      flattens to one row per event by selecting the first passing quadruplet.
- [x] Signal and background in separate files? → Separate files; Stage 1 labels them
      `label=1` (signal) and `label=0` (background) and combines in one Parquet.
- [x] Event weights? → `evtWeight` (generator), `PileupWeight`, `llll_scaleFactor`
      (per-quadruplet); combined as `evtWeight_total = evtWeight × PileupWeight × scaleFactor`.
- [x] BDT input features? → Defined in Section 11 feature table (24 variables).

### Environment (required before running on lxplus)
- [ ] What is the exact LCG/CVMFS setup command used on lxplus for this analysis?
- [ ] What is the working directory / AFS/EOS path for storing scripts and outputs?
- [ ] What grid submission system and tools will be used for large jobs?

### Physics & Analysis
- [x] Full cutflow list → Documented in Section 2 (confirmed from `ZdZdPlottingAlg.cxx`).
- [ ] Primary figure of merit for BDT vs cutflow comparison (e.g. S/√B, expected CLs)?

### Team
- [ ] Who will ultimately use and maintain this code?
- [ ] Are there other analysts or collaborators whose input is needed on
      variable selection or performance targets?
