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
- **Existing cutflow:** A sequential set of selection cuts is currently used to
  separate signal from background. Key cuts include (but are not limited to):
  - Electron transverse momentum (pT) > 5 GeV
  - _(additional cuts to be listed here)_
- **Analysis framework:**
    - The custom AnalysisCam framework [gitlab](https://gitlab.cern.ch/atlas-phys/exot/ueh/EXOT-2016-22/AnalysisCam) is used for constructing multi-lepton objects.
    - Signal, background and ATLAS data are supplied as `.root` files in DAOD_PHYS format.
    - These are passed into the ZdZd13TeV algorithm [gitlab](https://gitlab.cern.ch/atlas-phys/exot/ueh/EXOT-2016-22/ZdZd13TeV/-/tree/r25_run3?ref_type=heads), which identifies events with viable lepton quadruplets and saves relevant information from them, as well as building candidate dilepton and quadruplet objects, into output Ntuples.
    - These output Ntuples are passed through the much lighter ZdZdPostProcessing algorithm [gitlab](https://gitlab.cern.ch/as_followups/ZdZdPostProcessing/-/tree/r25_run3_ZdZd?ref_type=heads), which performs a cutflow analysis to identify events passing a given signal region.
    - The TMVA algorithm sits somewhere between the beginning and end of the cutflow: after some cuts have been passed, such as cleaning and trigger cuts, but before too many events have been disqualified. The goal of the TMVA algorithm is to be an improved alternative to most of the remaining cutflow.

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

## 10. Assumptions

The following assumptions have been made and should be confirmed or corrected
as the project progresses:

1. ROOT 6.26.06 on the local machine is compatible with the PyROOT TMVA
   interface used in this project (TMVA 4.3.0, requires ROOT ≥ 6.12).
2. lxplus has ROOT available via CVMFS and scripts will be run with `python3`
   after sourcing the appropriate LCG release.
3. The analysis code output contains dilepton objects and other combined
   quantities stored as **scalar** (per-event) branches in the TTree — this
   must be confirmed by the data exploration step.

---

## 11. To Do / Open Questions

The following questions are open and must be answered before the corresponding
development steps can proceed. They will be addressed one by one.

### Data (required before writing training script)
- [ ] What is the TTree name in the analysis output `.root` file?
- [ ] Are branches scalar (flat) or do they include vectors / object
      collections (e.g. `std::vector<float>` per lepton)?
- [ ] Are signal and background stored in **separate `.root` files**, or in
      the **same file** distinguished by a label branch?
- [ ] Does the TTree contain event weights? If so, which branch name(s)?
- [ ] Which branches/variables should be used as BDT input features (e.g.
      lepton pT, MET, dilepton mass, angular quantities)?

### Environment (required before running on lxplus)
- [ ] What is the exact LCG/CVMFS setup command used on lxplus for this analysis?
- [ ] What is the working directory / AFS/EOS path for storing scripts and outputs?
- [ ] What grid submission system and tools will be used for large jobs?

### Physics & Analysis
- [ ] What is the full list of cuts in the existing cutflow (to serve as a
      baseline for comparison)?
- [ ] What is the primary figure of merit for comparing the BDT to the cutflow
      (e.g. S/√B, expected CLs limits)?

### Team
- [ ] Who will ultimately use and maintain this code?
- [ ] Are there other analysts or collaborators whose input is needed on
      variable selection or performance targets?
