# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is for developing a TMVA (Toolkit for Multivariate Analysis) for background reduction in the Run 3 H→ZdZd→4l (dark Z boson) analysis at CERN/LHC. It is part of a larger ZdZd analysis framework.

## Jargon & Abbreviations

| Object | Abbreviation | Description |
|--------|---------|-------------|
| Quadruplet | `llll` | Group of 4 leptons, which may reconstruct to a Higgs boson |
| Dilepton | `ll` | Group of two leptons, which may reconstruct to a Zd |
| Dark vector boson | `Zd` | Dark counterpart of the SM Z boson. |
| Standard Model | `SM` | Standard Model of particle physics |
| Transverse momentum | `pT` | Momentum of an object measured in the transverse plane, perpendicular to the beam axis |
| Missing transverse energy | `MET` | Transverse energy needed to balance out the transverse energies of detected particles |
| Primary Vertex | `PV` | The reconstructed proton-proton hard-scatter interaction point |
| Low Mass | `LM` | Analysis region targeting Zd masses below 15 GeV |
| High Mass | `HM` | Analysis region targeting Zd masses above 15 GeV |
| Signal Region | `SR` | Kinematic phase-space region enriched in signal events, used for the final measurement |
| Scale Factor | `SF` | Per-object correction applied to simulation to match data efficiency or yield |

## Context

- **Analysis**: Search for exotic Higgs decays H→ZdZd→4l (four leptons via dark Z bosons), Run 3 LHC data
    - 'Leptons' here refers only to electrons and muons, not taus.
    - This analysis is based on previous analyses from Run 1 and Run 2 which investigated the same decay, and some jargon (e.g. High Mass/ Low Mass), is carried over from these analyses.
- **Goal**: Train and evaluate multivariate discriminants (BDT, neural networks, etc.) using ROOT TMVA to improve signal-to-background separation
- **Related repos**: Lives under `run3_ZdZd_project/` alongside other analysis components
