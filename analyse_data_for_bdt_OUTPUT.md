(root_env) matt@MConn-MacBook-Pro-2 tmva_ZdZd_R3 % python3 analyse_data_for_bdt.py

======================================================================
  ZdZd Run-3 BDT Data Format Analysis
======================================================================

Examining data in:
  ZdZd13TeV     [mc23a_mZd30]  ✓
  ZdZd13TeV     [mc23a_mZd60]  ✓
  PostProcessing [mc23a_mZd30]  ✓
  PostProcessing [mc23a_mZd60]  ✓

======================================================================
SECTION 1: ZdZd13TeV output – file/tree structure
======================================================================

[mc23a_mZd30]  /Users/matt/Documents/work/0-PhD/particle/xxzx-analysis/code_ZdZd/git_Zd/ZdZd-repos/run3_ZdZd_project/tmva_ZdZd_R3/../data/example_data/ZdZd13TeV/mc23a_mZd30/my.output.root
  Top-level directories (systematic variants): 57
    Nominal  +  56 systematic variations
    Systematics: EG_RESOLUTION_ALL1down, EG_RESOLUTION_ALL1up, EG_SCALE_ALL1down, EG_SCALE_ALL1up, JET_EtaIntercalibration_NonClosure_Pre...
  Other top-level objects: ['Nominal', 'EG_RESOLUTION_ALL1up', 'EG_RESOLUTION_ALL1down', 'EG_SCALE_ALL1up', 'EG_SCALE_ALL1down', 'MUONS_CB1up', 'MUONS_CB1down', 'MUONS_SCALE1up', 'MUONS_SCALE1down', 'MUONS_SAGITTA_DATASTAT1up', 'MUONS_SAGITTA_DATASTAT1down', 'MUONS_SAGITTA_RESBIAS1up', 'MUONS_SAGITTA_RESBIAS1down', 'MUONS_SAGITTA_GLOBAL1up', 'MUONS_SAGITTA_GLOBAL1down', 'MUONS_SAGITTA_PTEXTRA1up', 'MUONS_SAGITTA_PTEXTRA1down', 'JET_GroupedNP_11up', 'JET_GroupedNP_11down', 'JET_GroupedNP_21up', 'JET_GroupedNP_21down', 'JET_GroupedNP_31up', 'JET_GroupedNP_31down', 'JET_EtaIntercalibration_NonClosure_PreRec1up', 'JET_EtaIntercalibration_NonClosure_PreRec1down', 'JET_InSitu_NonClosure_PreRec1up', 'JET_InSitu_NonClosure_PreRec1down', 'JET_JESUnc_mc20vsmc21_AF3_PreRec1up', 'JET_JESUnc_mc20vsmc21_AF3_PreRec1down', 'JET_JESUnc_Noise_PreRec1up', 'JET_JESUnc_Noise_PreRec1down', 'JET_JESUnc_VertexingAlg_PreRec1up', 'JET_JESUnc_VertexingAlg_PreRec1down', 'JET_RelativeNonClosure_AF31up', 'JET_RelativeNonClosure_AF31down', 'JET_JER_DataVsMC_MC161up', 'JET_JER_DataVsMC_MC161down', 'JET_JER_EffectiveNP_11up', 'JET_JER_EffectiveNP_11down', 'JET_JER_EffectiveNP_21up', 'JET_JER_EffectiveNP_21down', 'JET_JER_EffectiveNP_31up', 'JET_JER_EffectiveNP_31down', 'JET_JER_EffectiveNP_41up', 'JET_JER_EffectiveNP_41down', 'JET_JER_EffectiveNP_51up', 'JET_JER_EffectiveNP_51down', 'JET_JER_EffectiveNP_61up', 'JET_JER_EffectiveNP_61down', 'JET_JER_EffectiveNP_7restTerm1up', 'JET_JER_EffectiveNP_7restTerm1down', 'JET_JERUnc_mc20vsmc21_AF3_PreRec1up', 'JET_JERUnc_mc20vsmc21_AF3_PreRec1down', 'JET_JERUnc_Noise_PreRec1up', 'JET_JERUnc_Noise_PreRec1down', 'JET_JERUnc_AF3vsFullSim_AF3_PreRec1up', 'JET_JERUnc_AF3vsFullSim_AF3_PreRec1down', 'zdzdJets_cutflow', 'channelInfo']

  Nominal/llllTree:
    Total entries (events): 60000
    Total branches:         250
    Scalar (event-level):   52
    Jagged (per-object):    198

  Jagged branches by collection:
    truth_l_               5 branches
      fields: tlv_pt, tlv_eta, tlv_phi, pdgId, from_photon
    l_                    18 branches
      fields: truthType, truthOrigin, tlv_pt, tlv_eta, tlv_phi, pdgId, d0Sig, type, ... (+10 more)
    hard_l_                4 branches
      fields: tlv_pt, tlv_eta, tlv_phi, pdgId
    truth_ll_              8 branches
      fields: tlv_pt, tlv_eta, tlv_phi, tlv_m, dR, dPhi, l1, l2
    hard_ll_               8 branches
      fields: tlv_pt, tlv_eta, tlv_phi, tlv_m, dR, dPhi, l1, l2
    truth_llll_           31 branches
      fields: tlv_pt, tlv_eta, tlv_phi, tlv_m, type, avgM, dM, charge, ... (+23 more)
    ll_                    8 branches
      fields: tlv_pt, tlv_eta, tlv_phi, tlv_m, dR, dPhi, l1, l2
    llll_                 66 branches
      fields: tlv_pt, tlv_eta, tlv_phi, tlv_m, type, avgM, dM, charge, ... (+58 more)
    j_                    46 branches
      fields: tlv_pt, tlv_eta, tlv_phi, tlv_m, matched_l, is_btagged_85, scaleFactor, scaleFactorDL1dv01, ... (+38 more)
    (unclassified): ['truth_nu_tlv_pt', 'truth_nu_tlv_eta', 'truth_nu_tlv_phi', 'truth_nu_pdgId']

  Example systematic tree (EG_RESOLUTION_ALL1up/llllTree):
    Branches: 94
    (Systematic trees only carry kinematic branches, not full metadata)

======================================================================
SECTION 3: Key distributions in ZdZd13TeV Nominal trees
======================================================================

==================================================
Sample: mc23a_mZd30
==================================================
Total events in file: 60000

--- 3a. Event-level preselection ---
  passCleaning:           60000 / 60000 (100.0%)
  + passNPV:              60000 / 60000 (100.0%)
  + passTriggers != 0:    51707 / 60000 (86.2%)
  events with llll_n>0:   23526 / 60000 (39.2%)

--- 3b. Object multiplicities ---
  leptons (l)              : n=0: 35895, n=4: 13006, n=5: 5718, n=6: 2688, n=7: 1277, n=8: 647, n=9: 357, n=10: 168, n=11: 107, n=12: 57, n=13: 30, n=14: 29, n=15: 11, n=16: 6, n=17: 3, n=18: 1
  dileptons (ll)           : n=0: 35897, n=1: 169, n=2: 11196, n=3: 460, n=4: 12115, n=5: 10, n=6: 147, n=7: 1, n=8: 3, n=9: 2
  quadruplets (llll)       : n=0: 36474, n=1: 10962, n=2: 12397, n=3: 3, n=4: 1, n=6: 157, n=12: 4, n=18: 2

--- 3c. Lepton kinematics (all reco leptons, MeV -> GeV) ---
  Total reco leptons: 118737 (electrons: 61266, muons: 57471)
  pT: n=118737, min=5, mean=29, median=23.6, max=961 GeV
  |η|: n=118737, min=-2.56, mean=0.000872, median=-0.0018, max=2.57 
  quality values: {0: 84743, 1: 8077, 2: 4236, 3: 516, 4: 21165}

--- 3d. Dilepton (ll) masses ---
  m_ll: n=73382, min=0.0192, mean=38.3, median=30.3, max=622 GeV
  Distribution (GeV):
  [    0.00,    10.83) |                                     207
  [   10.83,    21.67) | #                                   1457
  [   21.67,    32.50) | ################################### 48658
  [   32.50,    43.33) | ####                                5135
  [   43.33,    54.17) | ####                                5130
  [   54.17,    65.00) | ####                                4987
  [   65.00,    75.83) | ###                                 3989
  [   75.83,    86.67) | ##                                  2430
  [   86.67,    97.50) | #                                   1072
  [   97.50,   108.33) |                                     264
  [  108.33,   119.17) |                                     25
  [  119.17,   130.00) |                                     3

--- 3e. Selected quadruplet (llll[0]) properties ---
  Events with ≥1 quadruplet: 23526
  m_4l: n=23526, min=31.8, mean=123, median=124, max=324 GeV
  mab (leading dilepton): n=23526, min=12.2, mean=30.8, median=30.3, max=125 GeV
  mcd (subleading dilepton): n=23526, min=0.0195, mean=29.2, median=29.5, max=74 GeV
  avgM = (mab+mcd)/2: n=23526, min=6.16, mean=30, median=29.9, max=77.7 GeV
  ΔM   = mab-mcd: n=23526, min=0.000209, mean=1.64, median=0.889, max=95.8 GeV

  m_4l distribution (GeV):
  [  100.00,   103.33) |                                     104
  [  103.33,   106.67) | #                                   156
  [  106.67,   110.00) | #                                   225
  [  110.00,   113.33) | #                                   372
  [  113.33,   116.67) | ##                                  654
  [  116.67,   120.00) | #####                               1390
  [  120.00,   123.33) | #################                   5315
  [  123.33,   126.67) | ################################### 10635
  [  126.67,   130.00) | ###########                         3464
  [  130.00,   133.33) | ##                                  590
  [  133.33,   136.67) | #                                   173
  [  136.67,   140.00) |                                     77

  mab distribution (GeV):
  [    0.00,     5.83) |                                     0
  [    5.83,    11.67) |                                     0
  [   11.67,    17.50) |                                     4
  [   17.50,    23.33) |                                     5
  [   23.33,    29.17) | ##                                  1053
  [   29.17,    35.00) | ################################### 21837
  [   35.00,    40.83) |                                     234
  [   40.83,    46.67) |                                     63
  [   46.67,    52.50) |                                     105
  [   52.50,    58.33) |                                     135
  [   58.33,    64.17) |                                     63
  [   64.17,    70.00) |                                     15

  mcd distribution (GeV):
  [    0.00,     5.83) |                                     21
  [    5.83,    11.67) |                                     17
  [   11.67,    17.50) |                                     128
  [   17.50,    23.33) | #                                   553
  [   23.33,    29.17) | ###################                 7881
  [   29.17,    35.00) | ################################### 14573
  [   35.00,    40.83) |                                     37
  [   40.83,    46.67) |                                     52
  [   46.67,    52.50) |                                     101
  [   52.50,    58.33) |                                     111
  [   58.33,    64.17) |                                     45
  [   64.17,    70.00) |                                     6

--- 3f. Cutflow-relevant flags for selected quadruplet ---
  pdgIdSum=44 (4e   ): 16.4%
  pdgIdSum=48 (2e2mu): 47.9%
  pdgIdSum=52 (4mu  ): 35.7%

  charge==0 and dCharge==0 (SFOS):  100.0%
  overlaps & 0x36 == 0 (no OR):     100.0%
  triggerMatched != 0:               97.0%
  l_isIsolCloseBy == 15 (all isol):  82.2%
  nCTorSA < 2 (muon type ok):        99.9%
  min_sf_dR > 0.1:                   99.9%
  min_of_dR > 0.2:                   99.9%
  max_el_d0Sig < 5:                  99.7%
  max_mu_d0Sig < 3:                  98.2%

--- 3g. Lepton pTs inside selected quadruplet ---
    pT(leading        ): n=23526, min=13.4, mean=53.5, median=45.5, max=961 GeV
    pT(subleading     ): n=23526, min=7.98, mean=35.3, median=32.2, max=373 GeV
    pT(subsubleading  ): n=23526, min=5.52, mean=23.2, median=21.9, max=238 GeV
    pT(trailing       ): n=23526, min=5, mean=14.3, median=13.1, max=163 GeV

--- 3h. Potential BDT discriminant variables ---
  vtx_reduced_chi2: sentinel (-999) fraction = 0.7%
    vtx_reduced_chi2 (non-sentinel): n=23350, min=0.00317, mean=0.882, median=0.648, max=19.9 
    min_sf_dR (same-flavour ΔR): n=23526, min=0, mean=0.994, median=0.996, max=3.1 rad
  min_of_dR: sentinel (4e/4mu, no OF pairs) = 52.1% of quads
    min_of_dR (2e2mu quads only): n=11262, min=5.34e-05, mean=1.83, median=1.91, max=3.44 rad
    mcd/mab ratio: n=23526, min=0.000654, mean=0.948, median=0.971, max=1 
    truth_zdzd_avgM: n=60000, min=30, mean=30, median=30, max=30 GeV
    (confirms mZd ~ 30.0 ± 0.00 GeV)

==================================================
Sample: mc23a_mZd60
==================================================
Total events in file: 27707

--- 3a. Event-level preselection ---
  passCleaning:           27707 / 27707 (100.0%)
  + passNPV:              27707 / 27707 (100.0%)
  + passTriggers != 0:    27269 / 27707 (98.4%)
  events with llll_n>0:   27115 / 27707 (97.9%)

--- 3b. Object multiplicities ---
  leptons (l)              : n=4: 15071, n=5: 6554, n=6: 3049, n=7: 1527, n=8: 725, n=9: 367, n=10: 183, n=11: 100, n=12: 53, n=13: 38, n=14: 21, n=15: 7, n=16: 5, n=17: 2, n=18: 2, n=19: 1, n=20: 2
  dileptons (ll)           : n=1: 211, n=2: 13193, n=3: 490, n=4: 13615, n=5: 8, n=6: 185, n=7: 1, n=8: 3, n=9: 1
  quadruplets (llll)       : n=0: 592, n=1: 12969, n=2: 13943, n=3: 3, n=4: 2, n=6: 193, n=12: 4, n=18: 1

--- 3c. Lepton kinematics (all reco leptons, MeV -> GeV) ---
  Total reco leptons: 135808 (electrons: 71735, muons: 64073)
  pT: n=135808, min=5, mean=28.9, median=25.5, max=587 GeV
  |η|: n=135808, min=-2.57, mean=0.0045, median=-0.000556, max=2.56 
  quality values: {0: 97713, 1: 8930, 2: 4916, 3: 570, 4: 23679}

--- 3d. Dilepton (ll) masses ---
  m_ll: n=83717, min=0.00314, mean=52.2, median=58.9, max=449 GeV
  Distribution (GeV):
  [    0.00,    10.83) | #                                   1616
  [   10.83,    21.67) | ###                                 4042
  [   21.67,    32.50) | ###                                 5306
  [   32.50,    43.33) | ####                                6529
  [   43.33,    54.17) | ######                              8542
  [   54.17,    65.00) | ################################### 54317
  [   65.00,    75.83) | ##                                  2976
  [   75.83,    86.67) |                                     311
  [   86.67,    97.50) |                                     29
  [   97.50,   108.33) |                                     17
  [  108.33,   119.17) |                                     11
  [  119.17,   130.00) |                                     1

--- 3e. Selected quadruplet (llll[0]) properties ---
  Events with ≥1 quadruplet: 27115
  m_4l: n=27115, min=14.3, mean=123, median=124, max=262 GeV
  mab (leading dilepton): n=27115, min=3.7, mean=59.5, median=60.4, max=182 GeV
  mcd (subleading dilepton): n=27115, min=0.0592, mean=56.3, median=58.8, max=69.8 GeV
  avgM = (mab+mcd)/2: n=27115, min=3.01, mean=57.9, median=59.5, max=121 GeV
  ΔM   = mab-mcd: n=27115, min=0.000145, mean=3.19, median=1.76, max=121 GeV

  m_4l distribution (GeV):
  [  100.00,   103.33) |                                     160
  [  103.33,   106.67) | #                                   224
  [  106.67,   110.00) | #                                   360
  [  110.00,   113.33) | ##                                  519
  [  113.33,   116.67) | ###                                 909
  [  116.67,   120.00) | #####                               1874
  [  120.00,   123.33) | ##################                  6154
  [  123.33,   126.67) | ################################### 11928
  [  126.67,   130.00) | ###########                         3647
  [  130.00,   133.33) | ##                                  613
  [  133.33,   136.67) | #                                   211
  [  136.67,   140.00) |                                     85

  mab distribution (GeV):
  [    0.00,     5.83) |                                     3
  [    5.83,    11.67) |                                     28
  [   11.67,    17.50) |                                     88
  [   17.50,    23.33) |                                     176
  [   23.33,    29.17) |                                     173
  [   29.17,    35.00) |                                     173
  [   35.00,    40.83) |                                     167
  [   40.83,    46.67) |                                     175
  [   46.67,    52.50) |                                     213
  [   52.50,    58.33) | ##                                  1427
  [   58.33,    64.17) | ################################### 23362
  [   64.17,    70.00) | #                                   953

  mcd distribution (GeV):
  [    0.00,     5.83) |                                     29
  [    5.83,    11.67) |                                     107
  [   11.67,    17.50) |                                     154
  [   17.50,    23.33) |                                     202
  [   23.33,    29.17) |                                     209
  [   29.17,    35.00) | #                                   282
  [   35.00,    40.83) | #                                   372
  [   40.83,    46.67) | #                                   590
  [   46.67,    52.50) | ###                                 1320
  [   52.50,    58.33) | #################                   7867
  [   58.33,    64.17) | ################################### 15952
  [   64.17,    70.00) |                                     31

--- 3f. Cutflow-relevant flags for selected quadruplet ---
  pdgIdSum=44 (4e   ): 17.7%
  pdgIdSum=48 (2e2mu): 49.1%
  pdgIdSum=52 (4mu  ): 33.2%

  charge==0 and dCharge==0 (SFOS):  100.0%
  overlaps & 0x36 == 0 (no OR):     99.9%
  triggerMatched != 0:               98.0%
  l_isIsolCloseBy == 15 (all isol):  83.6%
  nCTorSA < 2 (muon type ok):        99.9%
  min_sf_dR > 0.1:                   99.3%
  min_of_dR > 0.2:                   96.9%
  max_el_d0Sig < 5:                  99.6%
  max_mu_d0Sig < 3:                  98.4%

--- 3g. Lepton pTs inside selected quadruplet ---
    pT(leading        ): n=27115, min=12.2, mean=46.5, median=39.1, max=470 GeV
    pT(subleading     ): n=27115, min=8.22, mean=35.4, median=31.3, max=352 GeV
    pT(subsubleading  ): n=27115, min=5.46, mean=25.5, median=24.5, max=230 GeV
    pT(trailing       ): n=27115, min=5, mean=18.2, median=18, max=130 GeV

--- 3h. Potential BDT discriminant variables ---
  vtx_reduced_chi2: sentinel (-999) fraction = 1.1%
    vtx_reduced_chi2 (non-sentinel): n=26828, min=0.0065, mean=0.892, median=0.648, max=19.8 
    min_sf_dR (same-flavour ΔR): n=27115, min=4.44e-16, mean=1.58, median=1.4, max=4.21 rad
  min_of_dR: sentinel (4e/4mu, no OF pairs) = 50.9% of quads
    min_of_dR (2e2mu quads only): n=13314, min=0.000142, mean=0.777, median=0.714, max=2.87 rad
    mcd/mab ratio: n=27115, min=0.000981, mean=0.945, median=0.97, max=1 

======================================================================
SECTION 2: ZdZdPostProcessing output – file structure
======================================================================

[mc23a_mZd30]  /Users/matt/Documents/work/0-PhD/particle/xxzx-analysis/code_ZdZd/git_Zd/ZdZd-repos/run3_ZdZd_project/tmva_ZdZd_R3/../data/example_data/ZdZdPostProcessing/mc23a_mZd30/myfile.root
  Object type: TH1 histograms only (no TTrees)
  Histogram variable: avgMll = (m12 + m34) / 2
  Regions found:  ['SR_2e2m', 'SR_4e', 'SR_4m', 'VR1_4e', 'VR1_4m', 'VR2_4e', 'VR2_4m', 'VR3_2e2m', 'VR3_4e', 'VR3_4m']
  Systematics found (26): EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1down, EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR__1up, EL_EFF_Iso_TOTAL_1...
  MC channel numbers (datasets): ['_5615']

  Example histogram: avgMll/SR_4m/_5615
    Bins: 64, range: [0.0, 64.0] GeV
    Integral (sum of bin contents): 3897.6856

======================================================================
SECTION 4: ZdZdPostProcessing histogram yields (events passing full cutflow)
======================================================================

[mc23a_mZd30]
  avgMll/SR_2e2m/_5615                           integral=7517.3164  bins=64
  avgMll/SR_4e/_5615                             integral=1760.0000  bins=64
  avgMll/SR_4m/_5615                             integral=3897.6856  bins=64

[mc23a_mZd60]
  avgMll/SR_2e2m/_561517                         integral=8740.9957  bins=64
  avgMll/SR_4e/_561517                           integral=2845.0000  bins=64
  avgMll/SR_4m/_561517                           integral=5973.6140  bins=64

======================================================================
SECTION 5: Recommended strategy for formatting data for BDT training
======================================================================

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


======================================================================
Analysis complete.
======================================================================
(root_env) matt@MConn-MacBook-Pro-2 tmva_ZdZd_R3 % 
(root_env) matt@MConn-MacBook-Pro-2 tmva_ZdZd_R3 % 
