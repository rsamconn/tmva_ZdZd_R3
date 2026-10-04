#!/usr/bin/env python3
"""
sensitivity_compare.py
======================
Irreducible-background-only sensitivity comparison: cut-and-count (cf_v2
MediumSR) against the TMVA BDT, mc23a / 2022 luminosity.

Both arms are expressed as efficiencies relative to a COMMON denominator --
the SR cutflow row `LeptonsPt` (cleaning + NPV + trigger + SFOS + pT), which
is where the BDT is inserted.  The absolute signal normalisation (sigma x BR)
is unknown, so the headline number is the normalisation-free ratio

    R = (eps_S,BDT / eps_S,cut) * sqrt(eps_B,cut / eps_B,BDT)

which is the exact ratio of S/sqrt(B) between the two selections.  Absolute
Z_A is tabulated against a free overall signal scale k.

The BDT background yield is NOT taken from the BDT run's own 'expected yield'
column: that column is biased low by Stage 1 subsampling (see the report).
Instead the BDT's per-process background EFFICIENCIES (a ratio measured inside
its own sample) are applied to the cf_v2 normalised yields.

Inputs, all local:
  background/current_code/ZdZdPostProcessing/cutflow_automation/
      crossSections_run3.csv, sumw_total_p7266.csv
  data/example_data/ZdZdPostProcessing/cf_v2/bkg/signal_region/<proc>/*.SR.csv
  data/example_data/ZdZdPostProcessing/cf_v2/signal/mc23a_p6697_SR/*.csv
  terminal_outputs/20261003_bdt-output.md   (eps_B table, transcribed below)

Usage:  python3 sensitivity_compare.py [--project DIR] [--den CUT]
"""
import argparse, csv, glob, math, os, sys

LUMI_PB = {"mc23a": 26328.8, "mc23d": 25204.3, "mc23e": 107890.0}
EXCLUDE = {701185, 701190}                 # m4l overlap with 701040
SUMW_OVERRIDE = {(603293, "mc23a"): 18767.0}   # accepted estimate, ~0.4% high
PROCS = {
    "H_ZZ_4l":   [601500, 601501, 601502, 601503, 601504, 601505, 601634, 604263],
    "ZZ_4l":     [603293, 701040],
    "Tribosons": [701236, 701238, 701240, 701241],
    "ttbarZ":    [701274],
}
# Stage 3 of the 2026-10-03 lxplus run: background efficiency per process at
# fixed INCLUSIVE signal efficiency (test half, 3 signal masses combined).
BDT_EPS_B = {
    0.50: {"H_ZZ_4l": 0.00205, "Tribosons": -1.37e-05, "ZZ_4l": 7.67e-05, "ttbarZ": 0.0},
    0.80: {"H_ZZ_4l": 0.0118,  "Tribosons":  4.11e-05, "ZZ_4l": 2.34e-04, "ttbarZ": 0.00621},
    0.90: {"H_ZZ_4l": 0.0299,  "Tribosons":  2.72e-04, "ZZ_4l": 9.30e-04, "ttbarZ": 0.00621},
}
# N_eff per process in the BDT TEST half (full-sample N_eff / 2, same run)
BDT_NEFF_TEST = {"H_ZZ_4l": 22947.3/2, "Tribosons": 15982.4/2,
                 "ZZ_4l": 25285.8/2,   "ttbarZ": 330.9/2}


def asimov_Z(S, B):
    """Discovery significance, Cowan et al. Eq. (97). No background systematic."""
    if B <= 0 or S <= 0:
        return float("nan")
    return math.sqrt(2.0 * ((S + B) * math.log(1.0 + S / B) - S))


def load_scales(ca_dir, campaign):
    xs = {int(r["dsid"]): float(r["sigma_eff_pb"])
          for r in csv.DictReader(open(os.path.join(ca_dir, "crossSections_run3.csv")))}
    sumw = {}
    for r in csv.DictReader(open(os.path.join(ca_dir, "sumw_total_p7266.csv"))):
        d, f = int(r["mc_channel_number"]), r["file"]
        for c in LUMI_PB:
            if "." + c + "." in f:
                sumw[(d, c)] = float(r["sumw"])
    sumw.update(SUMW_OVERRIDE)
    L = LUMI_PB[campaign]
    out = {}
    for ds in PROCS.values():
        for d in ds:
            if d in EXCLUDE:
                continue
            key = (d, campaign)
            if key not in sumw:
                sys.exit(f"ERROR: no SumW_total for DSID {d} {campaign}")
            out[d] = L * xs[d] / sumw[key]
    return out


def load_background(sr_dir, scales, campaign):
    """-> {process: {cut: (yield, variance)}}, cf_v2 SR tables, normalised."""
    per = {}
    for proc, ds in PROCS.items():
        per[proc] = {}
        for d in ds:
            if d in EXCLUDE:
                continue
            g = [f for f in glob.glob(f"{sr_dir}/{proc}/{d}.*.{campaign}.*.SR.csv")
                 if ".shard" not in f]
            if len(g) != 1:
                sys.exit(f"ERROR: expected 1 combined CSV for {d}, found {len(g)}")
            sc = scales[d]
            for r in csv.DictReader(open(g[0])):
                c = r["Cut"]
                y, v = per[proc].get(c, (0.0, 0.0))
                per[proc][c] = (y + sc * float(r["weights_All"] or 0),
                                v + sc * sc * float(r["sumw2_All"] or 0))
    return per


def load_signal_eff(sig_dir, den):
    eps = {}
    for f in glob.glob(os.path.join(sig_dir, "*noSyst_mZd*.csv")):
        m = int(f.split("mZd")[-1].split(".")[0])
        rd = {r["Cut"]: r for r in csv.DictReader(open(f))}
        d = float(rd[den]["weights_All"])
        eps[m] = (float(rd["MediumSR"]["weights_All"]) / d, d)
    return dict(sorted(eps.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.path.expanduser("~/mnt/run3_ZdZd_project"))
    ap.add_argument("--campaign", default="mc23a")
    ap.add_argument("--den", default="LeptonsPt",
                    help="common denominator cutflow row (where the BDT is inserted)")
    a = ap.parse_args()

    ca  = f"{a.project}/background/current_code/ZdZdPostProcessing/cutflow_automation"
    srb = f"{a.project}/data/example_data/ZdZdPostProcessing/cf_v2/bkg/signal_region"
    sig = f"{a.project}/data/example_data/ZdZdPostProcessing/cf_v2/signal/{a.campaign}_p6697_SR"

    scales = load_scales(ca, a.campaign)
    bkg    = load_background(srb, scales, a.campaign)
    den    = a.den

    B_den  = sum(bkg[p][den][0] for p in PROCS)
    B_cut  = sum(bkg[p]["MediumSR"][0] for p in PROCS)
    B_cut_e = math.sqrt(sum(bkg[p]["MediumSR"][1] for p in PROCS))

    print(f"=== cf_v2 irreducible background, {a.campaign}, "
          f"L = {LUMI_PB[a.campaign]:,.1f} pb^-1 ===\n")
    print(f"{'process':12s} {den:>12s} {'MediumSR':>12s} {'eps_B':>11s}")
    for p in PROCS:
        print(f"{p:12s} {bkg[p][den][0]:12.4f} {bkg[p]['MediumSR'][0]:12.4f} "
              f"{bkg[p]['MediumSR'][0]/bkg[p][den][0]:11.3e}")
    print(f"{'TOTAL':12s} {B_den:12.4f} {B_cut:12.4f} {B_cut/B_den:11.3e}")
    print(f"  MediumSR MC stat: +- {B_cut_e:.4f}\n")

    eps_cut = load_signal_eff(sig, den)
    print("=== cut-and-count signal efficiency (MediumSR / " + den + ") ===\n")
    for m, (e, d) in eps_cut.items():
        note = "   <-- no acceptance (LooseSR m_ll > 10 GeV)" if e == 0 else ""
        print(f"  mZd {m:3d} GeV : {e:.4f}{note}")

    print("\n=== BDT background, cf_v2 normalisation x per-process eps_B ===\n")
    hdr = f"{'eps_S':>6s} " + " ".join(f"{p:>11s}" for p in PROCS) + \
          f" {'B':>9s} {'MCstat':>8s} {'eps_B':>10s}"
    print(hdr); print("-" * len(hdr))
    B_bdt = {}
    for e, tab in sorted(BDT_EPS_B.items()):
        parts = {p: bkg[p][den][0] * tab[p] for p in PROCS}
        B = sum(parts.values())
        var = sum((parts[p] ** 2) / (BDT_NEFF_TEST[p] * abs(tab[p]))
                  for p in PROCS if tab[p] != 0)
        B_bdt[e] = (B, math.sqrt(var))
        print(f"{e:6.2f} " + " ".join(f"{parts[p]:11.4f}" for p in PROCS) +
              f" {B:9.4f} {math.sqrt(var):8.4f} {B/B_den:10.3e}")

    print("\n=== Relative figure of merit (signal normalisation cancels) ===")
    print("  R = (eps_S,BDT/eps_S,cut) * sqrt(eps_B,cut/eps_B,BDT)\n")
    cols = sorted(BDT_EPS_B)
    print(f"{'mZd':>5s} {'eps_S,cut':>10s} | " +
          " | ".join(f"R @ eps_S={e:.2f}" for e in cols))
    for m, (ec, _) in eps_cut.items():
        if ec == 0:
            continue
        print(f"{m:5d} {ec:10.4f} | " +
              " | ".join(f"{(e/ec)*math.sqrt(B_cut/B_bdt[e][0]):13.2f}" for e in cols))

    print("\n=== Absolute Z_A vs free signal scale k  (S = k * eps_S) ===\n")
    ref = 30 if 30 in eps_cut else max(eps_cut)
    print(f"  reference mass for eps_S,cut: mZd {ref} GeV = {eps_cut[ref][0]:.4f}\n")
    print(f"{'k':>7s} {'S_cut':>7s} {'Z_cut':>7s} | " +
          " | ".join(f"{'S':>6s} {'Z':>6s}@{e:.2f}" for e in cols))
    for k in (0.5, 1, 2, 5, 10, 20):
        s = k * eps_cut[ref][0]
        print(f"{k:7.1f} {s:7.3f} {asimov_Z(s, B_cut):7.3f} | " +
              " | ".join(f"{k*e:6.3f} {asimov_Z(k*e, B_bdt[e][0]):6.3f}     " for e in cols))


if __name__ == "__main__":
    main()
