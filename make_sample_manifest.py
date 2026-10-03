#!/usr/bin/env python3
"""
make_sample_manifest.py
=======================
Stage 0 of the TMVA pipeline: build the sample manifest that drives Stage 1.

The manifest is a CSV with one row per input Ntuple, carrying the path, the
physics process, the signal/background class, and the normalisation factor
`scale_d` that converts raw summed MC weights into expected yields:

    scale_d = (L * sigma * k * eps_filt) / SumW_total
            = (L[pb^-1] * sigma_eff_pb) / SumW_total

Writing it out as a file (rather than hardcoding paths in a driver script)
keeps the sample list in one place, lets it be regenerated when the production
registry changes, and makes the normalisation inputs auditable after the fact.

INPUTS
------
Three CSVs, copied into this repository under `normalisation_inputs/` so the
pipeline is self-contained and does not depend on a sibling checkout of
ZdZdPostProcessing being present.  They are the defaults for the three input
options, resolved relative to this script, and each can still be overridden.
See `normalisation_inputs/README.md` for provenance and how to refresh them.

    cutflow_inputs.csv        the production registry: process, DSID, campaign,
                              Merged_file_path, Ntuple_events
    crossSections_run3.csv    sigma, kFactor, genFiltEff, sigma_eff_pb per DSID
                              (28 DSIDs, all status = OK)
    sumw_total_p7266.csv      SumW_total per merged Ntuple (44 of 50 samples)

Luminosity comes from background/ATLAS_info/GRLs/GRL_calcs/ via
claude/lumi_run3_grl_lumicalc.md and is held in LUMI_PB below, per campaign.

SAMPLE SELECTION
----------------
Defaults follow the analysis' canonical irreducible set:

  * processes  H_ZZ_4l, ZZ_4l, Tribosons, ttbarZ
  * DSIDs 701185 and 701190 EXCLUDED on m4l-overlap grounds (they overlap the
    701040 inclusive Sh_llll sample); recorded in the cf_v2 production report
    and held as a named exclusion by aggregate_cutflows.py.
  * one campaign at a time (default mc23a), because scale_d is per campaign
    against its own year's luminosity.

MISSING SumW_total
------------------
Six registry samples have no SumW_total row, because their channelInfo was
written as empty stubs by an hadd lacking the AnalysisCam dictionary:
603293 mc23a, 701185 mc23a/mc23d, 701190 mc23a/mc23e, 701274 mc23e.

Of these only 603293 mc23a matters for an mc23a training, and the estimate
SumW_total ~ 18767 (believed ~0.4% high) is the one already in use for the
mc23a/data22 background estimate.  It is applied here by default via
SUMW_OVERRIDES and flagged in the manifest's `sumw_estimated` column, so a
training cannot silently depend on an estimate.  Pass --no-default-overrides
to drop it and have 603293 mc23a skipped instead.

SIGNAL
------
The signal DSIDs are not in crossSections_run3.csv and no signal Ntuple path is
recorded in the production registry, so signal rows get:

  * `path` as a glob pattern (resolved by Stage 1), since signal filenames carry
    a per-job ID,
  * `scale_d = 1.0`, i.e. no cross-section normalisation.

That is deliberate and matches the cf_v2 convention of ignoring the ~1e-18 ZdZd
generator weights entirely: the signal class is rescaled as a whole by TMVA's
NormMode=EqualNumEvents, so an absolute signal normalisation would be discarded
anyway.  What scale_d = 1.0 does mean is that the relative weight of the
different mZd samples is set by their selection acceptance.  Use
--signal-equalise to give each signal DSID the same total weight instead.

USAGE
-----
    python3 make_sample_manifest.py \\
        --campaign        mc23a \\
        --signal-dir      /eos/.../signal_Ntuples/mc23a_p6697_noSyst \\
        --signal-dsids    561509 561511 561515 \\
        --output          data/training_ntuples/manifest_mc23a.csv

with --cutflow-inputs / --cross-sections / --sumw only needed to point at
copies other than the ones bundled in `normalisation_inputs/`.
"""

import argparse
import csv
import os
import sys


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The bundled normalisation inputs, resolved relative to this script so the
# defaults work from any working directory.
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "normalisation_inputs")
DEFAULT_CUTFLOW_INPUTS = os.path.join(INPUT_DIR, "cutflow_inputs.csv")
DEFAULT_CROSS_SECTIONS = os.path.join(INPUT_DIR, "crossSections_run3.csv")
DEFAULT_SUMW = os.path.join(INPUT_DIR, "sumw_total_p7266.csv")

# Integrated luminosity per MC campaign, pb^-1, LAr-Corrected column.
# Source: background/ATLAS_info/GRLs/GRL_calcs/, documented in
# claude/lumi_run3_grl_lumicalc.md; all three match the official TWiki.
# Campaign -> data year mapping: mc23a<->2022, mc23d<->2023, mc23e<->2024.
LUMI_PB = {
    "mc23a": 26_328.8,    # 2022
    "mc23d": 25_204.3,    # 2023
    "mc23e": 107_890.0,   # 2024
}

# The analysis' canonical irreducible background set.
DEFAULT_PROCESSES = ["H_ZZ_4l", "ZZ_4l", "Tribosons", "ttbarZ"]

# Excluded on m4l-overlap grounds with the inclusive 701040 Sh_llll sample.
DEFAULT_EXCLUDE_DSIDS = [701185, 701190]

# SumW_total values not present in sumw_total_p7266.csv, keyed (DSID, campaign).
# 603293 mc23a: estimated 18767, believed ~0.4% high; the same value is used by
# the mc23a/data22 irreducible estimate (--allow-estimated-sumw there).
SUMW_OVERRIDES = {
    (603293, "mc23a"): 18_767.0,
}

# Signal DSID -> mZd [GeV], from production_lists.txt / signal_mc23_DAODs.txt.
SIGNAL_MZD = {
    561504: 5, 561505: 8, 561506: 10, 561507: 12, 561508: 15, 561509: 20,
    561511: 30, 561515: 50, 561516: 55, 561517: 60,
}

DEFAULT_SIGNAL_PATTERN = "user.*.{dsid}.*.my.output.root"

MANIFEST_COLUMNS = [
    "path",              # file path, or a glob pattern (signal)
    "process",           # physics process short name, or "signal_mZd<N>"
    "sample_class",      # "signal" or "background"
    "dsid",
    "campaign",
    "sample_stem",
    "sigma_eff_pb",      # sigma * k * eps_filt, pb
    "sumw_total",
    "luminosity_pb",
    "scale_d",           # L * sigma_eff_pb / sumw_total
    "sumw_estimated",    # 1 if sumw_total came from SUMW_OVERRIDES
    "notes",
]


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------

def read_cross_sections(path):
    """Return {dsid: sigma_eff_pb} for rows with status == OK."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "").strip().upper() != "OK":
                continue
            out[int(row["dsid"])] = float(row["sigma_eff_pb"])
    if not out:
        raise ValueError(f"No usable cross-section rows in {path}")
    return out


def read_sumw(path):
    """Return {merged_file_path: sumw_total} from the Events_Processed counter."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("counter", "").strip() != "Events_Processed":
                continue
            out[row["file"].strip()] = float(row["sumw"])
    if not out:
        raise ValueError(f"No Events_Processed rows in {path}")
    return out


def read_registry(path):
    """Return the production registry rows as a list of dicts."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

def build_background_rows(registry, xsec, sumw, campaign, processes,
                          exclude_dsids, lumi_pb, overrides):
    """Build the background manifest rows, plus a list of skip reasons."""
    rows, skipped = [], []

    for reg in registry:
        proc = reg["Physics_process_short"].strip()
        if proc not in processes:
            continue
        if reg["MC_campaign"].strip() != campaign:
            continue

        dsid = int(reg["DSID"])
        path = reg["Merged_file_path"].strip()
        stem = reg.get("sample_stem", "").strip()

        if dsid in exclude_dsids:
            skipped.append((proc, dsid, "excluded (m4l overlap with 701040)"))
            continue
        if not path:
            skipped.append((proc, dsid, "no Merged_file_path in registry"))
            continue
        if dsid not in xsec:
            skipped.append((proc, dsid, "no OK cross-section row"))
            continue

        sigma_eff = xsec[dsid]
        estimated = 0
        sumw_total = sumw.get(path)
        if sumw_total is None:
            sumw_total = overrides.get((dsid, campaign))
            if sumw_total is None:
                skipped.append(
                    (proc, dsid, "no SumW_total row and no override"))
                continue
            estimated = 1

        rows.append({
            "path": path,
            "process": proc,
            "sample_class": "background",
            "dsid": dsid,
            "campaign": campaign,
            "sample_stem": stem,
            "sigma_eff_pb": f"{sigma_eff:.9e}",
            "sumw_total": f"{sumw_total:.9e}",
            "luminosity_pb": f"{lumi_pb:.1f}",
            "scale_d": f"{lumi_pb * sigma_eff / sumw_total:.9e}",
            "sumw_estimated": estimated,
            "notes": "SumW_total estimated" if estimated else "",
        })

    rows.sort(key=lambda r: (r["process"], r["dsid"]))
    return rows, skipped


def build_signal_rows(signal_dir, signal_dsids, pattern, campaign, equalise):
    """Build the signal manifest rows.  `path` stays a glob for Stage 1."""
    rows = []
    for dsid in signal_dsids:
        mzd = SIGNAL_MZD.get(dsid)
        name = f"signal_mZd{mzd}" if mzd else f"signal_{dsid}"
        rows.append({
            "path": os.path.join(signal_dir, pattern.format(dsid=dsid)),
            "process": name,
            "sample_class": "signal",
            "dsid": dsid,
            "campaign": campaign,
            "sample_stem": "",
            "sigma_eff_pb": "",
            "sumw_total": "",
            "luminosity_pb": "",
            # No Run 3 cross-section for the signal DSIDs, and the signal class
            # is rescaled as a whole by NormMode=EqualNumEvents regardless.
            "scale_d": "1.000000e+00",
            "sumw_estimated": 0,
            "notes": ("equalise per DSID in Stage 2" if equalise
                      else "no cross-section normalisation"),
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_manifest_summary(rows, skipped):
    sep = "=" * 78
    print(f"\n{sep}\nManifest summary\n{sep}")

    by_proc = {}
    for r in rows:
        by_proc.setdefault(r["process"], []).append(r)

    print(f"  {'process':<20}{'files':>7}{'scale_d range':>34}{'est.':>6}")
    for proc in sorted(by_proc):
        rs = by_proc[proc]
        scales = [float(r["scale_d"]) for r in rs]
        n_est = sum(int(r["sumw_estimated"]) for r in rs)
        if min(scales) == max(scales):
            rng = f"{scales[0]:.4e}"
        else:
            rng = f"{min(scales):.4e} .. {max(scales):.4e}"
        print(f"  {proc:<20}{len(rs):>7}{rng:>34}{n_est:>6}")

    print(f"\n  Total samples: {len(rows)}"
          f"  (signal {sum(1 for r in rows if r['sample_class'] == 'signal')},"
          f" background {sum(1 for r in rows if r['sample_class'] == 'background')})")

    n_est = sum(int(r["sumw_estimated"]) for r in rows)
    if n_est:
        print(f"\n  WARNING: {n_est} sample(s) use an ESTIMATED SumW_total:")
        for r in rows:
            if int(r["sumw_estimated"]):
                print(f"    {r['process']:<14} {r['dsid']} {r['campaign']}"
                      f"  SumW_total = {float(r['sumw_total']):.6g}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} registry row(s):")
        for proc, dsid, why in skipped:
            print(f"    {proc:<14} {dsid}  {why}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 0: build the Stage 1 sample manifest with scale_d.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cutflow-inputs", default=DEFAULT_CUTFLOW_INPUTS,
                   metavar="CSV",
                   help="Production registry CSV "
                        "(default: normalisation_inputs/cutflow_inputs.csv).")
    p.add_argument("--cross-sections", default=DEFAULT_CROSS_SECTIONS,
                   metavar="CSV",
                   help="Cross-section CSV "
                        "(default: normalisation_inputs/crossSections_run3.csv).")
    p.add_argument("--sumw", default=DEFAULT_SUMW, metavar="CSV",
                   help="SumW_total CSV "
                        "(default: normalisation_inputs/sumw_total_p7266.csv).")
    p.add_argument("--campaign", default="mc23a", choices=sorted(LUMI_PB),
                   help="MC campaign to build the manifest for (default mc23a).")
    p.add_argument("--processes", nargs="+", default=DEFAULT_PROCESSES,
                   metavar="PROC",
                   help=f"Background processes (default: {' '.join(DEFAULT_PROCESSES)}).")
    p.add_argument("--exclude-dsids", nargs="*", type=int,
                   default=DEFAULT_EXCLUDE_DSIDS, metavar="DSID",
                   help=f"DSIDs to exclude (default: {DEFAULT_EXCLUDE_DSIDS}).")
    p.add_argument("--luminosity-pb", type=float, default=None,
                   help="Override the campaign luminosity in pb^-1.")
    p.add_argument("--no-default-overrides", action="store_true",
                   help="Do not apply the built-in estimated SumW_total values; "
                        "samples without a measured SumW_total are skipped.")
    p.add_argument("--sumw-override", nargs="*", default=[], metavar="D:C=V",
                   help="Extra SumW_total overrides, e.g. 603293:mc23a=18767.")
    p.add_argument("--signal-dir", default=None, metavar="DIR",
                   help="Directory holding the signal Ntuples.")
    p.add_argument("--signal-dsids", nargs="*", type=int, default=[],
                   metavar="DSID", help="Signal DSIDs to include.")
    p.add_argument("--signal-pattern", default=DEFAULT_SIGNAL_PATTERN,
                   help="Signal filename glob, with {dsid} substituted "
                        f"(default: {DEFAULT_SIGNAL_PATTERN}).")
    p.add_argument("--signal-equalise", action="store_true",
                   help="Mark signal rows so Stage 2 gives each signal DSID the "
                        "same total weight.")
    p.add_argument("--output", required=True, metavar="CSV",
                   help="Output manifest CSV path.")
    return p.parse_args(argv)


def parse_overrides(specs, base):
    out = dict(base)
    for spec in specs:
        try:
            key, value = spec.split("=", 1)
            dsid_s, campaign = key.split(":", 1)
            out[(int(dsid_s), campaign)] = float(value)
        except ValueError:
            raise SystemExit(
                f"ERROR: bad --sumw-override '{spec}'; expected DSID:CAMPAIGN=VALUE")
    return out


def main(argv=None):
    args = parse_args(argv)

    for path in (args.cutflow_inputs, args.cross_sections, args.sumw):
        if not os.path.exists(path):
            sys.exit(
                f"ERROR: input not found: {path}\n"
                f"       The bundled copies live in {INPUT_DIR}; see its "
                f"README.md for how they are refreshed.")

    lumi_pb = args.luminosity_pb
    if lumi_pb is None:
        lumi_pb = LUMI_PB[args.campaign]
    print(f"Registry    : {args.cutflow_inputs}")
    print(f"Cross-sect. : {args.cross_sections}")
    print(f"SumW_total  : {args.sumw}")
    print(f"Campaign    : {args.campaign}")
    print(f"Luminosity  : {lumi_pb:,.1f} pb^-1")
    print(f"Processes   : {', '.join(args.processes)}")
    print(f"Excluded    : {args.exclude_dsids or 'none'}")

    base_overrides = {} if args.no_default_overrides else SUMW_OVERRIDES
    overrides = parse_overrides(args.sumw_override, base_overrides)

    xsec = read_cross_sections(args.cross_sections)
    sumw = read_sumw(args.sumw)
    registry = read_registry(args.cutflow_inputs)

    bkg_rows, skipped = build_background_rows(
        registry, xsec, sumw, args.campaign, set(args.processes),
        set(args.exclude_dsids), lumi_pb, overrides,
    )
    if not bkg_rows:
        sys.exit("ERROR: no background samples selected; check --campaign "
                 "and --processes against the registry.")

    sig_rows = []
    if args.signal_dsids:
        if not args.signal_dir:
            sys.exit("ERROR: --signal-dsids requires --signal-dir.")
        sig_rows = build_signal_rows(
            args.signal_dir, args.signal_dsids, args.signal_pattern,
            args.campaign, args.signal_equalise,
        )
    else:
        print("\n  Note: no --signal-dsids given; manifest is background-only.")

    rows = sig_rows + bkg_rows
    print_manifest_summary(rows, skipped)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {args.output}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
