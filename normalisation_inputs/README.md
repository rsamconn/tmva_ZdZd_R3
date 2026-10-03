# `normalisation_inputs/` — bundled copies of the normalisation CSVs

These three CSVs are **copies**, vendored into this repository so the BDT
pipeline is self-contained and does not require a sibling checkout of
`ZdZdPostProcessing` to be present on lxplus. They are the defaults for
`make_sample_manifest.py` (Stage 0), resolved relative to that script, and each
can be overridden with `--cutflow-inputs`, `--cross-sections` and `--sumw`.

## Contents

| File | What it carries | Used for |
|---|---|---|
| `cutflow_inputs.csv` | the production registry: process, DSID, campaign, `Merged_file_path`, `Ntuple_events` | which samples exist and where they live |
| `crossSections_run3.csv` | σ, k-factor, filter efficiency and `sigma_eff_pb` per DSID (28 DSIDs, all `status = OK`) | the numerator of `scale_d` |
| `sumw_total_p7266.csv` | `ΣW_total` per merged Ntuple, from the `Events_Processed` counter (44 of 50 samples) | the denominator of `scale_d` |

Stage 0 joins them on `Merged_file_path` and computes

```
scale_d = L × σ × k × ε_filt / ΣW_total = L[pb⁻¹] × sigma_eff_pb / ΣW_total
```

with `L` per campaign held in `LUMI_PB` in `make_sample_manifest.py`
(26,328.8 / 25,204.3 / 107,890.0 pb⁻¹ for mc23a / mc23d / mc23e, i.e.
2022 / 2023 / 2024), sourced from
`background/ATLAS_info/GRLs/GRL_calcs/` via `claude/lumi_run3_grl_lumicalc.md`.

## Provenance

Copied from `ZdZdPostProcessing/cutflow_automation/` on **2026-10-03**, at
commit `408e175a00af84646833afb8f8f5eda7723061c8` ("Major documentation
updates", 2026-09-28) of branch `r25_run3_ZdZd`.

| File | MD5 at copy | Source mtime |
|---|---|---|
| `cutflow_inputs.csv` | `42e9637a2f289a41ab3368a38f41e42a` | 2026-09-07 |
| `crossSections_run3.csv` | `e66184d169c7e10c36622e751c73b4a3` | 2026-09-25 |
| `sumw_total_p7266.csv` | `16282204e7d4b22dc25a4c496da38e78` | 2026-09-26 |

## Refreshing them

`ZdZdPostProcessing` remains the authority. When any of the three changes
there — a new production, a corrected cross-section, a recovered `ΣW_total` —
re-copy and re-record the table above:

```bash
CF=../background/current_code/ZdZdPostProcessing/cutflow_automation
cp -p "$CF"/{cutflow_inputs.csv,crossSections_run3.csv,sumw_total_p7266.csv} .
md5sum *.csv
```

Then regenerate the manifest, because `scale_d` is baked into it and from there
into the Parquet:

```bash
python3 ../make_sample_manifest.py --campaign mc23a \
    --signal-dir /eos/.../signal_Ntuples/mc23a_p6697_noSyst \
    --signal-dsids 561509 561511 561515 \
    --output ../../data/training_ntuples/manifest_mc23a.csv
```

A training run's `<parquet>.samples.json` sidecar records the `scale_d` actually
used per sample, so an older training can always be checked against the CSVs it
was built from.

## Known gaps in these files

- **Six samples have no `ΣW_total` row** — 603293 mc23a, 701185 mc23a/mc23d,
  701190 mc23a/mc23e, 701274 mc23e — because their `channelInfo` was written as
  an empty stub by an `hadd` lacking the AnalysisCam dictionary. For an mc23a
  training only **603293 mc23a** matters; the accepted estimate
  `ΣW_total ≈ 18,767` (believed ~0.4% high) is applied by default via
  `SUMW_OVERRIDES`, flagged in the manifest's `sumw_estimated` column and warned
  about on stdout. `--no-default-overrides` drops the sample instead.
- **701185 and 701190 are excluded** by default on m4l-overlap grounds with the
  inclusive 701040 `Sh_llll` sample, matching the named exclusion held by
  `aggregate_cutflows.py`.
- `crossSections_run3.csv` covers the reducible processes too, but the BDT
  pipeline only reads the DSIDs named in the registry rows it selects.
