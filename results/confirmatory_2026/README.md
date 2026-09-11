# Confirmatory results — publication pending

The September 2026 campaign is in progress according to the campaign owner's record. This directory intentionally contains no performance results or fabricated final artifacts. Historical results elsewhere in the repository remain separate.

After all 450 tuning identities have been accounted for under the frozen rules, and the separately approved final phase has finished, suitable public exports may include:

| Artifact | Intended content |
|---|---|
| `summary.csv` | Aggregate tuning outcomes, with explicit scope and run counts |
| `per_run_metrics.csv` | Audited metrics and status per scientific identity, without silently omitting failures |
| `selected_configs.json` | Configurations selected by the frozen validation-only procedure |
| `provenance_public.json` | Public code, protocol, plan, data and split digests and environment metadata |
| `final_multiseed_summary.csv` | Results of the authorized final five-seed procedure per architecture |

These are intended exports, not files already produced. The 25 final runs with seeds `1, 2, 3, 42, 123` are not yet authorized. Test must not influence tuning or selection; final evaluation follows the existing protocol only after the relevant decisions are frozen.

Before publication, verify completeness and identity uniqueness, retain failure accounting, separate tuning from final evaluation, and remove private credentials, cloud resource identifiers and billing information. Do not include datasets, checkpoints, locks or mass runtime output. Do not reinterpret partial results as a definitive architecture ranking.

See [the campaign record](../../docs/confirmatory_campaign_2026.md) for provenance and limitations.
