# Interim confirmatory tuning status — 12 September 2026

> **Interim validation-only record. Not a final architecture ranking.**

This snapshot documents the frozen confirmatory tuning campaign while execution is still in progress. It is not a test-set result, a final model selection, or an authorization of the planned final phase.

## Campaign progress

- Campaign: `tfm_campaign_v2`
- Planned tuning identities: **450**
- Completed: **380/450 (84.4%)**
- Remaining: **70**
- Failed: **0**
- Tuning seeds: **11, 29, 47**
- Test data used during tuning: **no**

| Architecture | Completed | Status |
|---|---:|---|
| CNN-LSTM | 90/90 | Complete |
| ConvTransformer | 90/90 | Complete |
| Transformer | 90/90 | Complete |
| LSTM | 90/90 | Complete |
| xLSTM | 20/90 | In progress; 70 remaining |

At this snapshot, all remaining tuning identities belong to xLSTM. The four completed architectures can be inspected descriptively while xLSTM continues, but the frozen cross-architecture selection procedure must wait until all 450 tuning identities have been accounted for.

## Interim aggregation for the four completed architectures

Each configuration below aggregates the three frozen tuning seeds. `mean_F1` is the mean validation F1 across seeds 11, 29 and 47; `std_F1` is the population standard deviation across those same three seeds. These are descriptive tuning statistics only.

### CNN-LSTM

| Configuration | mean_F1 | std_F1 | min_F1 | max_F1 |
|---|---:|---:|---:|---:|
| `cfg_06` | 0.800410 | 0.001294 | 0.799473 | 0.802241 |
| `cfg_01` | 0.798902 | 0.000180 | 0.798676 | 0.799115 |
| `cfg_13` | 0.798475 | 0.001299 | 0.796746 | 0.799880 |

### ConvTransformer

| Configuration | mean_F1 | std_F1 | min_F1 | max_F1 |
|---|---:|---:|---:|---:|
| `cfg_12` | 0.809751 | 0.001498 | 0.808625 | 0.811867 |
| `cfg_29` | 0.809554 | 0.001213 | 0.807842 | 0.810490 |
| `cfg_09` | 0.809468 | 0.001029 | 0.808540 | 0.810902 |

### Transformer

| Configuration | mean_F1 | std_F1 | min_F1 | max_F1 |
|---|---:|---:|---:|---:|
| `cfg_06` | 0.802209 | 0.008515 | 0.790769 | 0.811186 |
| `cfg_02` | 0.796591 | 0.003346 | 0.791860 | 0.799039 |
| `cfg_05` | 0.751046 | 0.033761 | 0.703620 | 0.779530 |

### LSTM

| Configuration | mean_F1 | std_F1 | min_F1 | max_F1 |
|---|---:|---:|---:|---:|
| `cfg_02` | 0.800651 | 0.002072 | 0.798021 | 0.803085 |
| `cfg_03` | 0.800361 | 0.000700 | 0.799398 | 0.801042 |
| `cfg_12` | 0.798980 | 0.003310 | 0.794329 | 0.801767 |

## Interpretation boundary

The values above are **not** the official selected configurations and do not establish a definitive winning architecture. The frozen selection procedure still requires the complete tuning campaign and its feasibility/ranking rules. xLSTM is not yet comparable because only 20 of its 90 tuning identities were complete at this snapshot.

The planned final phase remains separate: one configuration per architecture will only be frozen after completion of tuning under the existing protocol, and the planned final seeds are `1, 2, 3, 42, 123`. Those 25 final runs are **not launched or authorized by this document**.

The completed runs also show that multiple seeds are informative: some configurations are highly stable across seeds while others are much more variable or collapse to degenerate predictions. Such cases remain part of the tuning record rather than being silently discarded.
