"""Read-only validation of the published 450-run tuning export (standard library)."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/tuning'
ARCHS = ['lstm', 'xlstm', 'cnn_lstm', 'transformer', 'convtransformer']
SEEDS = [11, 29, 47]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(name):
    with (ROOT / name).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def validate():
    hashes = json.loads((ROOT / 'artifact_hashes.json').read_text())
    for name, expected in hashes.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                'Artifact hash mismatch: ' + name)
    merged = read_csv('tuning_merged.csv')
    expected = {(a, f'cfg_{i:02}', seed) for a, i, seed in
                itertools.product(ARCHS, range(30), SEEDS)}
    def identity(r):
        return r['architecture'], r['configuration_slot'], int(r['seed'])
    require(len(merged) == 450, 'Expected 450 registry rows')
    require({identity(r) for r in merged} == expected, 'Missing/duplicate/unexpected identity')
    require(len({r['run_id'] for r in merged}) == 450, 'Duplicate run ID')
    for r in merged:
        a, c, s = identity(r)
        require(r['run_id'] == f'{a}_{c}_seed{s}', 'Invalid run ID')
        require(r['status'] == 'completed', 'Non-completed tuning record')
    require({r['source_shard'] for r in merged} ==
            {f'results/shard_{i:02}' for i in range(10)}, 'Unexpected shards')
    require(all(sum(r['source_shard'] == f'results/shard_{i:02}' for r in merged) == 45
                for i in range(10)), 'Expected 45 runs per shard')
    metrics = read_csv('tuning_run_metrics.csv')
    require(len(metrics) == 450 and {identity(r) for r in metrics} == expected,
            'Metric export identity mismatch')
    indexed = {identity(r): r for r in metrics}
    for r in metrics:
        require(r['status'] == 'completed' and r['selection_feasible'] == 'True',
                'Infeasible/non-completed run')
        require(all(math.isfinite(float(r[k])) for k in ['f1','mae_all','fpr']), 'Nonfinite metric')
        require(0 <= float(r['fpr']) <= .20 + 1e-15, 'FPR constraint failed')
    saved = json.loads((ROOT / 'selected_configs.json').read_text())['choices']
    require(set(saved) == set(ARCHS), 'Expected exactly five selected architectures')
    output = []
    for arch in ARCHS:
        candidates = saved[arch]['all_configuration_summaries']
        require(len(candidates) == 30 and {s['configuration_slot'] for s in candidates} ==
                {f'cfg_{i:02}' for i in range(30)}, 'Candidate count/slots mismatch')
        for s in candidates:
            require(s['eligible'] is True and s['expected_seeds'] == SEEDS and
                    s['completed_feasible_seeds'] == SEEDS and
                    s['n_completed_feasible'] == 3 and not s['problems'], 'Invalid seed summary')
            for metric in ['f1','mae_all','fpr']:
                # Three binary64 values in the frozen seed order; exact comparison.
                mean = sum(float(indexed[(arch,s['configuration_slot'],seed)][metric])
                           for seed in SEEDS) / 3
                require(mean == s['mean_' + metric], 'Candidate mean differs: ' + arch + '/' + s['configuration_slot'])
        best = min(candidates, key=lambda s: (-s['mean_f1'],s['mean_mae_all'],s['mean_fpr']))
        require(best == saved[arch]['summary'] and
                best['configuration_slot'] == saved[arch]['configuration_slot'], 'Selection mismatch')
        output.append(dict(architecture=arch, selected_config=best['configuration_slot'],
                           mean_f1=best['mean_f1'], mean_mae_all=best['mean_mae_all'],
                           mean_fpr=best['mean_fpr'], valid_seeds=3))
    exported = read_csv('tuning_summary.csv')
    require(exported == [{k:str(v) for k,v in r.items()} for r in output], 'Summary CSV mismatch')
    comparison = json.loads((ROOT / 'scientific_comparison.json').read_text())
    require(comparison['status'] == 'PASS' and not comparison['differences'], 'Independent comparison failed')
    for c in comparison['comparisons']:
        choice = saved[c['architecture']]
        got = choice['configuration_slot'] if c['field'] == 'configuration_slot' else choice['summary'][c['field']]
        require(got == c['actual'] == c['expected'], 'Expected value mismatch')
    return output


if __name__ == '__main__':
    rows = validate()
    print('PASS: 450 unique completed runs; 10 x 45 shards; 150 candidate summaries; 5 selections; 20 exact comparisons.')
    print('architecture | selected_config | mean_f1 | mean_mae_all | mean_fpr | valid_seeds')
    for row in rows:
        print(' | '.join(map(str,row.values())))
    print('Validation/tuning only; no independent final test results.')
