"""Read-only checks of the sealed final-training/calibration export; no test access."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/final_validation_2026'
ARCHS = dict(lstm='LSTM', xlstm='xLSTM', cnn_lstm='CNN-LSTM',
             transformer='Transformer', convtransformer='ConvTransformer')
SEEDS = {1, 2, 3, 42, 123}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def validate():
    rows = read_csv('final_25_runs.csv')
    summary = read_csv('architecture_summary.csv')
    proof = json.loads((DATA / 'source_provenance.json').read_text(encoding='utf-8'))
    tuning = json.loads((ROOT / 'results/tuning/selected_configs.json').read_text(encoding='utf-8'))
    choices = {a: tuning['choices'][a]['configuration_slot'] for a in ARCHS}
    require(choices == dict(zip(ARCHS, ['cfg_02', 'cfg_09', 'cfg_06', 'cfg_06', 'cfg_12'])),
            'Unexpected selected slots')
    expected = {f'{a}_{choices[a]}_seed{s}' for a in ARCHS for s in SEEDS}
    require(len(rows) == 25 and {r['run_id'] for r in rows} == expected,
            'Missing, duplicate or extra final run')
    require(len(proof['runs']) == 25 and {r['run_id'] for r in proof['runs']} == expected,
            'Proof identities differ')
    evidence = {r['run_id']: r for r in proof['runs']}
    require(proof['metric_source'] == 'final_seal.json/calibration_metrics', 'Wrong metric source')
    for r in rows:
        require(r['status'] == 'completed' and r['metric_split'] == 'validation_calibration',
                'Wrong status or metric split')
        require(r['configuration_slot'] == choices[r['architecture']] and
                r['run_id'] == f"{r['architecture']}_{r['configuration_slot']}_seed{int(r['seed'])}",
                'Identity mismatch')
        require(0 <= int(r['best_epoch']) < int(r['epochs_completed']) <= 100, 'Invalid epoch')
        for key in ['best_epoch', 'epochs_completed']:
            require(int(r[key]) == evidence[r['run_id']][key], 'Epoch evidence mismatch')
        m = evidence[r['run_id']]['calibration_metrics']
        for key in ['f1', 'mae_all', 'fpr']:
            require(math.isfinite(float(r[key])) and float(r[key]) == m[key], 'Metric mismatch: ' + key)
        threshold = float(r['frozen_threshold'])
        require(threshold == m['threshold'] and 0 <= threshold <= 1 and
                abs(100 * threshold - round(100 * threshold)) < 1e-8, 'Invalid frozen threshold')
        require(0 <= m['fpr'] <= .20, 'FPR cap violated')
        require(m['fpr'] == m['fp'] / (m['fp'] + m['tn']), 'FPR counts inconsistent')
        # Float arithmetic order may differ; counts are a separate sanity check,
        # whereas export-to-seal metric agreement above is exact.
        require(math.isclose(m['f1'], 2*m['tp']/(2*m['tp']+m['fp']+m['fn']),
                             rel_tol=1e-14, abs_tol=0), 'F1 counts inconsistent')
    require(len(summary) == 5 and {r['architecture'] for r in summary} == set(ARCHS),
            'Summary identities differ')
    summary = {r['architecture']: r for r in summary}
    markdown = []
    for arch, label in ARCHS.items():
        group = [r for r in rows if r['architecture'] == arch]
        require(len(group) == 5 and {int(r['seed']) for r in group} == SEEDS, 'Wrong seeds')
        item = summary[arch]
        require(item['configuration_slot'] == choices[arch] and int(item['n_seeds']) == 5,
                'Summary slot/count mismatch')
        displays = []
        for metric in ['f1', 'mae_all', 'fpr']:
            values = [float(r[metric]) for r in group]
            mean, std = statistics.mean(values), statistics.stdev(values)
            require(float(item[metric+'_mean']) == mean and float(item[metric+'_std']) == std,
                    'Summary differs from independent CSV recomputation')
            displays.append(f'{mean:.6f} ± {std:.6f}')
        markdown.append(f'| {label} | {choices[arch]} | 5 | ' + ' | '.join(displays) + ' |')
    for path in [ROOT/'README.md', DATA/'README.md']:
        text = path.read_text(encoding='utf-8')
        require(all(line in text for line in markdown), 'README summary mismatch: '+str(path))
    print('PASS: 25 unique completed runs; 0 failed; five architectures x five seeds; '
          'sealed metric agreement; frozen thresholds; FPR cap; five summaries and README tables.')
    print('Sample standard deviation, ddof=1. Validation-calibration only; independent test excluded.')
    for line in markdown:
        print(line)
    return rows


if __name__ == '__main__':
    validate()
