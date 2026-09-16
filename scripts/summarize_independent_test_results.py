"""Read-only validation of independent-test exports; no model or dataset access."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/independent_test_2026'
ARCHS = dict(lstm='LSTM', xlstm='xLSTM', cnn_lstm='CNN-LSTM',
             transformer='Transformer', convtransformer='ConvTransformer')
METRICS = ['f1', 'fpr', 'mae_all', 'mae_tp', 'rmse_all', 'rmse_tp',
           'accuracy', 'precision', 'recall']
SEEDS = {1, 2, 3, 42, 123}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def validate():
    rows = csv_rows('independent_test_25_runs.csv')
    summaries = csv_rows('architecture_summary.csv')
    comparisons = csv_rows('validation_vs_test.csv')
    provenance = json.loads((DATA / 'source_provenance.json').read_text(encoding='utf-8'))
    calibration = json.loads((ROOT / 'results/final_validation_2026/source_provenance.json').read_text(encoding='utf-8'))
    selected = json.loads((ROOT / 'results/tuning/selected_configs.json').read_text(encoding='utf-8'))
    slots = {a: selected['choices'][a]['configuration_slot'] for a in ARCHS}
    expected = {f'{a}_{slots[a]}_seed{s}' for a in ARCHS for s in SEEDS}
    require(len(rows) == 25 and {r['run_id'] for r in rows} == expected, 'Missing/duplicate/extra runs')
    require(len(provenance['runs']) == 25 and {r['run_id'] for r in provenance['runs']} == expected, 'Source identities')
    source = {r['run_id']: r for r in provenance['runs']}
    seals = {r['run_id']: r for r in calibration['runs']}
    require(set(seals) == expected, 'Calibration identities')
    require(len(summaries) == 5 and {r['architecture'] for r in summaries} == set(ARCHS), 'Summary identities')
    require(len(comparisons) == 45 and {(r['architecture'], r['metric']) for r in comparisons}
            == {(a, k) for a in ARCHS for k in METRICS}, 'Comparison identities')
    summaries = {r['architecture']: r for r in summaries}
    comparisons = {(r['architecture'], r['metric']): r for r in comparisons}
    for r in rows:
        rid = r['run_id']
        require(r['status'] == 'completed' and r['metric_split'] == 'independent_held_out_test', 'Status/split')
        require(r['configuration_slot'] == slots[r['architecture']] and
                rid == f"{r['architecture']}_{r['configuration_slot']}_seed{int(r['seed'])}", 'Run identity')
        require(source[rid]['final_seal_sha256'] == seals[rid]['seal_sha256'], 'Frozen seal hash')
        require(float(r['threshold']) == seals[rid]['calibration_metrics']['threshold'], 'Frozen threshold')
        for key, value in source[rid]['metrics'].items():
            require(math.isfinite(float(r[key])) and float(r[key]) == value, 'Source metric mismatch: '+key)
        m = source[rid]['metrics']
        require(m['tp']+m['fn'] == m['tn']+m['fp'] == 100000, 'Class counts')
        computed = dict(f1=2*m['tp']/(2*m['tp']+m['fp']+m['fn']), fpr=m['fp']/100000,
                        accuracy=(m['tp']+m['tn'])/200000, precision=m['tp']/(m['tp']+m['fp']), recall=m['tp']/100000)
        for key, value in computed.items():
            require(math.isclose(m[key], value, rel_tol=1e-14), 'Confusion-count inconsistency')
    for arch, label in ARCHS.items():
        group = [r for r in rows if r['architecture'] == arch]
        require(len(group) == 5 and {int(r['seed']) for r in group} == SEEDS, 'Seed coverage')
        item = summaries[arch]
        require(item['configuration_slot'] == slots[arch] and int(item['n_seeds']) == 5, 'Summary slot/count')
        displays = {}
        for metric in METRICS:
            values = [float(r[metric]) for r in group]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            for suffix, value in dict(mean=mean, sd=sd, min=min(values), max=max(values)).items():
                require(float(item[metric+'_'+suffix]) == value, 'Recomputed test summary differs')
            cal_values = [seals[r['run_id']]['calibration_metrics'][metric] for r in group]
            cm, cs = statistics.mean(cal_values), statistics.stdev(cal_values)
            comp = comparisons[(arch, metric)]
            require(comp['configuration_slot'] == slots[arch] and int(comp['n_seeds']) == 5, 'Comparison slot/count')
            for key, value in dict(validation_calibration_mean=cm, validation_calibration_sd=cs,
                                   independent_test_mean=mean, independent_test_sd=sd,
                                   test_minus_calibration_mean=mean-cm).items():
                require(float(comp[key]) == value, 'Recomputed comparison differs')
            displays[metric] = f'{mean:.6f} ± {sd:.6f}'
        for metrics in [METRICS[:3], METRICS[3:6], METRICS[6:]]:
            line = '| '+label+' | '+' | '.join(displays[k] for k in metrics)+' |'
            require(line in (DATA/'README.md').read_text(encoding='utf-8'), 'Results README table')
            if metrics == METRICS[:3]:
                require(line in (ROOT/'README.md').read_text(encoding='utf-8'), 'Main README table')
                print(line)
    print('PASS: 25 completed unique runs; 5 architectures x 5 seeds; source metrics exact; '
          '25 seals/thresholds match pre-test calibration; 9 metrics recomputed with ddof=1; '
          '45 separate validation/test comparisons; README tables match.')
    return rows


if __name__ == '__main__':
    validate()
