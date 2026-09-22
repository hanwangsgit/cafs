"""Recompute the paper tables from bundled measurements, without model execution.

No arguments: Python standard library only; reads reproduction/paper_runs.csv.
Optional FACE MRI arguments: additionally audit unpacked original archives.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(face, mri):
    import torch
    from cafs.acquisition import MeasurementOracle
    from cafs.data import tensor_sha256
    from cafs.experiment import initial_operator, seeds, mri_static

    records = {}
    checked_policies = 0
    for domain, directory in [('face', face), ('mri', mri)]:
        rows = [json.loads(p.read_text()) for p in directory.glob('job-*/element-*/record.json')]
        config = json.loads((ROOT / f'configs/{domain}.json').read_text())
        units = json.loads((ROOT / f'reproduction/{domain}_samples.json').read_text())
        expected = {(u['unit_id'], b) for u in units for b in config['budgets']}
        assert len(rows) == 90, (domain, 'expected 90 records', len(rows))
        assert {(r['unit']['unit_id'], r['requested_budget']) for r in rows} == expected
        initial = {b: initial_operator(domain, b, 'cpu') for b in config['budgets']}
        group_costs = next(iter(initial.values()))[0].group_costs.tolist()
        static_histories = {}
        if domain == 'mri':
            for budget, (op, costs, _) in initial.items():
                for policy in ['uniform', 'variable_density']:
                    context = MeasurementOracle(target=torch.zeros(1, *op.signal_shape),
                        initial_operator=op, event_costs=costs).policy_context(
                        policy_id=policy, selector_generator=torch.Generator(),
                        intermediate_reconstructor=lambda *_: None)
                    static_histories[budget, policy] = [list(e) for e in mri_static(context, budget).action_history]
        for row in rows:
            assert row['status'] == 'complete' and row['run_scope'] == 'final'
            assert row['source'] == {'commit': '5dd9e6b2ee288f6f9bc8a60078f6d47877808a08', 'status': ''}
            assert row['seed'] == 0
            unit = next(u for u in units if u['unit_id'] == row['unit']['unit_id'])
            assert row['unit']['source_sha256'] == unit['source_sha256']
            if domain == 'mri':
                assert all(row['unit'][k] == unit[k] for k in ['slice_index', 'target_sha256'])
            assert [p['policy_id'] for p in row['policies']] == config['policies']
            budget = row['requested_budget']
            op, costs, target = initial[budget]
            for policy in row['policies']:
                accounting = policy['acquisition']
                assert accounting['nfe'] == (accounting['forward_sample_evaluations']
                                             + accounting['backward_sample_evaluations'])
                streams = seeds(domain, unit['unit_id'], budget, policy['policy_id'])
                assert all(streams[k] == v for k, v in policy['seeds'].items())
                assert policy['seed_action_ids'] == op.selected_group_ids.tolist()
                assert policy['event_costs'] == list(costs)
                assert len(policy['action_history']) == 5
                for event, cost in zip(policy['action_history'], costs):
                    assert sum(group_costs[i] for i in event) == cost
                ids = [i for event in policy['action_history'] for i in event]
                final_op = op.add_groups(ids)
                assert final_op.m == policy['achieved_m'] == target.achieved_m
                if domain == 'mri':
                    mask_hash = tensor_sha256(final_op.line_mask)
                    if policy['policy_id'] in ['uniform', 'variable_density']:
                        assert policy['action_history'] == static_histories[budget, policy['policy_id']]
                else:
                    mask = final_op.freq_mask.contiguous()
                    header = str(mask.dtype).encode() + json.dumps(list(mask.shape), separators=(',', ':')).encode()
                    mask_hash = hashlib.sha256(header + mask.numpy().tobytes()).hexdigest()
                assert mask_hash == policy['mask_sha256']
                checked_policies += 1
        records[domain] = rows
    binding = json.loads((mri / 'binding.json').read_text())
    binding_hash = binding.pop('binding_sha256')
    assert digest(binding) == binding_hash
    package = {name: hashlib.sha256((mri / 'source' / name).read_bytes()).hexdigest()
               for name in ['mri_final.py', 'protocol.json', 'eligibility.json']}
    assert digest(package) == binding['package_sha256']
    for row in records['mri']:
        assert row['binding_sha256'] == binding_hash
        assert row['package_sha256'] == binding['package_sha256']
        assert row['config'] == binding['protocol']
        assert row['config_sha256'] == digest(row['config'])
        assert row['unit'] == binding['units'][row['array']['unit_index']]
    return {**verify_tables(records), 'checked_policy_histories': checked_policies,
            'mri_package_files': package, 'mri_binding_sha256': binding_hash}


def verify_tables(records):
    quality = []
    for expected in csv.DictReader((ROOT / 'reproduction/paper_metrics.csv').open()):
        domain, budget, policy = expected['domain'], float(expected['budget']), expected['policy']
        final = 'ddrm' if domain == 'face' else 'cm'
        values = [x['metrics'] for r in records[domain] if r['requested_budget'] == budget
                  for p in r['policies'] if p['policy_id'] == policy
                  for x in p['reconstructions'] if x['reconstructor_id'] == final]
        assert len(values) == 30
        means = {k: statistics.mean(x[k] for x in values) for k in values[0]}
        assert f"{means['psnr']:.2f}" == expected['psnr'], (domain, budget, policy, means)
        assert f"{means['ssim']:.3f}" == expected['ssim'], (domain, budget, policy, means)
        quality.append({'domain': domain, 'budget': budget, 'policy': policy, 'n': 30, **means})
    efficiency = []
    for expected in csv.DictReader((ROOT / 'reproduction/paper_efficiency.csv').open()):
        domain, policy = expected['domain'], expected['policy']
        values = [(r, p['acquisition']) for r in records[domain] for p in r['policies'] if p['policy_id'] == policy]
        nfe = {a['nfe'] for _, a in values}
        assert nfe == {int(expected['sample_nfe'])}
        peak = max(a['peak_memory_bytes'] for _, a in values) / 2**30
        times = [a['wall_seconds'] for r, a in values if r['requested_budget'] == .1
                 and r['hardware']['gpu'] == 'NVIDIA A100-SXM4-80GB']
        assert len(times) == int(expected['timing_n'])
        median = statistics.median(times)
        assert abs(peak - float(expected['peak_gib'])) <= .000500001, (domain, policy, peak)
        assert abs(median - float(expected['median_seconds'])) <= .000500001, (domain, policy, median)
        efficiency.append({'domain': domain, 'policy': policy, 'nfe': next(iter(nfe)),
                           'peak_gib': peak, 'timing_n': len(times), 'median_seconds': median})
    return {'blocks': {domain: len(rows) for domain, rows in records.items()},
            'quality': quality, 'efficiency': efficiency}


def bundled_records():
    """Read each archived case once, rejecting missing or duplicate measurements."""
    grouped = {'face': {}, 'mri': {}}
    seen = set()
    with (ROOT / 'reproduction/paper_runs.csv').open() as stream:
        for row in csv.DictReader(stream):
            domain, unit, budget = row['domain'], row['unit_id'], float(row['budget'])
            policy = row['policy']
            key = (domain, unit, budget, policy)
            assert key not in seen, ('duplicate measurement', key)
            seen.add(key)
            assert row['reconstructor'] == ('ddrm' if domain == 'face' else 'cm')
            record = grouped[domain].setdefault((unit, budget), {
                'requested_budget': budget, 'hardware': {'gpu': row['gpu']}, 'policies': []})
            assert record['hardware']['gpu'] == row['gpu']
            metrics = {k: float(row[k]) for k in ['psnr', 'ssim']}
            if domain == 'mri':
                metrics['nmse'] = float(row['nmse'])
            record['policies'].append({'policy_id': policy,
                'reconstructions': [{'reconstructor_id': row['reconstructor'], 'metrics': metrics}],
                'acquisition': {'nfe': int(row['nfe']), 'wall_seconds': float(row['wall_seconds']),
                                'peak_memory_bytes': int(row['peak_memory_bytes'])}})
    expected = set()
    for domain in grouped:
        units = json.loads((ROOT / f'reproduction/{domain}_samples.json').read_text())
        config = json.loads((ROOT / f'configs/{domain}.json').read_text())
        expected.update((domain, u['unit_id'], b, p) for u in units
                        for b in config['budgets'] for p in config['policies'])
    assert seen == expected, 'Recorded measurements do not cover the paper grid'
    return {domain: list(rows.values()) for domain, rows in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('face', type=Path, nargs='?')
    parser.add_argument('mri', type=Path, nargs='?')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Run without -O so verification assertions remain enabled')
    if (args.face is None) != (args.mri is None):
        parser.error('Provide both archive directories, or neither')
    report = verify_tables(bundled_records()) if args.face is None else verify(args.face, args.mri)
    print(f"Verified {sum(report['blocks'].values())} blocks, "
          f"{len(report['quality'])} quality rows, and {len(report['efficiency'])} efficiency rows.",
          file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
