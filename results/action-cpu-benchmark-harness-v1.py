"""Offline CPU latency benchmark of frozen task-count policies; no tuning."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, resource, statistics, time
from importlib.metadata import version
from pathlib import Path
from .action_improvement import ROOT, configurations, config_hash, positive
from .action_decomposition import predict_stage
from .consistency import negative_questions


def percentile(values, q):
    if not values or not 0 < q <= 1:
        raise ValueError('Need measurements and quantile in (0,1]')
    return sorted(values)[math.ceil(q * len(values)) - 1]


def select_sample(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault(row['pair_id'], []).append(row)
    groups = {}
    for key, pair in pairs.items():
        if len(pair) != 2 or {r['language'] for r in pair} != {'en', 'fr'}:
            raise ValueError('Need complete translation pairs')
        strata = {(r['risk_stratum'], r['expected']) for r in pair}
        if len(strata) != 1:
            raise ValueError('Inconsistent pair strata')
        groups.setdefault(next(iter(strata)), []).append(key)
    selected = []
    for risk in ('routine', 'suspicious'):
        for label in (False, True):
            keys = groups.get((risk, label), [])
            if len(keys) < 2:
                raise ValueError('Need two pairs per risk/label stratum')
            chosen = sorted(keys, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:2]
            selected.extend(r for key in chosen for r in sorted(pairs[key], key=lambda r: r['language']))
    return selected


def portable_identity(value):
    """Preserve revisions/content metadata while ignoring host-local cache paths."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key in ('device', 'dtype', 'amp_enabled', 'mps_amp_min_rows'):
                continue
            if key in ('snapshot', 'resolved_snapshot', 'weights_blob', 'resolved', 'path'):
                result[key] = Path(child).name
            else:
                result[key] = portable_identity(child)
        return result
    if isinstance(value, list):
        return [portable_identity(v) for v in value]
    return value


def run(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['LAYA_CPU_AMP'] = ''
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    examples = select_sample(rows)
    report = dict(version='action-cpu-latency-v1', backend=args.backend, threads=args.threads,
        interop_threads=1, repeats=args.repeats, device='cpu', precision='torch.float32',
        host=dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                  logical_cpus=os.cpu_count(), torch_build=torch.__config__.show()),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_pair_ids=list(dict.fromkeys(r['pair_id'] for r in examples)),
        timing_note='Sequential batch-one message requests. Positive and actual opposition passes timed separately; warmup excluded. Includes tokenization/schema preparation. No HTTP or container overhead.', predictions=[])
    started = time.perf_counter()
    if args.backend == 'gliner':
        from .action_policy import TaskCountPolicy, validate_policy
        from .backends import create_backend
        from .action_gliner import checkpoint_identity
        frozen = json.loads(args.policy.read_text()); validate_policy(frozen)
        backend = create_backend(frozen['model']['model'], device='cpu', revision=frozen['model']['revision'])
        actual = checkpoint_identity(backend)
        if portable_identity(actual) != portable_identity(frozen['model']['checkpoints']):
            raise ValueError('Frozen GLiNER checkpoint mismatch')
        if any(version(name) != expected for name, expected in frozen['packages'].items()):
            raise ValueError('Frozen GLiNER package mismatch')
        policy = TaskCountPolicy(args.policy, device='cpu', backend=backend)
        report.update(policy=frozen, checkpoint_identity=actual, packages=frozen['packages'])
        def predict(row):
            return policy.predict(row['state'], check_opposition=True)
        models = [backend.model]
    else:
        from .backends import LayaBackend
        from .schema_sweep import model_identity
        backend = LayaBackend(device='cpu')
        cfg = configurations()['balanced']
        frozen = json.loads(args.laya_selection.read_text())
        if config_hash(cfg) != frozen['comparator_hashes']['balanced']:
            raise ValueError('Balanced Laya schema changed')
        def predict(row):
            answer = positive(backend, row['state'], cfg)
            schema = negative_questions(answer['answers'], questions=cfg['questions'])
            negative, elapsed = predict_stage(backend, row['state'], schema)
            answer.update(negative=negative, negative_questions=schema, negative_ms=elapsed)
            return answer
        # Router loads lazily; load the evaluated checkpoints without scored inference.
        for name in frozen['development_model']['checkpoints']:
            backend.router.load(name)
        actual = model_identity(backend)
        if portable_identity(actual) != portable_identity(frozen['development_model']['checkpoints']):
            raise ValueError('Frozen Laya checkpoint mismatch')
        expected = frozen['development_packages']
        if any(version(name) != expected[name] for name in expected):
            raise ValueError('Frozen Laya package mismatch')
        report.update(config=cfg, checkpoint_identity=actual, packages=expected)
        models = [backend.router.load(name).model for name in backend.router.loaded]
    report['cold_load_ms'] = (time.perf_counter() - started) * 1000
    report['cold_load_note'] = 'Checkpoint construction plus identity/package validation; both Laya routed checkpoints loaded.'
    for model in models:
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('Expected CPU FP32 parameters')
    for language in ('en', 'fr'):
        for _ in range(2):
            predict(next(r for r in examples if r['language'] == language))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        ordered = examples if repeat % 2 == 0 else list(reversed(examples))
        for example in ordered:
            value = predict(example)
            row = dict(id=example['id'], pair_id=example['pair_id'], language=example['language'],
                risk_stratum=example['risk_stratum'], expected=example['expected'], repeat=repeat,
                input_chars=len(json.dumps(example['state'], ensure_ascii=False)), result=value)
            report['predictions'].append(row)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'{args.backend} CPU {args.threads} threads repeat {repeat+1}/{args.repeats}', flush=True)
    def summary(measurements):
        positive_ms = [r['result']['positive_ms'] for r in measurements]
        total_ms = [r['result']['positive_ms'] + r['result']['negative_ms'] for r in measurements]
        def stats(values):
            return dict(p50=statistics.median(values), p95=percentile(values, .95), mean=statistics.mean(values), n=len(values))
        return dict(positive_ms=stats(positive_ms), including_opposition_ms=stats(total_ms))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['process_peak_rss_bytes'] = peak_rss if platform.system() == 'Darwin' else peak_rss * 1024
    report['memory_note'] = 'Peak resident memory of the whole benchmark process, not model-only memory.'
    report['summary'] = dict(overall=summary(report['predictions']),
        per_language={lang:summary([r for r in report['predictions'] if r['language']==lang]) for lang in ('en','fr')})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('gliner','laya'), required=True)
    parser.add_argument('--threads', type=int, choices=(2,4), default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/action_mixed_eval_v2.jsonl')
    parser.add_argument('--policy', type=Path, default=ROOT/'schemas/task-count-gliner-binary-v2.json')
    parser.add_argument('--laya-selection', type=Path, default=ROOT/'schemas/laya-cpu-benchmark-lock-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.repeats < 1: parser.error('--repeats must be positive')
    run(args)

if __name__ == '__main__': main()
