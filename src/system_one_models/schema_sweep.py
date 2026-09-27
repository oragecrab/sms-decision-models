"""Controlled local Laya wording/order sweep and request-grouping comparison."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

from .backends import LayaBackend
from .benchmark import load_dataset, score_answer, summarize_languages, summarize_pairs
from .classifier import _check_token_budget
from .question_schemas import load_question_schema, schema_hash, validate_questions
from .questions import QUESTIONS

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def decision(answer: dict):
    return answer['choice'] if 'choice' in answer else answer['noul'] >= .5


def distributions(answer: dict) -> dict:
    if 'choice' in answer:
        return answer['probabilities']
    return {'true': answer['noul'], 'false': 1 - answer['noul']}


def build_candidates(catalog: dict, rubric: dict) -> list[dict]:
    """Cross authored wording with two description styles; test order independently."""
    candidates, seen = [], set()
    def add(key, name, kind, spec):
        full = deepcopy(QUESTIONS)
        full[key] = spec
        validate_questions(full)
        signature = schema_hash({key: spec})
        if (key, signature) in seen:
            return
        seen.add((key, signature))
        candidates.append({'id': f'{key}--{name}', 'question_id': key,
                           'family': name, 'kind': kind, 'sha256': signature,
                           'definition': deepcopy(spec)})
    for key, original in QUESTIONS.items():
        add(key, 'baseline', 'baseline', original)
        add(key, 'rubric', 'combined_rubric', rubric[key])
        compact = catalog['compact_criteria'][key]
        # Compact descriptions retain baseline option order and labels.
        compact = {label: compact[label] for label in original['criteria']}
        spec = deepcopy(original)
        spec['criteria'] = compact
        add(key, 'baseline-compact', 'description', spec)
        for family, instructions in catalog['instructions'][key].items():
            for style in ('original', 'compact'):
                spec = deepcopy(original)
                spec['instructions'] = instructions
                if style == 'compact':
                    spec['criteria'] = compact
                add(key, f'{family}-{style}', 'wording_description', spec)
        if original['type'] == 'choice':
            labels = list(original['criteria'])
            if len(labels) == 3:
                orders = list(itertools.permutations(labels))
            else:
                orders = [list(reversed(labels)), labels[1:] + labels[:1],
                          labels[len(labels)//2:] + labels[:len(labels)//2]]
                for seed in (11, 29, 47, 83):
                    order = labels.copy()
                    random.Random(seed).shuffle(order)
                    orders.append(order)
            for index, order in enumerate(orders):
                spec = deepcopy(original)
                spec['criteria'] = {label: original['criteria'][label] for label in order}
                add(key, f'order-{index:02d}', 'option_order', spec)
        else:
            # Noul renders false then true irrespective of dict insertion order.
            for name, true, false in [('swap', 'B', 'A'), ('xy', 'X', 'Y'), ('yesno', 'Yes', 'No')]:
                spec = deepcopy(original)
                spec['labels'] = {'true': true, 'false': false}
                add(key, f'labels-{name}', 'boolean_labels', spec)
    return candidates


def preflight(backend, examples: list[dict], questions: dict) -> None:
    for example in examples:
        route = backend.router.route(example['state'], questions)
        full_route = backend.router.route(example['state'], QUESTIONS)
        if route['model'] != full_route['model']:
            raise ValueError('Subset schema changes checkpoint routing')
        agent = backend.router.load(route['model'])
        context = SimpleNamespace(agent=agent, states=[example['state']], questions=questions)
        _check_token_budget(context)


def synchronize(backend) -> None:
    import torch
    devices = {str(backend.router.load(name).device) for name in backend.router.loaded}
    if any(device.startswith('cuda') for device in devices):
        torch.cuda.synchronize()
    if 'mps' in devices:
        torch.mps.synchronize()


def timed_prediction(backend, state: dict, questions: dict) -> tuple[dict, float, dict]:
    synchronize(backend)
    started = time.perf_counter()
    answer = backend.predict(state, questions=questions)
    synchronize(backend)
    elapsed = (time.perf_counter() - started) * 1000
    return answer, elapsed, deepcopy(backend.last_routing)


def warmup(backend, examples: list[dict], questions: dict) -> None:
    languages = set()
    for example in examples:
        if example.get('language') not in languages:
            backend.predict(example['state'], questions=questions)
            languages.add(example.get('language'))
    synchronize(backend)


def summarize_candidate(rows: list[dict], key: str, baseline_rows: list[dict] | None) -> dict:
    summary = {'correct': sum(row['correct'] for row in rows), 'total': len(rows)}
    summary['accuracy'] = summary['correct'] / summary['total']
    summary['per_language'] = {}
    for language in sorted({row['language'] for row in rows}):
        group = [row for row in rows if row['language'] == language]
        summary['per_language'][language] = {'correct': sum(row['correct'] for row in group),
                                            'total': len(group), 'accuracy': statistics.mean(row['correct'] for row in group)}
    times = sorted(row['inference_ms'] for row in rows)
    summary['latency_ms'] = {'p50': statistics.median(times), 'p95': times[math.ceil(.95*len(times))-1]}
    if baseline_rows is not None:
        baseline = {row['id']: row for row in baseline_rows}
        summary.update({
            'answers_changed': sum(decision(row['answer']) != decision(baseline[row['id']]['answer']) for row in rows),
            'errors_fixed': sum(row['correct'] and not baseline[row['id']]['correct'] for row in rows),
            'correct_broken': sum(not row['correct'] and baseline[row['id']]['correct'] for row in rows),
        })
    return summary


def run_candidate(backend, examples: list[dict], candidate: dict) -> dict:
    key = candidate['question_id']
    questions = {key: candidate['definition']}
    try:
        preflight(backend, examples, questions)
    except ValueError as error:
        return {**candidate, 'status': 'rejected', 'reason': str(error), 'predictions': []}
    warmup(backend, examples, questions)
    rows = []
    for example in examples:
        answer, elapsed, routing = timed_prediction(backend, example['state'], questions)
        # Validate returned probabilities and polarity using the selected definition.
        score = score_answer(answer, {key: example['expected'][key]}, questions=questions)
        rows.append({'id': example['id'], 'language': example.get('language', 'unspecified'),
                     'pair_id': example.get('pair_id'), 'expected': example['expected'][key],
                     'answer': answer[key], 'correct': score['matches'][key],
                     'inference_ms': elapsed, 'routing': routing})
    return {**candidate, 'status': 'complete', 'predictions': rows}


def grouped_comparison(backend, examples: list[dict], questions: dict) -> dict:
    """Alternate execution order; both policies use identical input and schemas."""
    preflight(backend, examples, questions)
    warmup(backend, examples, questions)
    for key, spec in questions.items():
        warmup(backend, examples, {key: spec})
    rows, full_rows, single_rows = [], [], []
    for index, example in enumerate(examples):
        result = {}
        for policy in (('all_at_once', 'one_by_one') if index % 2 == 0 else ('one_by_one', 'all_at_once')):
            if policy == 'all_at_once':
                answer, elapsed, routing = timed_prediction(backend, example['state'], questions)
                result[policy] = {'answers': answer, 'inference_ms': elapsed, 'routing': routing}
            else:
                answer, times, routes = {}, {}, {}
                # Time the whole sequence, including dispatch overhead.
                synchronize(backend)
                started = time.perf_counter()
                for key, spec in questions.items():
                    partial, elapsed, route = timed_prediction(backend, example['state'], {key: spec})
                    answer.update(partial)
                    times[key] = elapsed
                    routes[key] = route
                synchronize(backend)
                result[policy] = {'answers': answer, 'inference_ms': (time.perf_counter()-started)*1000,
                                  'question_inference_ms': times, 'routing': routes}
        changed, deltas, route_matches = {}, {}, {}
        for key in questions:
            a = result['all_at_once']['answers'][key]
            b = result['one_by_one']['answers'][key]
            changed[key] = decision(a) != decision(b)
            da, db = distributions(a), distributions(b)
            deltas[key] = max(abs(da[label]-db[label]) for label in da)
            route_matches[key] = result['all_at_once']['routing']['model'] == result['one_by_one']['routing'][key]['model']
        rows.append({'id': example['id'], 'language': example.get('language', 'unspecified'),
                     'answers_changed': changed, 'max_probability_delta': deltas, 'routing_matches': route_matches, **result})
        for policy, target in [('all_at_once', full_rows), ('one_by_one', single_rows)]:
            target.append({**example, 'answers': result[policy]['answers'],
                           **score_answer(result[policy]['answers'], example['expected'], questions=questions)})
    summary = {'changed_labels': sum(sum(row['answers_changed'].values()) for row in rows),
               'total_labels': len(rows)*len(questions),
               'max_probability_delta': max(max(row['max_probability_delta'].values()) for row in rows),
               'routing_mismatches': sum(sum(not value for value in row['routing_matches'].values()) for row in rows),
               'per_question_changed': {key: sum(row['answers_changed'][key] for row in rows) for key in questions},
               'policies': {}}
    for policy, scored in [('all_at_once', full_rows), ('one_by_one', single_rows)]:
        times = sorted(row[policy]['inference_ms'] for row in rows)
        summary['policies'][policy] = {'per_language': summarize_languages(scored, questions=questions),
                                     'translation_pairs': summarize_pairs(scored, questions=questions),
                                     'correct_labels': sum(sum(row['matches'].values()) for row in scored),
                                     'exact_messages': sum(row['exact_match'] for row in scored),
                                     'latency_ms': {'p50': statistics.median(times), 'p95': times[math.ceil(.95*len(times))-1]}}
    return {'questions': questions, 'schema_sha256': schema_hash(questions), 'summary': summary, 'predictions': rows}


def model_identity(backend) -> dict:
    from huggingface_hub import hf_hub_download
    result = {}
    for name in backend.router.loaded:
        agent = backend.router.load(name)
        repo, subfolder = backend.router.models[name]
        prefix = f'{subfolder}/' if subfolder else ''
        config_path = Path(hf_hub_download(repo, prefix + 'rl_agent_config.json', local_files_only=True))
        folder = config_path.parent
        snapshot = folder.parent if subfolder else folder
        weights = folder / 'model.safetensors'
        result[name] = {'repo': repo, 'subfolder': subfolder, 'snapshot': str(snapshot),
                        'weights_blob': str(weights.resolve()), 'weights_bytes': weights.stat().st_size,
                        'device': str(agent.device), 'dtype': str(agent.dtype), 'amp_enabled': agent.amp_enabled,
                        'mps_amp_min_rows': agent.mps_amp_min_rows,
                        'temperature_raw': agent.temperature_raw, 'temperature': agent.temperature,
                        'temperature_by_options_raw': agent.temperature_by_options_raw,
                        'temperature_by_options': agent.temperature_by_options,
                        'config': agent.cfg}
    return result


def run_sweep(*, dataset: Path, output: Path, device: str = 'cpu', threads: int = 2) -> dict:
    import torch
    torch.set_num_threads(threads)
    examples = load_dataset(dataset)
    rubric, _ = load_question_schema(ROOT / 'schemas/rubric-v1.json')
    catalog = json.loads((ROOT / 'schemas/wording-catalog-v1.json').read_text())
    candidates = build_candidates(catalog, rubric)
    print(f'{len(candidates)} unique question definitions; {len(examples)} messages; English questions only', flush=True)
    backend = LayaBackend(device=device)
    warmup(backend, examples, QUESTIONS)
    packages = {name: version(name) for name in ('laya', 'torch', 'transformers', 'numpy')}
    identity = {'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'model': model_identity(backend), 'packages': packages, 'torch_threads': torch.get_num_threads(),
                'candidate_hashes': [candidate['sha256'] for candidate in candidates],
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = output / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity_sha256'] != identity_hash:
            raise ValueError('Existing sweep has a different dataset/model/catalog/runtime; choose a new output directory')
    else:
        write_json(manifest, {'created_at': datetime.now(timezone.utc).isoformat(), 'identity_sha256': identity_hash,
                             'identity': identity, 'dataset': str(dataset.resolve()), 'candidates': candidates})
    side_path = output / 'grouping-baseline.json'
    if not side_path.exists():
        print('Side experiment: baseline all at once versus one by one', flush=True)
        write_json(side_path, grouped_comparison(backend, examples, deepcopy(QUESTIONS)))
    side = json.loads(side_path.read_text())
    print(f"Baseline grouping changes {side['summary']['changed_labels']}/{side['summary']['total_labels']} labels; max P delta {side['summary']['max_probability_delta']:.6g}", flush=True)
    results, baseline = [], {}
    for index, candidate in enumerate(candidates, 1):
        path = output / 'candidates' / (candidate['id'] + '.json')
        if path.exists():
            result = json.loads(path.read_text())
            if result['sha256'] != candidate['sha256']:
                raise ValueError('Candidate checkpoint hash mismatch')
        else:
            result = run_candidate(backend, examples, candidate)
            if result['status'] == 'complete':
                result['summary'] = summarize_candidate(result['predictions'], candidate['question_id'], baseline.get(candidate['question_id']))
            write_json(path, result)
        results.append(result)
        if candidate['kind'] == 'baseline':
            if result['status'] != 'complete':
                raise ValueError('Baseline could not be evaluated')
            baseline[candidate['question_id']] = result['predictions']
        status = result['status']
        details = str(result.get('summary', {}).get('correct', '')) + '/' + str(len(examples)) if status == 'complete' else result['reason']
        print(f'[{index}/{len(candidates)}] {candidate["id"]}: {status} {details}', flush=True)
    rankings = {}
    for key in QUESTIONS:
        group = [result for result in results if result['question_id'] == key and result['status'] == 'complete']
        group.sort(key=lambda result: (-result['summary']['correct'],
                    -min(value['accuracy'] for value in result['summary']['per_language'].values()),
                    result['kind'] != 'baseline', result['id']))
        rankings[key] = [{k: result[k] for k in ('id', 'kind', 'sha256', 'summary')} for result in group]
    selected = deepcopy(QUESTIONS)
    selected_ids = {}
    for key in QUESTIONS:
        winner = next(result for result in results if result['id'] == rankings[key][0]['id'])
        selected[key] = winner['definition']
        selected_ids[key] = winner['id']
    write_json(output / 'selected-balanced-v1.json', {'version': 'sweep-selected-balanced-v1', 'questions': selected})
    selected_path = output / 'grouping-selected.json'
    if not selected_path.exists():
        print('Verifying selected full schema and side comparison', flush=True)
        write_json(selected_path, grouped_comparison(backend, examples, selected))
    final = json.loads(selected_path.read_text())
    # The grouping/sweep backend is no longer needed; avoid duplicate model residency.
    del backend
    import gc
    gc.collect()
    from .consistency import evaluate_consistency
    opposition = {}
    for name, schema_path in [('baseline', None), ('selected', output / 'selected-balanced-v1.json')]:
        target = output / f'opposition-{name}.json'
        if not target.exists():
            print(f'Independent opposition check: {name}', flush=True)
            evaluate_consistency(['laya'], dataset=dataset, output=target, device=device, question_schema=schema_path)
        cached = json.loads(target.read_text())
        expected_questions, _ = load_question_schema(schema_path)
        cached_model = cached['runs'][0]['model']
        if (cached.get('dataset_sha256') != identity['dataset_sha256']
                or cached.get('question_schema', {}).get('sha256') != schema_hash(expected_questions)
                or cached.get('mode') != 'separate_negative_confirmation'):
            raise ValueError('Cached opposition report has incompatible dataset/schema/mode')
        if 'checkpoints' in cached_model and (
                cached_model['checkpoints'] != identity['model']
                or cached_model.get('torch_threads') != identity['torch_threads']
                or any(cached.get('packages', {}).get(key) != value for key,value in identity['packages'].items())):
            raise ValueError('Cached opposition report has incompatible model/runtime')
        opposition[name] = cached['runs'][0]['summary']
    report = {'identity_sha256': identity_hash, 'identity': identity, 'candidate_count': len(candidates),
              'completed': sum(result['status'] == 'complete' for result in results),
              'rejected': [{k: result[k] for k in ('id', 'reason')} for result in results if result['status'] == 'rejected'],
              'rankings': rankings, 'selected_candidates': selected_ids,
              'selected_schema_sha256': schema_hash(selected), 'baseline_grouping': side['summary'],
              'selected_grouping': final['summary'], 'opposition': opposition,
              'selection_note': 'Selected and measured on the same synthetic development examples, not held-out performance.'}
    write_json(output / 'summary.json', report)
    print(json.dumps({'completed': report['completed'], 'rejected': len(report['rejected']),
                      'baseline': side['summary']['policies']['all_at_once']['per_language'],
                      'selected': final['summary']['policies']['all_at_once']['per_language']}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/base_eval_v4.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run_sweep(dataset=args.dataset, output=args.output, device=args.device, threads=args.threads)


if __name__ == '__main__':
    main()
