"""Measure paired normal decode requests, or profile Engram and the shared output projection."""
import argparse
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

import executor_repro as r
from executor.utils.forward_metadata import get_forward_metadata

torch = r.torch


@contextmanager
def profile_regions(main_model, samples):
    """Place event pairs around the gate (after KV projection) and the vocabulary projection."""
    engram = main_model.model.layers[1].engram
    lm_head = main_model.lm_head
    gate_forward = engram.forward
    project_kv = engram.precompute_kv
    vocab_forward = lm_head.forward
    gate_begin = None

    def start():
        event = torch.npu.Event(enable_timing=True)
        event.record()
        return event, time.perf_counter_ns()

    def finish(label, begin):
        host_us = (time.perf_counter_ns() - begin[1]) / 1000
        end = torch.npu.Event(enable_timing=True)
        end.record()
        samples.append((label, begin[0], end, host_us))

    def project(*args, **kwargs):
        nonlocal gate_begin
        values = project_kv(*args, **kwargs)
        if not get_forward_metadata().is_prefill:
            gate_begin = start()
        return values

    def gate(*args, **kwargs):
        nonlocal gate_begin
        if kwargs.get('is_prefill'):
            return gate_forward(*args, **kwargs)
        gate_begin = start() if kwargs.get('precomputed') is not None else None
        output = gate_forward(*args, **kwargs)
        assert gate_begin is not None, 'Missing Engram gate input boundary'
        finish('engram_gate', gate_begin)
        gate_begin = None
        return output

    def project_vocab(*args, **kwargs):
        if get_forward_metadata().is_prefill:
            return vocab_forward(*args, **kwargs)
        begin = start()
        output = vocab_forward(*args, **kwargs)
        x = args[0]
        finish(f'lm_head_m{x.numel() // x.shape[-1]}', begin)
        return output

    with patch.object(engram, 'forward', gate), patch.object(engram, 'precompute_kv', project), \
         patch.object(lm_head, 'forward', project_vocab):
        yield


def paired_summary(rows):
    """Resample complete A/B pairs for the interval of the median latency difference."""
    rng = random.Random(41)
    result = {}
    for metric in ('wall_ms', 'decode_step_ms'):
        deltas = [pair['compiled'][metric] - pair['matched_eager'][metric] for pair in rows]
        draws = sorted(median(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
        result[metric] = {key: median(pair[key][metric] for pair in rows) for key in ('compiled', 'matched_eager')}
        result[metric].update(delta_median_ms=median(deltas), paired_bootstrap_95_ms=[draws[49], draws[1949]])
    result['tokens_per_s'] = {arm: median(pair[arm]['tokens_per_s'] for pair in rows)
                              for arm in ('compiled', 'matched_eager')}
    return result


def benchmark(args, tmp):
    torch.manual_seed(41)
    torch.npu.set_device(0)
    recipe = r.RECIPES_DIR / 'models/deepseek_v4_1/config/deepseek_v4_1_flash_rank_8_8ep_multimodal.yaml'
    cfg = r.make_fixture(tmp, args.source, recipe, args.dspark, 'npugraph_ex')
    r.dist.init_process_group('hccl', init_method=(tmp / 'rank0').as_uri(), rank=0, world_size=1)
    report = {'mode': args.mode, 'dspark': args.dspark, 'card': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  'torch': str(torch.__version__), 'torch_npu': r.torch_npu.__version__,
                  'cann': os.environ.get('ASCEND_HOME_PATH'), 'rounds': args.rounds}
    try:
        with ExitStack() as scope:
            scope.enter_context(r.fixture_patches())
            audit = None
            if args.mode == 'profile':
                from decode_reference import matched_eager_decode
                scope.enter_context(matched_eager_decode())
            elif args.mode in ('ascendc', 'pto'):
                from tilelang_decode import ordered_decode_backend
                audit = scope.enter_context(ordered_decode_backend(args.mode))
            t0 = time.perf_counter()
            engine = r.OfflineInference(cfg)
            torch.npu.synchronize()
            report['engine_init_including_warmup_s'] = time.perf_counter() - t0
            main = engine.engine.main_worker
            compiled = main.model_compiled
            eager = main.model.main_decode
            prompt = [r.make_prompt(tmp)]
            sampling = r.SamplingParams(temperature=0, seed=9)
            reference = None
            get_valid_output = engine.get_valid_output

            def request(forward):
                nonlocal reference
                token_ids = None

                def record_output(req):
                    nonlocal token_ids
                    token_ids = get_valid_output(req)
                    return token_ids

                with patch.object(main, 'model_compiled', forward), patch.object(engine, 'get_valid_output', record_output):
                    torch.npu.synchronize()
                    begin = time.perf_counter()
                    outputs, stats, stages = engine.generate(prompt, sampling)
                    torch.npu.synchronize()
                    wall = (time.perf_counter() - begin) * 1000
                assert len(outputs) == 1 and outputs[0].finish_reason == 'length'
                assert token_ids is not None
                tokens = list(token_ids)
                expected_count = 5 if args.dspark else 3
                assert len(tokens) == expected_count and len(stages) == 2 + expected_count
                if reference is None:
                    reference = tokens
                if args.mode != 'profile':
                    assert tokens == reference, (tokens, reference)
                return {'wall_ms': wall, 'decode_step_ms': 1000 * mean(stages[2:]), 'tokens_per_s': 1000 * len(tokens) / wall,
                            'tokens': tokens, 'verification_rounds': stats['spec_num_forward_ct']}

            for _ in range(3):
                request(eager)
                if args.mode != 'profile':
                    request(compiled)
            report['tokens'] = reference
            if args.mode == 'profile':
                intervals = []
                with profile_regions(main.model, intervals):
                    rows = [request(eager) for _ in range(args.rounds)]
                traces = defaultdict(list)
                for label, begin, end, host_us in intervals:
                    traces[label].append({'host_submit_us': host_us, 'stream_elapsed_us': 1000 * begin.elapsed_time(end)})
                expected_regions = ['engram_gate', 'lm_head_m2', 'lm_head_m3'] if args.dspark else ['engram_gate', 'lm_head_m1']
                expected_calls = args.rounds * (5 if args.dspark else 3)
                for label in expected_regions:
                    assert len(traces[label]) == expected_calls, (label, len(traces[label]), expected_calls)
                report.update(requests=rows, events=traces)
                report['median_region_us'] = {name: {metric: median(s[metric] for s in values) for metric in values[0]}
                                              for name, values in traces.items()}
            else:
                if audit is not None:
                    before = audit.counts()
                    request(compiled)
                    report['audit'] = audit.report(before)
                else:
                    replay = torch.npu.NPUGraph.replay
                    with patch.object(torch.npu.NPUGraph, 'replay', autospec=True, side_effect=replay) as record:
                        request(compiled)
                    assert record.call_count == 3
                    report['device_replays_audit'] = record.call_count
                pairs = []
                rng = random.Random(41)
                orders = [n % 2 for n in range(args.rounds)]
                rng.shuffle(orders)
                for eager_first in orders:
                    arms = [('matched_eager', eager), ('compiled', compiled)]
                    if not eager_first:
                        arms.reverse()
                    pairs.append({name: request(fn) for name, fn in arms})
                report.update(pairs=pairs, summary=paired_summary(pairs))
    finally:
        r.dist.destroy_process_group()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['profile', 'ascendc', 'pto', 'npugraph_ex'])
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('MODEL_SOURCE', Path(__file__).resolve().parents[1] / 'build/model')))
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--dspark', action='store_true', help='Profile the draft and verification shapes.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error('use at least two rounds')
    if args.dspark and args.mode != 'profile':
        parser.error('--dspark is supported for profile measurements')
    with tempfile.TemporaryDirectory(prefix='decode_ab_') as tmp:
        result = benchmark(args, Path(tmp))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result.get('summary') or result['median_region_us']), flush=True)


if __name__ == '__main__':
    main()
