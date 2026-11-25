"""Run a single-rank V4.1 OfflineInference request with synthetic MX weights.

Requires the matching V4.1 operators and a sourced CANN environment (see
README.md). Use --dspark to run proposal and verification. Select
--exe-mode npugraph_ex for NPU graph replay or tilelang_ascendc or
tilelang_pto to compile ordered arithmetic partitions during decode.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
from unittest.mock import patch

RECIPES_DIR = Path(__file__).resolve().parent.parent / '3rdparty' / 'cann-recipes-infer'
if not (RECIPES_DIR / 'executor' / 'offline' / 'offline_inference.py').is_file():
    raise SystemExit(
        f'Missing cann-recipes-infer submodule at {RECIPES_DIR}. '
        'From the repository root, run: git submodule update --init --recursive'
    )
sys.path.insert(0, str(RECIPES_DIR))
# These upstream directories are namespace packages. Limit their search paths
# to this checkout even if the environment also provides a package named models.
for name in ('executor', 'models', 'module'):
    spec = PathFinder.find_spec(name, [str(RECIPES_DIR)])
    if spec is None:
        raise ImportError(f'Missing upstream package {name} in {RECIPES_DIR}')
    spec.submodule_search_locations = list(spec.submodule_search_locations)
    package = module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)

import torch
import torch.distributed as dist
import torch_npu
import yaml
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from executor.core.config import InferenceConfig
from executor.core.forward_data_info import SamplingParams
from executor.core.kv_cache import cache_utils
from executor.offline.offline_inference import OfflineInference
from models.deepseek_v4_1.models.modeling_deepseek import DeepseekV41ForCausalLM, DeepseekV41Model, DeepseekV41MoE
from models.deepseek_v4_1.models.modeling_deepseek_vision import DeepseekV41VisionModel
from models.deepseek_v4_1.models.modules.attention_data import AttnMetaData
from models.deepseek_v4_1.utils.convert_config import main as convert_config
from models.deepseek_v4_1.utils.encoding_dsv4 import IMAGE_PLACEHOLDER


TILELANG_MODES = {"tilelang_ascendc": "ascendc", "tilelang_pto": "pto"}


def make_fixture(path, source, recipe, dspark, exe_mode=None):
    raw = json.loads((source / 'config.json').read_text())
    raw['image_token_id'] = 1022
    raw['text_config'].update(
        vocab_size=1024, num_hidden_layers=6, n_routed_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=512, engram_layer_ids=[1], engram_vocab_size=257,
        engram_num_embeddings=[16384], engram_compressed_vocab_size=1024,
        kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 4], candidate_source_layer_id=3,
        compress_ratios=[0, 2, 2, 1, 1, 1, 0, 0, 0], dspark_target_layer_ids=[4, 5],
        dspark_noise_token_id=1023, dspark_n_routed_experts=8, dspark_num_experts_per_tok=2,
    )
    raw['vision_config'].update(num_hidden_layers=1, min_pixels=42**2)
    (path / 'config.json').write_text(json.dumps(raw))
    convert_config(path, is_mx=True)
    vocab = {'<s>': 0, '</s>': 1, '<pad>': 2, '<unk>': 3}
    vocab.update({f'tok{i}': i for i in range(4, 1024) if i != 1022})
    vocab[IMAGE_PLACEHOLDER] = 1022
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token='<s>', eos_token='</s>', pad_token='<pad>',
        unk_token='<unk>', additional_special_tokens=[IMAGE_PLACEHOLDER],
        chat_template=(source / 'chat_template.jinja').read_text(),
    )
    assert hf.convert_tokens_to_ids(IMAGE_PLACEHOLDER) == 1022
    hf.save_pretrained(path)
    settings = yaml.safe_load(recipe.read_text())
    settings['model_config'].update(model_path=str(path), with_ckpt=False, force_eplb=True)
    if exe_mode is not None:
        settings['model_config']['exe_mode'] = 'npugraph_ex' if exe_mode in TILELANG_MODES else exe_mode
    params = settings['model_config']['custom_params']
    params.update(enable_mega_moe=False, enable_multi_streams=False, engram_tp_size=1,
                  enable_engram_offload=False, enable_ced_prefill_tail=True)
    params['kernel_config']['swiglu'] = 'custom'  # use the matching custom vendor ABI
    settings['parallel_config']['world_size'] = 1
    settings['scheduler_config'].update(batch_size=1, max_new_tokens=3, max_prefill_tokens=256)
    settings['data_config'].update(dataset='default_multimodal', input_truncated_len=256)
    if dspark:
        settings['speculative_config'] = dict(method='dspark', num_speculative_tokens=2,
                                              markov_lmhead_tp_size=1, confidence_threshold=0.0)
        settings['scheduler_config']['max_new_tokens'] = 5
    return InferenceConfig.from_dict(settings, global_rank=0, local_rank=0)


@torch.no_grad()
def initialize(model):
    for name, param in model.named_parameters():
        if param.dtype in (torch.int8, torch.uint8):
            if name.endswith('scale'):
                param.fill_(120)  # E8M0 code for 2**-7
            else:
                param.view(torch.uint8).random_(0, 256)
        elif param.dtype == torch.float8_e4m3fn:
            param.copy_(torch.randn(param.shape, device=param.device, dtype=torch.bfloat16).clamp(-4, 4))
        elif 'norm.weight' in name or name.endswith(('q_weight', 'k_weight')):
            param.fill_(1)
        elif param.is_floating_point():
            param.normal_(0, 0.01)


def single_rank_moe(self, x, topk_ids, topk_weight, shared_expert_stream, moe_events, engram_resource_events=None):
    assert self.moe_ep_size == 1 and not self.enable_mega_moe
    events = moe_events.get('shared_expert', [])
    shared = self.forward_shared_expert(x, shared_expert_stream, events) if self.n_shared_experts else None
    if self.exe_mode != 'npugraph_ex':
        return self.moe_infer_double_routing(x, topk_ids, topk_weight, shared, events, 1)

    # Every expert is local. Keep expert counts on-device through the grouped
    # FFN so decode can be captured without host-side all-to-all split sizes.
    assert self.moe_tp_size == 1 and not self.enable_multi_streams
    routing_args, quantized = self._build_moe_tp_routing_args(x, topk_ids)
    expanded, row_idx, expert_counts, scales = torch_npu.npu_moe_init_routing_v2(x, **routing_args)
    gmm_args = self._build_moe_tp_gmm_args(expanded, expert_counts, scales, quantized)
    routed = self.moe_ffn(**gmm_args)
    return torch_npu.npu_moe_finalize_routing(
        routed, skip1=shared, skip2=None, bias=None, scales=topk_weight.to(routed.dtype),
        expanded_src_to_dst_row=row_idx, export_for_source_row=None, drop_pad_mode=2,
    )


def compressor_state_block_size(model, ratio):
    # Compressor V2 ring mode requires block_size >= ratio + S - 1.
    # The same ring serves prompt chunks and fixed-width verification calls.
    cfg = model.infer_config
    max_query = max(cfg.scheduler_config.max_prefill_tokens, 1 + cfg.speculative_config.num_speculative_tokens)
    return ratio + max_query - 1


def inspect_compressor_rings(model):
    rings = {}
    for layer, entries in model.cache_entries_by_layer.items():
        for entry in entries:
            if entry.attn_type == 'RingCache':
                ratio = model.config.compress_ratios[layer]
                shape = list(model.kv_cache_by_layer[layer].state_cache.shape)
                assert shape[1] == entry.block_size == compressor_state_block_size(model, ratio), (entry.cache_name, shape)
                rings[entry.cache_name] = {'ratio': ratio, 'shape': shape}
    assert rings, 'Missing compressor state ring in the six-layer fixture'
    return rings


@contextmanager
def fixture_patches(trace=None):
    allocate = cache_utils._alloc_aligned_hbm_tensor
    original_layout = DeepseekV41Model._init_model_cache_layout

    def init_cache_layout(model):
        original_layout(model)
        for layer, entries in model.cache_entries_by_layer.items():
            for entry in entries:
                if entry.attn_type == 'RingCache':
                    entry.block_size = compressor_state_block_size(model, model.config.compress_ratios[layer])

    def allocate_cache(device, dtype, numel, shape):
        cache = allocate(device, dtype, numel, shape)
        if trace is None:
            # The warm-up null page must have deterministic packed KV and scales.
            cache[0].zero_()
        else:
            # Unwritten bytes in every physical cache page participate in the audit.
            cache.zero_()
            trace.caches.append(cache)
        return cache

    with patch.object(cache_utils, '_alloc_aligned_hbm_tensor', allocate_cache), \
         patch.object(DeepseekV41Model, '_init_model_cache_layout', init_cache_layout), \
         patch.object(AttnMetaData, 'get_state_block_size', compressor_state_block_size), \
         patch('executor.model_loader.dummy_loader.initialize_dummy_weights', initialize), \
         patch.object(DeepseekV41MoE, 'moe_infer_dispatch_combine', single_rank_moe):
        yield


def make_prompt(tmp):
    torch.manual_seed(9)
    pixels = torch.randint(0, 256, (42, 42, 3), dtype=torch.uint8)
    sample = tmp / 'input.png'
    Image.fromarray(pixels.numpy()).save(sample)
    return [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': str(sample)}},
        {'type': 'text', 'text': ' '.join(f'tok{n}' for n in range(4, 143))},
    ]}]


def run(args, tmp):
    torch.manual_seed(41)
    torch.npu.set_device(0)
    from tilelang_decode import ordered_decode_backend
    from decode_reference import DecodeReference, matched_eager_decode

    target = TILELANG_MODES.get(args.exe_mode)
    trace = None
    if args.write_reference or args.compare_reference:
        trace = DecodeReference(args.write_reference, args.compare_reference, args.dspark)
    fixture_mode = 'npugraph_ex' if args.write_reference else args.exe_mode
    cfg = make_fixture(tmp, args.source, args.recipe, args.dspark, fixture_mode)
    graph_mode = cfg.model_config.exe_mode == 'npugraph_ex'
    dist.init_process_group('hccl', init_method=(tmp / 'rank0').as_uri(), rank=0, world_size=1)
    calls = Counter()
    head = DeepseekV41ForCausalLM.forward_lm_head
    vision = DeepseekV41VisionModel.forward
    diagnostics = {}

    def check_logits(logits, is_prefill):
        assert logits.device.type == 'npu' and logits.shape[-1] == 1024
        assert logits.isfinite().all().item(), 'nonfinite logits'
        assert logits.abs().max().item() > 0, 'all-zero logits'
        calls['prefill' if is_prefill else 'decode'] += 1

    def checked_logits(model, outputs, kv_len, is_prefill=True, attn_metadata=None):
        logits = head(model, outputs, kv_len, is_prefill, attn_metadata)
        if not graph_mode or not torch.compiler.is_compiling():
            check_logits(logits, is_prefill)
        return logits

    def check_compiled(compiled, name, logits_index, check_replay):
        assert callable(compiled), f'Missing compiled {name} decode'
        replay = torch.npu.NPUGraph.replay
        replay_key = f'{name}_graph_replays'

        def counted_replay(graph):
            replay(graph)
            calls[replay_key] += 1

        def run_compiled(**inputs):
            # The backend can fall back to eager. Count device replays for the
            # submitted request separately from compile/capture during warm-up.
            if check_replay:
                previous = calls[replay_key]
                with patch.object(torch.npu.NPUGraph, 'replay', counted_replay):
                    output = compiled(**inputs)
                assert calls[replay_key] > previous, f'{name} decode did not replay an NPU graph'
            else:
                output = compiled(**inputs)
            check_logits(output[logits_index], False)
            return output

        return run_compiled

    def checked_vision(model, *args, **kwargs):
        result = vision(model, *args, **kwargs)
        if any(value is not None and value.numel() for value in result):
            calls['vision'] += 1
        return result

    try:
        with ExitStack() as stack:
            stack.enter_context(fixture_patches(trace))
            stack.enter_context(patch.object(DeepseekV41ForCausalLM, 'forward_lm_head', checked_logits))
            stack.enter_context(patch.object(DeepseekV41VisionModel, 'forward', checked_vision))
            audit = stack.enter_context(ordered_decode_backend(target)) if target else None
            if args.write_reference:
                stack.enter_context(matched_eager_decode())
            engine = OfflineInference(cfg)
            diagnostics['compressor_rings'] = inspect_compressor_rings(engine.engine.main_worker.model.model)
            check_replay = graph_mode and not (target or args.write_reference)
            if graph_mode and not args.write_reference:
                main = engine.engine.main_worker
                main.model_compiled = check_compiled(main.model_compiled, 'main', 0, check_replay)
                if args.dspark:
                    draft = engine.engine.draft_worker.model
                    draft.compiled_forward_spec_decode = check_compiled(
                        draft.compiled_forward_spec_decode, 'draft', 1, check_replay)
            calls.clear()  # count only the submitted request
            if trace:
                stack.enter_context(trace.observe(engine))
            before = audit.counts() if audit else {}
            outputs, stats, _ = engine.generate(
                [make_prompt(tmp)], SamplingParams(temperature=0, seed=9 if trace else None))
            torch.npu.synchronize()
            assert len(outputs) == 1 and calls['vision'] > 0 and calls['prefill'] == 1 and calls['decode'] >= 1
            spec_steps = stats['spec_num_forward_ct'] if args.dspark else []
            assert not args.dspark or sum(spec_steps) > 0
            if check_replay:
                assert calls['main_graph_replays'] > 0 and (not args.dspark or calls['draft_graph_replays'] > 0)
            if audit:
                reports = audit.report(before)
                for role in (('main', 'draft') if args.dspark else ('main',)):
                    assert role in reports, f'Missing TileLang report for {role} decode'
                diagnostics['tilelang'] = reports
    finally:
        dist.destroy_process_group()
    if trace:
        diagnostics['reference'] = trace.finish(outputs[0], spec_steps)
    print(json.dumps({'result': 'pass', 'exe_mode': args.exe_mode or cfg.model_config.exe_mode, 'calls': calls,
                      'finish': outputs[0].finish_reason, 'output': outputs[0].output_text,
                      'speculative_forwards': spec_steps, **diagnostics}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = Path(__file__).resolve().parent.parent / 'build' / 'model'
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('MODEL_SOURCE', default_source)),
                        help='Directory containing the official config.json and chat_template.jinja.')
    model_config_dir = RECIPES_DIR / 'models' / 'deepseek_v4_1' / 'config'
    parser.add_argument('--recipe', type=Path,
                        default=model_config_dir.joinpath('deepseek_v4_1_flash_rank_8_8ep_multimodal.yaml'))
    parser.add_argument('--dspark', action='store_true')
    parser.add_argument('--exe-mode', choices=('eager', 'npugraph_ex', *TILELANG_MODES), help='Override the execution mode in the recipe.')
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--write-reference', type=Path, metavar='DIR', help='Save matched eager outputs and cache states.')
    reference.add_argument('--compare-reference', type=Path, metavar='DIR', help='Compare each model step against a saved reference.')
    args = parser.parse_args()
    if args.write_reference and args.exe_mode != 'eager':
        parser.error('--write-reference requires --exe-mode eager')
    if args.compare_reference and args.exe_mode not in TILELANG_MODES:
        parser.error('--compare-reference requires a TileLang execution mode')
    with tempfile.TemporaryDirectory(prefix='dsv41_fixture_') as work:
        run(args, Path(work))
