"""Run the recipe's fixed-shape decode entry points with the ordered TileLang backend."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from unittest.mock import patch

import torch

ARITHMETIC = frozenset({"add", "sub", "subtract", "mul", "multiply", "div", "divide", "truediv",
                        "true_divide", "relu", "sum", "mean", "amax", "softmax", "matmul", "mm",
                        "rsqrt", "sqrt", "exp", "square", "pow"})


def arithmetic_part(part):
    return any(ARITHMETIC.intersection(target.split(".")) for target in part.targets)


class DecodeReports:
    def __init__(self, target):
        from torch_tilelang import GraphConfig, make_backend

        self.backend = make_backend(GraphConfig(
            target=target, partition_policy="ordered", strict=True, fusion="off",
        ))
        self.graphs = defaultdict(list)
        self.backend_pair = {"ascendc": ("ascend", "tvm_ffi"), "pto": ("pto", "pto")}[target]

    def compile(self, model_forward, infer_config, **kwargs):
        role = {"main_decode": "main", "forward_spec_decode_graph": "draft"}[model_forward.__name__]

        def capture(gm, example_inputs):
            runner = self.backend(gm, example_inputs)
            self.graphs[role].append(runner.graph_report)
            return runner

        return torch.compile(model_forward, backend=capture, dynamic=False, fullgraph=True)

    def counts(self):
        counts = {}
        for role, graphs in self.graphs.items():
            parts = [p for graph in graphs for p in graph.partitions]
            counts[role] = sum(p.compiled_calls for p in parts)
            counts[role + "_arithmetic"] = sum(p.compiled_calls for p in parts if arithmetic_part(p))
        return counts

    def report(self, before):
        result = {}
        now = self.counts()
        for role, graphs in self.graphs.items():
            assert all(graph.fallback_reason is None for graph in graphs)
            parts = [part for graph in graphs for part in graph.partitions]
            assert parts and all(part.compiled and part.kernel_count > 0 and
                                 part.fallback_calls == 0 and (part.target_backend, part.execution_backend)
                                 == self.backend_pair for part in parts), (role, parts)
            arithmetic_calls = now[role + "_arithmetic"] - before.get(role + "_arithmetic", 0)
            assert arithmetic_calls > 0, f"No TileLang arithmetic during {role} decode"
            native = [node for graph in graphs for node in graph.native_nodes]
            result[role] = {
                "graphs": len(graphs), "partitions": len(parts), "kernels": sum(p.kernel_count for p in parts),
                "compiled_calls": now[role] - before.get(role, 0), "arithmetic_calls": arithmetic_calls,
                "fallback_calls": sum(p.fallback_calls for p in parts),
                "native_nodes": len(native), "native_reasons": dict(Counter(n.reason for n in native)),
                "selected_targets": dict(Counter(t for p in parts for t in p.targets)),
                "target_backend": parts[0].target_backend, "execution_backend": parts[0].execution_backend,
            }
        return result


@contextmanager
def ordered_decode_backend(target):
    import torchair
    from executor.utils import graph_utils

    reports = DecodeReports(target)
    torchair.patch_for_hcom()
    with torch._dynamo.config.patch(inline_inbuilt_nn_modules=False), \
         patch.object(graph_utils, "compile_model_forward", reports.compile):
        yield reports
