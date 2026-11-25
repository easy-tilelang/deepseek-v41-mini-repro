"""Compare native model calls and their persistent cache writes across fresh runs."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import torch


@contextmanager
def matched_eager_decode():
    from executor.utils import graph_utils

    # Keep the graph recipe's exact main/draft entry points and single-rank routing.
    def eager(model_forward, infer_config, **kwargs):
        return model_forward

    with patch.object(graph_utils, "compile_model_forward", eager):
        yield


class DecodeReference:
    def __init__(self, write_dir, compare_dir, dspark):
        self.path = (write_dir or compare_dir) / ("dspark.pt" if dspark else "normal.pt")
        self.dspark = dspark
        self.writing = write_dir is not None
        self.reference = None if self.writing else torch.load(self.path, map_location="cpu", weights_only=True)
        if dspark and self.reference is not None:
            drafts = [step for step in self.reference["steps"] if step["role"] == "draft"]
            if not drafts or any(not {"spec_tokens", "confidence"}.issubset(step) for step in drafts):
                raise ValueError(f"{self.path} lacks draft proposals and confidence. Regenerate with "
                                 f"--dspark --exe-mode eager --write-reference {compare_dir}")
        self.caches = []
        self.steps = []
        self.tokens = None
        self.max_logit_error = 0.0
        self.draft_steps = 0
        self.max_confidence_error = 0.0

    @contextmanager
    def observe(self, engine):
        from executor.core.model_worker.model_worker import ModelWorker

        execute = ModelWorker.execute_model_call
        get_valid_output = engine.get_valid_output
        # Warm-up exercises dummy slots (including page zero). Start the real
        # request from identical bytes while retaining every device address.
        for cache in self.caches:
            cache.zero_()
        torch.npu.synchronize()

        def record(worker, model_call, *args, **kwargs):
            output, duration = execute(worker, model_call, *args, **kwargs)
            if worker is engine.engine.main_worker:
                self._record_step("main", output[0])
            else:
                self._record_step("draft", output["logits"], output["spec_tokens"], output["confidence"])
            return output, duration

        def record_tokens(request):
            tokens = get_valid_output(request)
            self.tokens = list(tokens)
            return tokens

        with patch.object(ModelWorker, "execute_model_call", record), \
             patch.object(engine, "get_valid_output", record_tokens):
            yield

    def _record_step(self, role, logits, spec_tokens=None, confidence=None):
        logits = logits.detach().cpu()
        caches = [tensor.detach().view(torch.uint8).cpu() for tensor in self.caches]
        proposal = {}
        if role == "draft":
            self.draft_steps += 1
            proposal = {"spec_tokens": spec_tokens.detach().cpu(), "confidence": confidence.detach().cpu()}
        step = len(self.steps)
        if self.writing:
            self.steps.append({"role": role, "logits": logits, "caches": caches, **proposal})
            return

        expected = self.reference["steps"][step]
        assert role == expected["role"], (step, role, expected["role"])
        torch.testing.assert_close(logits, expected["logits"], rtol=0.005, atol=0.0001,
                                   msg=lambda msg: f"{role} step {step} logits: {msg}")
        error = (logits.float() - expected["logits"].float()).abs().max().item()
        self.max_logit_error = max(self.max_logit_error, error)
        if role == "draft":
            torch.testing.assert_close(proposal["spec_tokens"], expected["spec_tokens"], rtol=0, atol=0,
                                       msg=lambda msg: f"draft step {step} proposed tokens: {msg}")
            torch.testing.assert_close(proposal["confidence"], expected["confidence"], rtol=0.005, atol=0.0001,
                                       msg=lambda msg: f"draft step {step} confidence: {msg}")
            error = (proposal["confidence"].float() - expected["confidence"].float()).abs().max().item()
            self.max_confidence_error = max(self.max_confidence_error, error)
        assert len(caches) == len(expected["caches"])
        for index, (observed, wanted) in enumerate(zip(caches, expected["caches"])):
            if not torch.equal(observed, wanted):
                assert observed.shape == wanted.shape, (index, observed.shape, wanted.shape)
                differences = (observed != wanted).flatten(1).sum(1).tolist()
                raise AssertionError(f"{role} step {step} cache {index}: unequal bytes per page {differences}")
        self.steps.append(role)

    def finish(self, output, spec_steps):
        assert self.steps and self.caches and self.tokens
        assert output.finish_reason == "length", output.finish_reason
        assert len(self.tokens) == (5 if self.dspark else 3), self.tokens
        assert all(type(token) is int and 0 <= token < 1024 for token in self.tokens), self.tokens
        roles = [step["role"] for step in self.steps] if self.writing else self.steps
        if self.dspark:
            assert list(spec_steps) == [4], spec_steps
            # Prefill and the terminal scheduler pass also run both workers.
            expected_roles = ["main", "draft"] * (spec_steps[0] + 2)
        else:
            assert not spec_steps, spec_steps
            expected_roles = ["main"] * (len(self.tokens) + 1)
        assert roles == expected_roles, (roles, expected_roles)
        generation = {"tokens": self.tokens, "text": output.output_text,
                      "finish": output.finish_reason, "spec_steps": spec_steps}
        if self.writing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
                temp = Path(stream.name)
                try:
                    torch.save({"generation": generation, "steps": self.steps}, stream)
                    stream.flush()
                    temp.replace(self.path)
                finally:
                    temp.unlink(missing_ok=True)
        else:
            assert len(self.steps) == len(self.reference["steps"])
            assert generation == self.reference["generation"], (generation, self.reference["generation"])
        proposal = ({"draft_steps": self.draft_steps, "max_abs_confidence_error": self.max_confidence_error}
                    if self.draft_steps else {})
        return {"path": str(self.path), "action": "write" if self.writing else "compare",
                "tokens": self.tokens, "model_calls": len(self.steps),
                "cache_tensors_per_call": len(self.caches), "max_abs_logit_error": self.max_logit_error, **proposal}
