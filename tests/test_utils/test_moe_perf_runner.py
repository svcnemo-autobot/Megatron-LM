# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU mechanism tests; the managed moe_perf row remains the GPU validation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = Path(__file__).parents[1] / "functional_tests/test_cases/common/moe_perf/__main__.py"


class TensorState:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return TensorState(self.value)


class GeneratorState:
    def __init__(self, value):
        self.value = value

    def clone_state(self):
        return GeneratorState(self.value)


@pytest.mark.parametrize("state_type", [TensorState, GeneratorState])
def test_expert_rng_replay_preserves_snapshot_and_other_streams(state_type):
    tree = ast.parse(RUNNER.read_text())
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_clone_rng_state", "_reset_expert_rng_state"}
    ]
    assert len(helpers) == 2
    namespace = {
        "torch": SimpleNamespace(Generator=GeneratorState),
        "get_expert_parallel_rng_tracker_name": lambda: "expert",
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(RUNNER), "exec"), namespace)

    class Tracker:
        states = {"expert": state_type(7), "model": state_type(13)}

        def get_states(self):
            return self.states.copy()

        def set_states(self, states):
            self.states = states

    tracker = Tracker()
    snapshot = namespace["_clone_rng_state"](tracker.get_states()["expert"])
    for _ in range(3):
        other = tracker.states["model"]
        namespace["_reset_expert_rng_state"](tracker, snapshot)
        assert tracker.states["expert"].value == 7
        assert tracker.states["model"] is other
        tracker.states["expert"].value += 1
        tracker.states["model"].value += 1
        assert snapshot.value == 7
    assert tracker.states["model"].value == 16

    benchmark = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_moe_layer"
    )
    loop = next(node for node in benchmark.body if isinstance(node, ast.For))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_reset_expert_rng_state"
        for node in ast.walk(loop)
    )


@pytest.mark.parametrize("exit_code", [0, 1, 2, 5])
def test_module_entrypoint_propagates_pytest_exit_code(exit_code):
    tree = ast.parse(RUNNER.read_text())
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    calls = []

    def run_pytest(args):
        calls.append(args)
        return exit_code

    with pytest.raises(SystemExit) as error:
        exec(
            compile(ast.Module(body=[guard], type_ignores=[]), str(RUNNER), "exec"),
            {
                "__name__": "__main__",
                "__file__": str(RUNNER),
                "pytest": SimpleNamespace(main=run_pytest),
            },
        )
    assert error.value.code == exit_code
    assert calls == [["-x", "-v", "-s", str(RUNNER)]]
