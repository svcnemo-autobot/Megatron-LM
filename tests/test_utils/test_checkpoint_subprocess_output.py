# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU diagnostics coverage; not a substitute for the MIMO checkpoint GPU row."""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = (
    Path(__file__).parents[1]
    / "functional_tests/test_cases/mimo/mimo_hetero_gtp_checkpoint_round_trip"
    / "test_checkpoint_round_trip.py"
)


def _load_helpers(run):
    tree = ast.parse(RUNNER.read_text())
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_run_with_failure_output", "_print_failure_output"}
    ]
    assert len(helpers) == 2
    namespace = {
        "subprocess": SimpleNamespace(
            run=run,
            CompletedProcess=subprocess.CompletedProcess,
            TimeoutExpired=subprocess.TimeoutExpired,
        )
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["_run_with_failure_output"]


@pytest.mark.parametrize("returncode", [0, 1, -9])
def test_child_failure_preserves_early_diagnostics(returncode, capsys):
    stdout = "early rank failure\n" + "x" * 10000
    result = subprocess.CompletedProcess(["child"], returncode, stdout, "rank stderr")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return result

    assert _load_helpers(run)(["child"], timeout=1800) is result
    assert calls == [(["child"], dict(capture_output=True, text=True, timeout=1800))]
    output = capsys.readouterr().out
    if returncode:
        assert stdout in output
        assert "rank stderr" in output
    else:
        assert output == ""


@pytest.mark.parametrize("output", [None, "partial output", b"partial output\xff"])
def test_timeout_preserves_output_and_exception(output, capsys):
    error = subprocess.TimeoutExpired(["child"], 1800, output=output, stderr=output)

    def run(*args, **kwargs):
        raise error

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _load_helpers(run)(["child"], timeout=1800)
    assert raised.value is error
    captured = capsys.readouterr().out
    assert "--- child stdout ---" in captured
    assert "--- child stderr ---" in captured
    if output is not None:
        assert "partial output" in captured


def test_both_subprocess_paths_publish_failures():
    tree = ast.parse(RUNNER.read_text())
    for name in ("_run_launcher", "_run_comparator"):
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_with_failure_output"
            for node in ast.walk(function)
        )
