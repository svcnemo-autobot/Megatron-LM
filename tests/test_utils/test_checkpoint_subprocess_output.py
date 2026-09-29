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
        and node.name in {"_run_with_failure_output", "_print_failure_output", "_print_rank_logs"}
    ]
    assert len(helpers) == 3
    namespace = {
        "Path": Path,
        "subprocess": SimpleNamespace(
            run=run,
            CompletedProcess=subprocess.CompletedProcess,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
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


@pytest.mark.parametrize("timeout", [False, True])
def test_redirected_rank_logs_survive_truncated_console(timeout, tmp_path, capsys):
    rank = tmp_path / "run" / "attempt_0" / "5"
    rank.mkdir(parents=True)
    (rank / "stderr.log").write_text("complete checkpoint write exception")
    (rank / "stdout.log").write_text("rank stdout")
    (rank / "error.json").write_text('{"message": "write failure"}')
    (rank / "checkpoint.bin").write_bytes(b"do not print checkpoint data")
    error = subprocess.TimeoutExpired(["child"], 1800, stderr=b"truncated tee")

    def run(*args, **kwargs):
        assert "log_dir" not in kwargs
        if timeout:
            raise error
        return subprocess.CompletedProcess(["child"], 1, "", "truncated tee")

    helper = _load_helpers(run)
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            helper(["child"], log_dir=tmp_path, timeout=1800)
        assert raised.value is error
    else:
        assert helper(["child"], log_dir=tmp_path, timeout=1800).returncode == 1
    output = capsys.readouterr().out
    assert "complete checkpoint write exception" in output
    assert "rank stdout" in output
    assert '"message": "write failure"' in output
    assert "do not print checkpoint data" not in output


def test_success_does_not_publish_rank_logs(tmp_path, capsys):
    (tmp_path / "stderr.log").write_text("successful rank log")
    result = subprocess.CompletedProcess(["child"], 0, "", "")
    assert _load_helpers(lambda *args, **kwargs: result)(["child"], log_dir=tmp_path) is result
    assert capsys.readouterr().out == ""


def _load_pretrain_helper(checkpoint_exception):
    tree = ast.parse(RUNNER.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_pretrain_with_failure_output"
    )
    import sys

    namespace = {"CheckpointException": checkpoint_exception, "sys": sys}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["_run_pretrain_with_failure_output"]


def test_checkpoint_root_causes_precede_distributed_traceback(capsys):
    class CheckpointFailure(BaseException):
        failures = {4: (OSError("write failed"), None), 1: (ValueError("bad shard"), None)}

    error = CheckpointFailure()

    def main():
        raise error

    with pytest.raises(CheckpointFailure) as raised:
        _load_pretrain_helper(CheckpointFailure)(main)
    assert raised.value is error
    assert capsys.readouterr().err.splitlines() == [
        "Checkpoint failure on rank 1: ValueError: bad shard",
        "Checkpoint failure on rank 4: OSError: write failed",
    ]


def test_pretrain_success_has_no_failure_output(capsys):
    calls = []
    _load_pretrain_helper(BaseException)(lambda: calls.append(True))
    assert calls == [True]
    assert capsys.readouterr().err == ""


def test_other_pretrain_errors_are_unchanged(capsys):
    class CheckpointFailure(BaseException):
        pass

    error = RuntimeError("unrelated error")

    def main():
        raise error

    with pytest.raises(RuntimeError) as raised:
        _load_pretrain_helper(CheckpointFailure)(main)
    assert raised.value is error
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("free_bytes", [15, 16, 32])
def test_round_trip_space_preflight(free_bytes, tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "shard.distcp").write_bytes(b"x" * 12)
    metadata = source / "metadata"
    metadata.mkdir()
    (metadata / "index").write_bytes(b"y" * 4)
    tree = ast.parse(RUNNER.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_require_round_trip_space"
    )
    paths = []

    def disk_usage(path):
        paths.append(path)
        return SimpleNamespace(free=free_bytes)

    namespace = {"Path": Path, "shutil": SimpleNamespace(disk_usage=disk_usage)}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(RUNNER), "exec"), namespace)
    check = namespace["_require_round_trip_space"]
    if free_bytes < 16:
        with pytest.raises(OSError, match="need at least 16 bytes.*have 15 bytes free"):
            check(source, tmp_path)
    else:
        check(source, tmp_path)
    assert paths == [tmp_path]
    assert f"source_bytes=16, free_bytes={free_bytes}" in capsys.readouterr().out
    assert (source / "shard.distcp").read_bytes() == b"x" * 12


def test_space_preflight_precedes_resave():
    tree = ast.parse(RUNNER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_hetero_mimo_20l_checkpoint_round_trip_is_exact"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    check = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_require_round_trip_space"
    )
    resave = next(
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_run_launcher"
        and any(keyword.arg == "resave_after_load" for keyword in node.keywords)
    )
    assert check.lineno < resave.lineno
