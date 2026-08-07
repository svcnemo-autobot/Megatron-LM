# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pathlib
import threading
from unittest.mock import Mock

import click
import pytest

from tests.test_utils.python_scripts import launch_nemo_run_workload


def test_nccl_watchdog_timeout_is_flaky():
    log = "Watchdog caught collective operation timeout: WorkNCCL(SeqNum=281)"

    assert launch_nemo_run_workload.is_flaky_failure(log)


def test_hang_prone_flaky_failure_cancels_active_attempt():
    experiment = Mock()
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    stop_event = threading.Event()
    failure_detected_event = threading.Event()
    monitor = threading.Thread(
        target=launch_nemo_run_workload._cancel_on_flaky_failure,
        args=(experiment, "task-1", log_buffer, stop_event, failure_detected_event, 0.01),
    )

    monitor.start()
    log_buffer.write("Watchdog caught collective operation timeout")
    monitor.join(timeout=1)

    assert not monitor.is_alive()
    assert failure_detected_event.is_set()
    experiment.cancel.assert_called_once_with("task-1")


def test_non_hanging_flaky_failure_does_not_cancel_active_attempt():
    experiment = Mock()
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    log_buffer.write("found NaN in local forward loss calculation")
    stop_event = threading.Event()
    failure_detected_event = threading.Event()
    monitor = threading.Thread(
        target=launch_nemo_run_workload._cancel_on_flaky_failure,
        args=(experiment, "task-1", log_buffer, stop_event, failure_detected_event, 0.01),
    )

    monitor.start()
    assert not failure_detected_event.wait(timeout=0.05)
    stop_event.set()
    monitor.join(timeout=1)

    assert not monitor.is_alive()
    assert launch_nemo_run_workload.is_flaky_failure(log_buffer.getvalue())
    experiment.cancel.assert_not_called()


def test_stopped_monitor_does_not_cancel_attempt():
    experiment = Mock()
    log_buffer = launch_nemo_run_workload._ThreadSafeBuffer()
    log_buffer.write("Watchdog caught collective operation timeout")
    stop_event = threading.Event()
    stop_event.set()
    failure_detected_event = threading.Event()

    launch_nemo_run_workload._cancel_on_flaky_failure(
        experiment, "task-1", log_buffer, stop_event, failure_detected_event, poll_interval=0.01
    )

    assert not failure_detected_event.is_set()
    experiment.cancel.assert_not_called()


def test_render_workload_script_rebases_current_checkout_only():
    script = 'cd /opt/megatron-lm\nTEST_PATH="/opt/megatron-lm"\nLEGACY=/opt/megatron-lm-legacy\nLOG={assets_dir}'

    rendered = launch_nemo_run_workload._render_workload_script(
        script, {"assets_dir": "/workspace/megatron/assets_dir"}, pathlib.PurePosixPath("/workspace/megatron")
    )

    assert rendered == (
        "cd /workspace/megatron\n"
        'TEST_PATH="/workspace/megatron"\n'
        "LEGACY=/opt/megatron-lm-legacy\n"
        "LOG=/workspace/megatron/assets_dir"
    )


def test_resolve_docker_workspace_uses_custom_paths_and_image_user(tmp_path):
    workspace, docker_kwargs = launch_nemo_run_workload._resolve_docker_workspace(
        tmp_path, "/workspace/megatron", None
    )

    assert workspace.host_root == tmp_path.resolve()
    assert workspace.container_root == pathlib.PurePosixPath("/workspace/megatron")
    assert workspace.assets_dir == pathlib.PurePosixPath("/workspace/megatron/assets_dir")
    assert workspace.artifacts_dir == pathlib.PurePosixPath("/workspace/megatron/artifacts_dir")
    assert docker_kwargs == {}


def test_resolve_docker_workspace_sets_explicit_user(tmp_path):
    _, docker_kwargs = launch_nemo_run_workload._resolve_docker_workspace(tmp_path, "/opt/megatron-lm", "1000:1000")

    assert docker_kwargs == {"user": "1000:1000"}


@pytest.mark.parametrize("container_root", ["opt/megatron-lm", "/opt/../root"])
def test_resolve_docker_workspace_rejects_invalid_container_root(tmp_path, container_root):
    with pytest.raises(click.BadParameter, match="absolute normalized path"):
        launch_nemo_run_workload._resolve_docker_workspace(tmp_path, container_root, None)
