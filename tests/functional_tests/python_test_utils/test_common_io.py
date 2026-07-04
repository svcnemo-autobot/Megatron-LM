# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the TensorBoard / JSON IO helpers in ``common.py``.

``test_common.py`` already covers the comparison ``pipeline`` and the
``Test`` model hierarchy. This module covers the previously untested IO
surface of ``common.py``:

* ``read_tb_logs_as_list`` - reading TensorBoard event files into
  ``GoldenValueMetric`` objects, including missing-step filling, rounding,
  positional ``index`` selection, and the ``index == -1`` merge path.
* ``_load_event_accumulators_with_scalars`` - dropping scalar-less event
  files while preserving order.
* ``read_golden_values_from_json`` - deserialising golden values from JSON.
* ``GoldenValues`` / ``GoldenValueMetric.__repr__`` - the pydantic models.
"""

import json

import pytest

from tests.functional_tests.python_test_utils.common import (
    GoldenValueMetric,
    GoldenValues,
    _load_event_accumulators_with_scalars,
    read_golden_values_from_json,
    read_tb_logs_as_list,
)

# helpers


def write_events(path, scalars, walltime_offset: float = 0.0):
    """Write a single TensorBoard event file under ``path``.

    ``scalars`` maps a scalar tag to a ``{step: value}`` mapping. Returns the
    ``SummaryWriter`` so the caller can keep files distinct if needed.
    """
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(path))
    for tag, step_values in scalars.items():
        for step, value in step_values.items():
            writer.add_scalar(tag, value, global_step=step, walltime=walltime_offset + step)
    writer.flush()
    writer.close()
    return writer


# read_tb_logs_as_list - no data


class TestReadTbLogsNoData:
    def test_returns_none_when_no_event_files(self, tmp_path):
        assert read_tb_logs_as_list(str(tmp_path)) is None

    def test_returns_none_when_no_scalar_data(self, tmp_path):
        # An event dir with a writer that never logged a scalar yields no
        # accumulators with scalar tags.
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tmp_path))
        writer.flush()
        writer.close()
        assert read_tb_logs_as_list(str(tmp_path)) is None


# read_tb_logs_as_list - happy path


class TestReadTbLogsHappyPath:
    def test_reads_scalars_and_fills_missing_steps(self, tmp_path):
        # Log steps 1..5 for "lm loss"; ask for train_iters=10, step_size=5.
        # Sampled steps are start_idx (1) plus multiples of step_size (5, 10).
        write_events(tmp_path, {"lm loss": {s: float(s) for s in range(1, 6)}})

        result = read_tb_logs_as_list(
            str(tmp_path), index=0, train_iters=10, start_idx=1, step_size=5
        )

        assert result is not None
        assert "lm loss" in result
        metric = result["lm loss"]
        assert isinstance(metric, GoldenValueMetric)
        # Keys kept: 1 (start_idx), 5 and 10 (multiples of step_size).
        assert set(metric.values.keys()) == {1, 5, 10}
        assert metric.values[1] == 1.0
        assert metric.values[5] == 5.0
        # Step 10 was never logged -> filled with the "nan" sentinel string.
        assert metric.values[10] == "nan"
        assert metric.start_step == 1
        assert metric.end_step == 10
        assert metric.step_interval == 5

    def test_rounds_values_to_five_decimals(self, tmp_path):
        write_events(tmp_path, {"lm loss": {1: 1.123456789}})

        result = read_tb_logs_as_list(
            str(tmp_path), index=0, train_iters=1, start_idx=1, step_size=5
        )

        assert result["lm loss"].values[1] == pytest.approx(1.12346)

    def test_multiple_scalars_are_all_returned(self, tmp_path):
        write_events(
            tmp_path,
            {"lm loss": {1: 1.0}, "num-zeros": {1: 7.0}},
        )

        result = read_tb_logs_as_list(
            str(tmp_path), index=0, train_iters=1, start_idx=1, step_size=5
        )

        assert set(result.keys()) == {"lm loss", "num-zeros"}
        assert result["num-zeros"].values[1] == 7.0


# read_tb_logs_as_list - index selection


class TestReadTbLogsIndexSelection:
    def test_out_of_range_index_returns_none(self, tmp_path):
        write_events(tmp_path, {"lm loss": {1: 1.0}})
        # Only one accumulator (index 0) exists; index 5 is out of range.
        assert (
            read_tb_logs_as_list(str(tmp_path), index=5, train_iters=1, step_size=5) is None
        )

    def test_negative_index_merges_all_accumulators(self, tmp_path):
        # Two event files in one dir; index=-1 keeps all accumulators and the
        # first-seen value for a step wins across them. Each SummaryWriter
        # instance produces its own event file, so writing twice to the same
        # directory (with an increasing walltime offset to fix ordering) gives
        # two accumulators for read_tb_logs_as_list to merge.
        write_events(tmp_path, {"lm loss": {1: 1.0}}, walltime_offset=0.0)
        write_events(tmp_path, {"lm loss": {2: 2.0}}, walltime_offset=100.0)

        result = read_tb_logs_as_list(
            str(tmp_path), index=-1, train_iters=2, start_idx=1, step_size=1
        )

        assert result is not None
        assert result["lm loss"].values[1] == 1.0
        assert result["lm loss"].values[2] == 2.0


# _load_event_accumulators_with_scalars


class TestLoadEventAccumulators:
    def test_drops_scalarless_files_and_preserves_order(self, tmp_path):
        import glob

        from torch.utils.tensorboard import SummaryWriter

        # First dir: no scalars logged.
        empty_writer = SummaryWriter(log_dir=str(tmp_path / "empty"))
        empty_writer.flush()
        empty_writer.close()

        # Second dir: has a scalar.
        write_events(tmp_path / "withdata", {"lm loss": {1: 1.0}})

        empty_files = glob.glob(f"{tmp_path / 'empty'}/events*tfevents*")
        data_files = glob.glob(f"{tmp_path / 'withdata'}/events*tfevents*")
        assert empty_files and data_files

        accumulators = _load_event_accumulators_with_scalars(empty_files + data_files)

        # The scalar-less file is dropped; only the one with data remains.
        assert len(accumulators) == 1
        assert accumulators[0].Tags()["scalars"]

    def test_returns_empty_list_for_no_files(self):
        assert _load_event_accumulators_with_scalars([]) == []


# read_golden_values_from_json


class TestReadGoldenValuesFromJson:
    def test_roundtrip(self, tmp_path):
        payload = {
            "lm loss": {
                "start_step": 1,
                "end_step": 5,
                "step_interval": 5,
                "values": {"1": 1.0, "5": 2.0},
            }
        }
        path = tmp_path / "golden.json"
        path.write_text(json.dumps(payload))

        result = read_golden_values_from_json(str(path))

        assert set(result.keys()) == {"lm loss"}
        metric = result["lm loss"]
        assert isinstance(metric, GoldenValueMetric)
        assert metric.start_step == 1
        assert metric.end_step == 5
        assert metric.step_interval == 5
        # JSON object keys are strings; pydantic coerces them back to ints.
        assert metric.values == {1: 1.0, 5: 2.0}

    def test_accepts_pathlib_path(self, tmp_path):
        payload = {
            "loss": {"start_step": 1, "end_step": 1, "step_interval": 1, "values": {"1": 3.0}}
        }
        path = tmp_path / "golden.json"
        path.write_text(json.dumps(payload))

        result = read_golden_values_from_json(path)

        assert result["loss"].values == {1: 3.0}


# GoldenValues / GoldenValueMetric models


class TestGoldenValueModels:
    def test_golden_values_root_parses_mapping(self):
        gv = GoldenValues(
            **{
                "loss": {
                    "start_step": 1,
                    "end_step": 3,
                    "step_interval": 1,
                    "values": {1: 1.0, 2: 2.0, 3: 3.0},
                }
            }
        )
        assert set(gv.root.keys()) == {"loss"}
        assert gv.root["loss"].end_step == 3

    def test_metric_repr_includes_bounds_and_values(self):
        metric = GoldenValueMetric(
            start_step=1, end_step=3, step_interval=2, values={1: 1.0, 3: 3.0}
        )
        rendered = repr(metric)
        assert "(1,3,2)" in rendered
        assert "(1, 1.0)" in rendered
        assert "(3, 3.0)" in rendered
