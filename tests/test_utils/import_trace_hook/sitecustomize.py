# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Record repository modules imported by a functional CI workload."""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

_REPO = Path(os.environ.get("MCORE_IMPACT_REPO", "/opt/megatron-lm")).resolve()
_OUTPUT_DIR = os.environ.get("MCORE_IMPACT_TRACE_DIR")


def _repository_files() -> list[str]:
    files = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(_REPO)
        except (OSError, ValueError):
            continue
        normalized = relative.as_posix()
        if normalized.startswith(("megatron/", "examples/", "tools/")) or normalized in {
            "pretrain_gpt.py",
            "pretrain_hybrid.py",
            "pretrain_mamba.py",
            "pretrain_vlm.py",
            "train_rl.py",
        }:
            files.add(normalized)
    return sorted(files)


def _write_trace() -> None:
    if not _OUTPUT_DIR:
        return
    output = Path(_OUTPUT_DIR) / f"import-trace-{os.getpid()}.json"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"files": _repository_files()}, sort_keys=True) + "\n")
        temporary.replace(output)
    except OSError as error:
        print(f"warning: could not write CI import trace {output}: {error}", file=sys.stderr)


if _OUTPUT_DIR:
    atexit.register(_write_trace)
