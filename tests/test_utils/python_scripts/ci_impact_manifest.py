# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generate expected CI impact owners from the repository recipes."""

import argparse
import json
from pathlib import Path

from tests.test_utils.python_scripts import recipe_parser
from tests.test_utils.python_scripts.ci_impact import CATEGORIES, expected_manifest


def _unit_candidates(repo: Path, platform: str) -> list[dict[str, str]]:
    recipe = repo / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    owners = []
    for line in recipe.read_text().splitlines():
        marker = "- test_case: ["
        if marker in line:
            owners.append({"bucket": line.split(marker, 1)[1].split("]", 1)[0]})
    return owners


def _functional_candidates(repo: Path, platform: str, scope: str, cadence: str) -> list[dict]:
    return [
        {"model": workload.spec["model"], "test_case": workload.spec["test_case"]}
        for workload in recipe_parser.load_workloads(
            scope=scope,
            container_tag="latest",
            environment="dev",
            test_cases="all",
            platform=f"dgx_{platform}",
            cadence=cadence,
        )
        if workload.type != "build"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--scope", default="L1")
    parser.add_argument("--cadence", default="nightly")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrices = {
        "unit:h100": _unit_candidates(args.repo, "h100"),
        "unit:gb200": _unit_candidates(args.repo, "gb200"),
        "functional:h100": _functional_candidates(args.repo, "h100", args.scope, args.cadence),
        "functional:gb200": _functional_candidates(args.repo, "gb200", args.scope, args.cadence),
    }
    assert set(matrices) == set(CATEGORIES)
    args.output.write_text(
        json.dumps(expected_manifest(args.repo, matrices), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
