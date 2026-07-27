# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tests.test_utils.python_scripts.ci_impact import (
    CATEGORIES,
    build_index,
    filter_matrix,
    select_tests,
    write_metadata,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI Impact Test",
            "-c",
            "user.email=ci-impact@example.com",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return _git(repo, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "megatron/core").mkdir(parents=True)
    (repo / "tests/unit_tests/transformer").mkdir(parents=True)
    (repo / "tests/functional_tests/test_cases/gpt/case_a").mkdir(parents=True)
    (repo / "tests/test_utils/recipes/h100").mkdir(parents=True)
    (repo / "tests/test_utils/recipes/gb200").mkdir(parents=True)
    (repo / "megatron/__init__.py").write_text("")
    (repo / "megatron/core/__init__.py").write_text("")
    (repo / "megatron/core/a.py").write_text("VALUE = 1\n")
    (repo / "megatron/core/b.py").write_text("from megatron.core.a import VALUE\n")
    (repo / "tests/unit_tests/transformer/test_a.py").write_text(
        "from megatron.core.b import VALUE\n\ndef test_value():\n    assert VALUE == 1\n"
    )
    (repo / "tests/functional_tests/test_cases/gpt/case_a/model_config.yaml").write_text(
        "TEST_TYPE: regular\n"
    )
    h100 = """products:
  - test_case: [tests/unit_tests/transformer/**/*.py]
    products:
      - environment: [dev]
"""
    gb200 = """products:
  - test_case: [tests/unit_tests/**/*.py]
    products:
      - environment: [dev]
"""
    (repo / "tests/test_utils/recipes/h100/unit-tests.yaml").write_text(h100)
    (repo / "tests/test_utils/recipes/gb200/unit-tests.yaml").write_text(gb200)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    base = _commit(repo, "base")
    return repo, base


def _coverage_database(path: Path, files: list[str]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
        connection.execute("CREATE TABLE line_bits (file_id INTEGER, context_id INTEGER, numbits BLOB)")
        for index, file_path in enumerate(files, 1):
            connection.execute("INSERT INTO file VALUES (?, ?)", (index, file_path))
            connection.execute("INSERT INTO line_bits VALUES (?, 1, ?)", (index, b"\x01"))


def _expected(path: Path, **categories: list[str]) -> Path:
    payload = {category: categories.get(category, []) for category in CATEGORIES}
    path.write_text(json.dumps({"categories": payload}))
    return path


def test_build_index_combines_unit_coverage_and_functional_trace(tmp_path):
    repo, source_sha = _repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    unit = artifacts / "unit"
    functional = artifacts / "functional"
    unit.mkdir(parents=True)
    functional.mkdir(parents=True)
    write_metadata(unit / "impact-metadata.json", "unit", "h100", "bucket-a", source_sha)
    _coverage_database(unit / "coverage_report", [str(repo / "megatron/core/a.py")])
    write_metadata(
        functional / "impact-metadata.json", "functional", "h100", "gpt/case_a", source_sha
    )
    (functional / "import-trace-1.json").write_text(
        json.dumps({"files": ["megatron/core/b.py"]})
    )
    expected = _expected(
        tmp_path / "expected.json",
        **{"unit:h100": ["bucket-a"], "functional:h100": ["gpt/case_a"]},
    )

    index = build_index(
        repo,
        artifacts,
        expected,
        source_sha,
        required_categories=("unit:h100", "functional:h100"),
    )

    assert index["complete_categories"] == ["unit:h100", "functional:h100"]
    assert index["files"]["megatron/core/a.py"] == ["unit:h100:bucket-a"]
    assert index["files"]["megatron/core/b.py"] == ["functional:h100:gpt/case_a"]


def test_build_index_rejects_incomplete_category(tmp_path):
    repo, source_sha = _repo(tmp_path)
    artifacts = tmp_path / "artifacts"
    unit = artifacts / "unit"
    unit.mkdir(parents=True)
    write_metadata(unit / "impact-metadata.json", "unit", "h100", "bucket-a", source_sha)
    _coverage_database(unit / "coverage_report", [str(repo / "megatron/core/a.py")])
    expected = _expected(tmp_path / "expected.json", **{"unit:h100": ["bucket-a", "bucket-b"]})

    with pytest.raises(ValueError, match="incomplete unit:h100"):
        build_index(
            repo,
            artifacts,
            expected,
            source_sha,
            required_categories=("unit:h100",),
        )


def test_selector_uses_execution_owners_and_ast_frontier(tmp_path):
    repo, base = _repo(tmp_path)
    index = {
        "version": 1,
        "source_sha": base,
        "recipe_digest": __import__(
            "tests.test_utils.python_scripts.ci_impact", fromlist=["_recipe_digest"]
        )._recipe_digest(repo),
        "complete_categories": ["unit:h100", "functional:h100"],
        "expected_owners": {
            "unit:h100": ["tests/unit_tests/transformer/**/*.py"],
            "functional:h100": ["gpt/case_a"],
        },
        "files": {
            "megatron/core/b.py": [
                "unit:h100:tests/unit_tests/transformer/**/*.py",
                "functional:h100:gpt/case_a",
            ]
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    (repo / "megatron/core/a.py").write_text("VALUE = 2\n")
    head = _commit(repo, "change a")

    selection = select_tests(repo, index_path, base, head, None)

    assert selection["mode"] == "selective"
    assert selection["categories"]["unit:h100"]["owners"] == [
        "tests/unit_tests/transformer/**/*.py"
    ]
    assert selection["categories"]["functional:h100"]["owners"] == ["gpt/case_a"]
    assert selection["categories"]["unit:gb200"]["mode"] == "full"


def test_selector_fails_closed_for_unobserved_runtime_file(tmp_path):
    repo, base = _repo(tmp_path)
    index = {
        "version": 1,
        "source_sha": base,
        "recipe_digest": __import__(
            "tests.test_utils.python_scripts.ci_impact", fromlist=["_recipe_digest"]
        )._recipe_digest(repo),
        "complete_categories": ["unit:h100"],
        "expected_owners": {"unit:h100": ["tests/unit_tests/transformer/**/*.py"]},
        "files": {},
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    (repo / "megatron/core/a.py").write_text("VALUE = 2\n")
    head = _commit(repo, "change a")

    selection = select_tests(repo, index_path, base, head, None)

    assert selection["mode"] == "full"
    assert "no observed test owner" in selection["reason"]


def test_direct_test_change_selects_most_specific_bucket(tmp_path):
    repo, base = _repo(tmp_path)
    (repo / "tests/unit_tests/transformer/test_a.py").write_text(
        "from megatron.core.b import VALUE\n\ndef test_value():\n    assert VALUE > 0\n"
    )
    head = _commit(repo, "change test")

    selection = select_tests(repo, None, base, head, None)

    assert selection["mode"] == "selective"
    assert selection["categories"]["unit:h100"]["owners"] == [
        "tests/unit_tests/transformer/**/*.py"
    ]


def test_filter_matrix_uses_placeholder_for_empty_selection():
    selection = {
        "categories": {
            "unit:h100": {"mode": "selective", "owners": []},
        }
    }

    result = filter_matrix(selection, "unit:h100", [{"bucket": "bucket-a"}])

    assert result == {"matrix": [{"bucket": "__no_tests__"}], "has_tests": False}
