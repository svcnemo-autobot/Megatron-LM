# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build and consume execution-derived CI test impact indexes."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

INDEX_VERSION = 1
CATEGORIES = ("unit:h100", "unit:gb200", "functional:h100", "functional:gb200")
RUNTIME_PREFIXES = ("megatron/", "examples/", "tools/")
ROOT_ENTRYPOINTS = {
    "pretrain_gpt.py",
    "pretrain_hybrid.py",
    "pretrain_mamba.py",
    "pretrain_vlm.py",
    "train_rl.py",
}
FULL_RUN_PREFIXES = (
    ".github/",
    "docker/",
    "tests/test_utils/python_scripts/",
    "tests/test_utils/import_trace_hook/",
    "tests/test_utils/recipes/",
)
FULL_RUN_FILES = {"pyproject.toml", "uv.lock", ".python-version"}


@dataclass(frozen=True)
class Change:
    """One changed path, including both sides of a rename when applicable."""

    status: str
    path: str


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=check, capture_output=True, text=True
    )
    return result.stdout


def _normalize_path(path: str, repo: Path) -> str | None:
    value = path.replace("\\", "/")
    prefixes = (str(repo.resolve()).replace("\\", "/") + "/", "/opt/megatron-lm/")
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.removeprefix("./")
    if value.startswith(("megatron/", "examples/", "tools/")) or value in ROOT_ENTRYPOINTS:
        return value
    return None


def _recipe_digest(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "tests/test_utils/recipes").rglob("*.yaml")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metadata_files(artifacts: Path) -> list[Path]:
    return sorted(artifacts.rglob("impact-metadata.json"))


def _coverage_files(directory: Path) -> set[str]:
    database = next(
        (path for path in (directory / ".coverage", directory / "coverage_report") if path.is_file()),
        None,
    )
    if database is None:
        raise ValueError(f"unit impact artifact {directory} has no coverage database")
    with sqlite3.connect(database) as connection:
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "line_bits" in tables:
            rows = connection.execute(
                "SELECT DISTINCT file.path FROM file JOIN line_bits ON file.id = line_bits.file_id "
                "WHERE length(line_bits.numbits) > 0"
            )
        elif "arc" in tables:
            rows = connection.execute(
                "SELECT DISTINCT file.path FROM file JOIN arc ON file.id = arc.file_id"
            )
        else:
            raise ValueError(f"unsupported coverage schema in {database}")
        return {row[0] for row in rows}


def _trace_files(directory: Path, repo: Path) -> set[str]:
    traced: set[str] = set()
    for trace in directory.rglob("import-trace-*.json"):
        payload = json.loads(trace.read_text())
        files = payload.get("files")
        if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
            raise ValueError(f"invalid import trace {trace}")
        traced.update(path for path in files if _normalize_path(path, repo) is not None)
    if not traced:
        raise ValueError(f"functional impact artifact {directory} has no import traces")
    return traced


def build_index(
    repo: Path,
    artifacts: Path,
    expected_path: Path,
    source_sha: str,
    required_categories: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build a complete source-path to test-owner index from one full CI run."""
    expected_payload = json.loads(expected_path.read_text())
    expected = expected_payload.get("categories", {})
    if set(expected) != set(CATEGORIES):
        raise ValueError(f"expected manifest must define categories {CATEGORIES}")

    required = set(
        required_categories
        if required_categories is not None
        else (category for category in CATEGORIES if expected[category])
    )
    if not required.issubset(CATEGORIES):
        raise ValueError(f"unknown required categories: {sorted(required - set(CATEGORIES))}")
    owners_by_file: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, set[str]] = defaultdict(set)
    pending: list[tuple[str, str, set[str]]] = []
    for metadata_path in _metadata_files(artifacts):
        metadata = json.loads(metadata_path.read_text())
        kind = metadata.get("kind")
        platform = metadata.get("platform")
        owner = metadata.get("owner")
        artifact_sha = metadata.get("source_sha")
        category = f"{kind}:{platform}"
        if category not in CATEGORIES or not isinstance(owner, str):
            raise ValueError(f"invalid impact metadata {metadata_path}")
        if artifact_sha != source_sha:
            raise ValueError(
                f"artifact {metadata_path} was produced for {artifact_sha}, expected {source_sha}"
            )
        if owner in seen[category]:
            raise ValueError(f"duplicate {category} artifact for {owner}")
        seen[category].add(owner)
        relative_parts = metadata_path.relative_to(artifacts).parts
        artifact_root = artifacts / relative_parts[0] if len(relative_parts) > 1 else artifacts
        raw_paths = (
            _coverage_files(artifact_root)
            if kind == "unit"
            else _trace_files(artifact_root, repo)
        )
        pending.append((category, owner, raw_paths))

    complete_categories = []
    for category in CATEGORIES:
        expected_owners = expected[category]
        if not isinstance(expected_owners, list) or not all(
            isinstance(owner, str) for owner in expected_owners
        ):
            raise ValueError(f"expected owners for {category} must be strings")
        expected_set = set(expected_owners)
        if expected_set and seen[category] == expected_set:
            complete_categories.append(category)
        elif seen[category] or category in required:
            missing = sorted(expected_set - seen[category])
            extra = sorted(seen[category] - expected_set)
            raise ValueError(f"incomplete {category} artifacts; missing={missing}, extra={extra}")

    for category, owner, raw_paths in pending:
        if category not in complete_categories:
            continue
        target = f"{category}:{owner}"
        for raw_path in raw_paths:
            normalized = _normalize_path(raw_path, repo)
            if normalized is not None:
                owners_by_file[normalized].add(target)

    return {
        "version": INDEX_VERSION,
        "source_sha": source_sha,
        "recipe_digest": _recipe_digest(repo),
        "complete_categories": complete_categories,
        "expected_owners": {
            category: sorted(set(expected[category])) for category in CATEGORIES
        },
        "files": {
            path: sorted(owners) for path, owners in sorted(owners_by_file.items())
        },
    }


def _changes(repo: Path, base_sha: str, head_sha: str) -> list[Change]:
    output = subprocess.run(
        ["git", "diff", "--name-status", "-z", base_sha, head_sha],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    changes: list[Change] = []
    index = 0
    while index < len(output) and output[index]:
        status = output[index].decode()
        index += 1
        if status.startswith(("R", "C")):
            old_path = output[index].decode()
            new_path = output[index + 1].decode()
            index += 2
            changes.extend((Change(status[0], old_path), Change(status[0], new_path)))
        else:
            changes.append(Change(status[0], output[index].decode()))
            index += 1
    return changes


def _is_runtime(path: str) -> bool:
    return path.startswith(RUNTIME_PREFIXES) or path in ROOT_ENTRYPOINTS


def _module_for_path(path: str) -> str:
    module = path[:-3].replace("/", ".")
    return module.removesuffix(".__init__")


def _resolve_import(candidate: str, modules: set[str]) -> str | None:
    parts = candidate.split(".")
    for stop in range(len(parts), 0, -1):
        module = ".".join(parts[:stop])
        if module in modules:
            return module
    return None


def _module_imports(path: str, source: str, modules: set[str]) -> tuple[set[str], bool]:
    tree = ast.parse(source, filename=path)
    package = _module_for_path(path)
    if not path.endswith("/__init__.py"):
        package = package.rpartition(".")[0]
    imports: set[str] = set()
    unresolved_dynamic = False

    def add(candidate: str) -> None:
        resolved = _resolve_import(candidate, modules)
        if resolved:
            imports.add(resolved)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                trim = node.level - 1
                if trim:
                    parts = parts[:-trim]
                base = ".".join([*parts, *([base] if base else [])])
            if base:
                add(base)
                for alias in node.names:
                    if alias.name != "*":
                        add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Call):
            dynamic = (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            )
            if dynamic:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
                    node.args[0].value, str
                ):
                    add(node.args[0].value)
                else:
                    unresolved_dynamic = True
    return imports, unresolved_dynamic


def _ast_owner_frontier(
    repo: Path, changed: set[str], indexed_files: dict[str, list[str]]
) -> tuple[set[str], list[str]]:
    """Find the nearest indexed importers for changed paths without execution data."""
    tracked = _git(repo, "ls-files", "*.py").splitlines()
    paths = [path for path in tracked if _is_runtime(path) or path.startswith("tests/unit_tests/")]
    by_module = {_module_for_path(path): path for path in paths}
    modules = set(by_module)
    reverse: dict[str, set[str]] = defaultdict(set)
    dynamic_changes: list[str] = []
    for module, path in by_module.items():
        try:
            imports, unresolved_dynamic = _module_imports(
                path, (repo / path).read_text(errors="replace"), modules
            )
        except (OSError, SyntaxError) as error:
            if path in changed:
                raise ValueError(f"cannot parse changed Python file {path}: {error}") from error
            continue
        for dependency in imports:
            reverse[dependency].add(module)
        if path in changed and unresolved_dynamic:
            dynamic_changes.append(path)

    starts = {
        _module_for_path(path)
        for path in changed
        if not indexed_files.get(path) and _module_for_path(path) in modules
    }
    frontier: set[str] = set()
    seen = set(starts)
    queue = deque(starts)
    while queue:
        module = queue.popleft()
        for importer in reverse.get(module, ()):
            if importer in seen:
                continue
            seen.add(importer)
            importer_path = by_module[importer]
            if indexed_files.get(importer_path):
                frontier.add(importer_path)
            else:
                queue.append(importer)
    return frontier, dynamic_changes


def _unit_patterns(repo: Path, platform: str) -> list[str]:
    recipe = repo / f"tests/test_utils/recipes/{platform}/unit-tests.yaml"
    patterns = []
    for line in recipe.read_text().splitlines():
        marker = "- test_case: ["
        if marker in line:
            patterns.append(line.split(marker, 1)[1].split("]", 1)[0])
    return patterns


def _pattern_matches(path: str, pattern: str) -> bool:
    if "**" in pattern:
        prefix = pattern.split("/**", 1)[0].rstrip("/")
        return path.startswith(prefix + "/")
    return fnmatch.fnmatch(path, pattern)


def _owning_unit_bucket(repo: Path, path: str, platform: str) -> str | None:
    matches = [
        pattern for pattern in _unit_patterns(repo, platform) if _pattern_matches(path, pattern)
    ]
    if platform == "gb200" and "launch_on_gb200" not in (repo / path).read_text(
        errors="replace"
    ):
        return None
    return max(matches, key=lambda pattern: len(pattern.split("/**", 1)[0])) if matches else None


def _category_target(category: str, owner: str) -> str:
    return f"{category}:{owner}"


def _all_full(reason: str, changed: Iterable[str]) -> dict[str, object]:
    return {
        "mode": "full",
        "reason": reason,
        "changed_paths": sorted(set(changed)),
        "categories": {
            category: {"mode": "full", "owners": []} for category in CATEGORIES
        },
    }


def select_tests(
    repo: Path, index_path: Path | None, base_sha: str, head_sha: str, force_full: str | None
) -> dict[str, object]:
    """Select affected tests, failing closed to full matrices on uncertainty."""
    if force_full:
        return _all_full(force_full, [])
    changes = _changes(repo, base_sha, head_sha)
    changed_paths = {change.path for change in changes}
    if any(path in FULL_RUN_FILES or path.startswith(FULL_RUN_PREFIXES) for path in changed_paths):
        return _all_full("shared CI, dependency, or test-infrastructure change", changed_paths)

    runtime_changes = {path for path in changed_paths if _is_runtime(path)}
    unknown_source = {
        path
        for path in changed_paths
        if path.endswith((".py", ".pyi", ".so", ".cu", ".cpp", ".cuh", ".h"))
        and not (
            _is_runtime(path)
            or path.startswith("tests/unit_tests/")
            or path.startswith("tests/functional_tests/test_cases/")
        )
    }
    if unknown_source:
        return _all_full(f"unclassified source paths: {sorted(unknown_source)}", changed_paths)
    non_python_runtime = {path for path in runtime_changes if not path.endswith(".py")}
    if non_python_runtime:
        return _all_full(f"non-Python runtime paths changed: {sorted(non_python_runtime)}", changed_paths)

    direct_targets: set[str] = set()
    for path in changed_paths:
        if path.startswith("tests/unit_tests/") and path.endswith(".py"):
            if not Path(path).name.startswith("test_"):
                return _all_full(f"shared unit-test support changed: {path}", changed_paths)
            if not (repo / path).exists():
                return _all_full(f"unit test was deleted or renamed: {path}", changed_paths)
            for platform in ("h100", "gb200"):
                bucket = _owning_unit_bucket(repo, path, platform)
                if bucket:
                    direct_targets.add(_category_target(f"unit:{platform}", bucket))
        if path.startswith("tests/functional_tests/test_cases/"):
            parts = Path(path).parts
            if len(parts) >= 5:
                owner = f"{parts[3]}/{parts[4]}"
                for platform in ("h100", "gb200"):
                    direct_targets.add(_category_target(f"functional:{platform}", owner))

    if not runtime_changes:
        return {
            "mode": "selective",
            "reason": "only direct tests or non-runtime files changed",
            "changed_paths": sorted(changed_paths),
            "categories": _targets_to_categories(
                direct_targets, set(CATEGORIES), runtime=False
            ),
        }
    if index_path is None or not index_path.is_file():
        return _all_full("no execution impact index is available", changed_paths)
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return _all_full(f"cannot read execution impact index: {error}", changed_paths)
    if index.get("version") != INDEX_VERSION:
        return _all_full("unsupported execution impact index version", changed_paths)
    source_sha = index.get("source_sha")
    if not isinstance(source_sha, str):
        return _all_full("execution impact index has no source SHA", changed_paths)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_sha, base_sha], cwd=repo
    )
    if ancestor.returncode:
        return _all_full("impact index source is not an ancestor of the PR base", changed_paths)
    if index.get("recipe_digest") != _recipe_digest(repo):
        return _all_full("test recipe digest differs from the impact index", changed_paths)
    drift = {
        change.path
        for change in _changes(repo, source_sha, base_sha)
        if _is_runtime(change.path)
        or change.path.startswith(("tests/unit_tests/", "tests/functional_tests/"))
        or change.path.startswith(FULL_RUN_PREFIXES)
        or change.path in FULL_RUN_FILES
    }

    files = index.get("files")
    complete = set(index.get("complete_categories", []))
    expected_owners = index.get("expected_owners")
    if (
        not isinstance(files, dict)
        or not complete.issubset(CATEGORIES)
        or not isinstance(expected_owners, dict)
    ):
        return _all_full("execution impact index is malformed", changed_paths)
    stale_paths = runtime_changes & drift
    if stale_paths:
        return _all_full(
            f"changed paths also changed since the impact index: {sorted(stale_paths)}",
            changed_paths,
        )
    try:
        ast_paths, dynamic = _ast_owner_frontier(repo, runtime_changes, files)
    except ValueError as error:
        return _all_full(str(error), changed_paths)
    if dynamic:
        return _all_full(f"changed files use unresolved dynamic imports: {dynamic}", changed_paths)
    stale_frontier = ast_paths & drift
    if stale_frontier:
        return _all_full(
            f"AST owner frontier changed since the impact index: {sorted(stale_frontier)}",
            changed_paths,
        )
    targets = set(direct_targets)
    for path in runtime_changes | ast_paths:
        owners = files.get(path, [])
        if not isinstance(owners, list) or not all(isinstance(owner, str) for owner in owners):
            return _all_full(f"invalid owners for {path} in impact index", changed_paths)
        targets.update(owners)
        if path in runtime_changes and not owners:
            # A new file can inherit owners through its importers. Existing indexed files cannot.
            if path in files or not any(files.get(importer) for importer in ast_paths):
                return _all_full(f"changed runtime path has no observed test owner: {path}", changed_paths)

    return {
        "mode": "selective",
        "reason": "execution owners plus transitive AST importers",
        "index_source_sha": source_sha,
        "changed_paths": sorted(changed_paths),
        "categories": _targets_to_categories(
            targets, complete, runtime=True, baseline_owners=expected_owners
        ),
    }


def _targets_to_categories(
    targets: set[str],
    complete: set[str],
    runtime: bool,
    baseline_owners: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, object]]:
    categories: dict[str, dict[str, object]] = {}
    for category in CATEGORIES:
        prefix = category + ":"
        owners = sorted(target[len(prefix) :] for target in targets if target.startswith(prefix))
        if runtime and category not in complete:
            categories[category] = {"mode": "full", "owners": []}
        else:
            categories[category] = {"mode": "selective", "owners": owners}
            if baseline_owners is not None:
                categories[category]["baseline_owners"] = baseline_owners.get(category, [])
    return categories


def expected_manifest(repo: Path, matrices: dict[str, list[dict]]) -> dict[str, object]:
    """Build the list of owners required for a complete full-run impact index."""
    categories: dict[str, list[str]] = {}
    for category in CATEGORIES:
        candidates = matrices.get(category, [])
        if not isinstance(candidates, list):
            raise ValueError(f"matrix {category} must be a list")
        if category.startswith("unit:"):
            owners = [candidate["bucket"] for candidate in candidates]
        else:
            owners = [f"{candidate['model']}/{candidate['test_case']}" for candidate in candidates]
        categories[category] = sorted(set(owners))
    return {"categories": categories, "recipe_digest": _recipe_digest(repo)}


def filter_matrix(selection: dict[str, object], category: str, candidates: list[dict]) -> dict:
    """Filter a GitHub matrix while preserving a placeholder for an empty result."""
    config = selection["categories"][category]
    if not isinstance(config, dict):
        raise ValueError(f"selection category {category} must be an object")
    if category.startswith("unit:"):
        candidate_owners = {candidate["bucket"] for candidate in candidates}
    else:
        candidate_owners = {
            f"{candidate['model']}/{candidate['test_case']}" for candidate in candidates
        }
    baseline = set(config.get("baseline_owners", candidate_owners))
    if config["mode"] == "full" or not candidate_owners.issubset(baseline):
        selected = candidates
    else:
        owners = set(config["owners"])
        if category.startswith("unit:"):
            selected = [candidate for candidate in candidates if candidate["bucket"] in owners]
        else:
            selected = [
                candidate
                for candidate in candidates
                if f"{candidate['model']}/{candidate['test_case']}" in owners
            ]
    placeholder = (
        {"bucket": "__no_tests__"}
        if category.startswith("unit:")
        else {"model": "__no_tests__", "test_case": "__no_tests__"}
    )
    return {"matrix": selected or [placeholder], "has_tests": bool(selected)}


def write_metadata(
    output: Path, kind: str, platform: str, owner: str, source_sha: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"kind": kind, "platform": platform, "owner": owner, "source_sha": source_sha},
            sort_keys=True,
        )
        + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("write-metadata")
    metadata.add_argument("--output", type=Path, required=True)
    metadata.add_argument("--kind", choices=("unit", "functional"), required=True)
    metadata.add_argument("--platform", choices=("h100", "gb200"), required=True)
    metadata.add_argument("--owner", required=True)
    metadata.add_argument("--source-sha", required=True)

    build = subparsers.add_parser("build-index")
    build.add_argument("--repo", type=Path, default=Path.cwd())
    build.add_argument("--artifacts", type=Path, required=True)
    build.add_argument("--expected", type=Path, required=True)
    build.add_argument("--source-sha", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--required-category", choices=CATEGORIES, action="append")

    select = subparsers.add_parser("select")
    select.add_argument("--repo", type=Path, default=Path.cwd())
    select.add_argument("--index", type=Path)
    select.add_argument("--base-sha", required=True)
    select.add_argument("--head-sha", required=True)
    select.add_argument("--force-full")
    select.add_argument("--output", type=Path, required=True)

    expected = subparsers.add_parser("expected-manifest")
    expected.add_argument("--repo", type=Path, default=Path.cwd())
    expected.add_argument("--matrices", type=Path, required=True)
    expected.add_argument("--output", type=Path, required=True)

    matrix = subparsers.add_parser("filter-matrix")
    matrix.add_argument("--selection", type=Path, required=True)
    matrix.add_argument("--category", choices=CATEGORIES, required=True)
    matrix.add_argument("--candidates", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "write-metadata":
        write_metadata(args.output, args.kind, args.platform, args.owner, args.source_sha)
    elif args.command == "build-index":
        index = build_index(
            args.repo,
            args.artifacts,
            args.expected,
            args.source_sha,
            args.required_category,
        )
        args.output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    elif args.command == "select":
        selection = select_tests(
            args.repo, args.index, args.base_sha, args.head_sha, args.force_full
        )
        args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    elif args.command == "expected-manifest":
        matrices = json.loads(args.matrices.read_text())
        result = expected_manifest(args.repo, matrices)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif args.command == "filter-matrix":
        selection = json.loads(args.selection.read_text())
        candidates = json.loads(args.candidates.read_text())
        result = filter_matrix(selection, args.category, candidates)
        args.output.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
