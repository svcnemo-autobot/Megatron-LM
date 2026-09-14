# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Helpers for safely extracting downloaded archives."""

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable


def _validated_destination(destination: str | Path, member_names: Iterable[str]) -> Path:
    """Resolve archive members and require every target to stay below destination."""
    destination = Path(destination).resolve()
    for member_name in member_names:
        member_path = Path(member_name)
        if member_path.is_absolute():
            raise ValueError(f"Archive member escapes destination: {member_name!r}")
        try:
            (destination / member_path).resolve().relative_to(destination)
        except ValueError as error:
            raise ValueError(f"Archive member escapes destination: {member_name!r}") from error
    return destination


def safe_extract_zip(archive: zipfile.ZipFile, destination: str | Path) -> None:
    """Extract regular ZIP members after validating their paths and types."""
    members = archive.infolist()
    destination = _validated_destination(destination, (member.filename for member in members))
    for member in members:
        file_type = stat.S_IFMT(member.external_attr >> 16)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Archive special entries are not allowed: {member.filename!r}")

    for member in members:
        target = destination / member.filename
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def safe_extract_tar(archive: tarfile.TarFile, destination: str | Path) -> None:
    """Extract regular tar members after validating their paths and types."""
    members = archive.getmembers()
    destination = _validated_destination(destination, (member.name for member in members))
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not allowed: {member.name!r}")
        if not (member.isdir() or member.isreg()):
            raise ValueError(f"Archive special entries are not allowed: {member.name!r}")

    for member in members:
        target = destination / member.name
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Archive member has no file data: {member.name!r}")
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
