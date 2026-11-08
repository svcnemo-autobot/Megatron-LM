"""Print the package versions that distinguish the two crossover arms (experiment only)."""

import importlib.metadata as md
import sys

for name in (
    "flashinfer-python",
    "nvidia-cudnn-frontend",
    "nccl-extensions",
    "cuda-core",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
    "transformer-engine",
    "torch",
):
    try:
        version = md.version(name)
        location = md.distribution(name).locate_file("")
    except md.PackageNotFoundError:
        version, location = None, None
    print(f"{name}=={version} @ {location}")
print("sys.path head:", sys.path[:3])
