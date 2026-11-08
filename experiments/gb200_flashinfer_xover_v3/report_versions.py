"""Print the runtime versions that bound one FlashInfer crossover trial (scratch, not for merge)."""

import importlib
import importlib.metadata as metadata
import sys

import torch

DISTRIBUTIONS = (
    "torch",
    "transformer-engine",
    "flashinfer-python",
    "flashinfer-cubin",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
)

print("VER python", sys.version.split()[0])
for dist in DISTRIBUTIONS:
    try:
        print("VER", dist, metadata.version(dist))
    except metadata.PackageNotFoundError:
        print("VER", dist, "absent")
for name in ("flashinfer", "flashinfer_cubin", "flashinfer.jit.env"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # Record the failure; the trial log carries the evidence.
        print("VER", name, "import-error", type(exc).__name__, exc)
        continue
    print("VER", name, getattr(module, "__version__", "-"), getattr(module, "__file__", "-"))
    if name == "flashinfer.jit.env":
        print("VER cubin_dir", getattr(module, "FLASHINFER_CUBIN_DIR", "-"))
if torch.cuda.is_available():
    print(
        "VER device",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0),
        torch.cuda.device_count(),
    )
