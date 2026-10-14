"""Run a pipeline twice and confirm it writes byte-identical outputs."""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_determinism(cmd, output_name, runs=2, timeout=None):
    """Run `cmd` (a list, with "{out}" as the output-dir placeholder) `runs` times.

    Each run gets a fresh output directory. Returns a dict with the sha256 of
    every run and whether they all match.
    """
    digests = []
    for _ in range(runs):
        out = Path(tempfile.mkdtemp(prefix="det_"))
        try:
            argv = [a.replace("{out}", str(out)) for a in cmd]
            subprocess.run(argv, check=True, timeout=timeout)
            digests.append(_digest(out / output_name))
        finally:
            shutil.rmtree(out, ignore_errors=True)
    return {"digests": digests, "deterministic": len(set(digests)) == 1}
