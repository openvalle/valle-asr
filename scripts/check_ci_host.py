"""Reject emulated or cross-target CI before declaring native platform support."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    expected, runner_arch = sys.argv[1:]
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    host = next(line.removeprefix("host: ") for line in version.splitlines() if line.startswith("host: "))
    expected_arch = "ARM64" if expected.startswith("aarch64-") else "X64"
    if host != expected or runner_arch != expected_arch:
        raise SystemExit(f"Native runner required: expected {expected}/{expected_arch}, got {host}/{runner_arch}")
    directory = Path("artifacts")
    directory.mkdir(exist_ok=True)
    (directory / "runner.json").write_text(json.dumps({
        "target": expected, "rust_host": host, "runner_arch": runner_arch,
        "native": True, "rustc": version,
    }, indent=2) + "\n")
    print(f"Verified native {host} runner ({runner_arch})")


if __name__ == "__main__":
    main()
