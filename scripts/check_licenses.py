"""Fail closed on unresolved/non-permissive SPDX licenses in the resolved graph."""
import json
import pathlib
import re
import subprocess
import sys

ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib",
           "Unicode-3.0", "Unicode-DFS-2016", "Unlicense", "CC0-1.0", "OpenSSL",
           "CDLA-Permissive-2.0", "0BSD", "BSL-1.0", "MIT-0", "Apache-2.0-with-LLVM-exception"}

def permissive(expression):
    if not expression:
        return False
    expression = expression.replace("/", " OR ")
    expression = expression.replace("Apache-2.0 WITH LLVM-exception", "Apache-2.0-with-LLVM-exception")
    # Cargo SPDX expressions have AND/OR/parentheses. Each allowed identifier is
    # replaced by a boolean, preserving SPDX's choice/intersection semantics.
    tokens = re.findall(r"[A-Za-z0-9.+-]+|[()]", expression)
    values = []
    for token in tokens:
        if token in {"AND", "OR"}:
            values.append(token.lower())
        elif token in {"(", ")"}:
            values.append(token)
        else:
            values.append(str(token in ALLOWED))
    try:
        return bool(eval(" ".join(values), {"__builtins__": {}}, {}))
    except (SyntaxError, TypeError):
        return False

assert permissive("MIT/Apache-2.0")
assert permissive("Apache-2.0 WITH LLVM-exception OR MIT")
assert not permissive("GPL-3.0-only")
assert not permissive("MIT AND GPL-3.0-only")
assert not permissive(None)
packages_by_id = {}
for target in ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "aarch64-apple-darwin"):
    metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", target]))
    resolved = {node["id"] for node in metadata["resolve"]["nodes"]}
    packages_by_id.update((p["id"], p) for p in metadata["packages"] if p["id"] in resolved)
packages = sorted(packages_by_id.values(), key=lambda p: (p["name"], p["version"]))
report = [{"name": p["name"], "version": p["version"], "license": p["license"], "repository": p["repository"]} for p in packages]
rejected = [p for p in report if not permissive(p["license"])]
if rejected:
    print(json.dumps(rejected, indent=2), file=sys.stderr)
    sys.exit("Dependency license is not on the permissive allowlist")
if "--write" in sys.argv:
    pathlib.Path("third_party/crates.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Permissive license choices verified for {len(report)} crates across Linux, Windows and macOS")
