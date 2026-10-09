"""Validate a release tag, its independent CI runs and the registry checksum."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

WORKFLOWS = (
    "ci-linux.yml", "ci-linux-arm64.yml",
    "ci-windows.yml", "ci-windows-arm64.yml",
    "ci-macos.yml", "ci-macos-x86_64.yml",
)


def request(url, token=None):
    headers = {"User-Agent": "valle-crate-release", "Accept": "application/json"}
    if token:
        headers.update(Authorization=f"Bearer {token}", Accept="application/vnd.github+json")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def passed_runs(runs, sha):
    selected = []
    for filename in WORKFLOWS:
        matches = [run for run in runs
                   if run["path"].split("@", 1)[0] == ".github/workflows/" + filename
                   and run["head_sha"] == sha and run["head_branch"] == "main"
                   and run["event"] == "push"]
        latest = max(matches, key=lambda run: run["id"], default=None)
        if latest is None or latest["status"] != "completed" or latest["conclusion"] != "success":
            raise RuntimeError(f"{filename} has not passed at release commit {sha}; complete CI first")
        selected.append(latest)
    return selected


def output(name, value):
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write(f"{name}={value}\n")
    print(f"{name}: {value}")


def check_ci(package):
    tag = os.environ["RELEASE_TAG"]
    if tag != "v" + package["version"]:
        raise RuntimeError(f"Release tag {tag} does not match Cargo version {package['version']}")
    repo = os.environ["GITHUB_REPOSITORY"]
    if package["repository"] != "https://github.com/" + repo:
        raise RuntimeError("Checkout repository does not match Cargo.toml")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tagged = subprocess.check_output(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], text=True).strip()
    if tagged != sha:
        raise RuntimeError("Checkout does not match the release tag")
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "refs/remotes/origin/main"], check=True)
    query = urllib.parse.urlencode({"head_sha": sha, "event": "push", "branch": "main", "per_page": 100})
    data = request(f"https://api.github.com/repos/{repo}/actions/runs?{query}", os.environ["GITHUB_TOKEN"])
    for run in passed_runs(data["workflow_runs"], sha):
        print(f"Verified {run['path']}: {run['html_url']}")
    output("sha", sha)
    output("name", package["name"])
    output("version", package["version"])


def registry_version(package):
    url = f"https://crates.io/api/v1/crates/{package['name']}/{package['version']}"
    try:
        return request(url)["version"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def check_registry(package, confirm=False):
    archive = Path("target/package") / f"{package['name']}-{package['version']}.crate"
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    for attempt in range(24 if confirm else 1):
        version = registry_version(package)
        if version is not None:
            if version["checksum"] != checksum:
                raise RuntimeError("This version exists on crates.io with a different archive; use a new version")
            output("publish_needed", "false")
            print(f"Verified published {package['name']} {package['version']}: SHA-256 {checksum}")
            return
        if not confirm:
            output("publish_needed", "true")
            print(f"Archive SHA-256: {checksum}")
            return
        if attempt < 23:
            time.sleep(5)
    raise RuntimeError("Uploaded version is not yet visible on crates.io; check registry state before retrying")


def main():
    with Path("Cargo.toml").open("rb") as stream:
        package = tomllib.load(stream)["package"]
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    if mode == "ci":
        check_ci(package)
    elif mode in ("registry", "confirm"):
        check_registry(package, confirm=mode == "confirm")
    else:
        raise SystemExit("Usage: check_release.py ci|registry|confirm")


if __name__ == "__main__":
    main()
