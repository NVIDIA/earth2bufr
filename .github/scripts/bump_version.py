# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan and apply project version changes for the release workflow.

``plan`` derives the release candidate (``X.Y.ZrcN`` on branch ``X.Y.Z-rc``)
and the next development version (``X.Y.Za0``) from the alpha version on
``main`` and prints them as ``key=value`` lines. ``apply`` rewrites every
version entry in the project to a given Python (PEP 440) version, converting
it to the equivalent semver form for Cargo.
"""

import argparse
import re
from pathlib import Path

DEV_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)a0")
PY_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:(a|rc)(\d+))?")

# file -> (pattern whose first group ends right before the version, uses Cargo form)
VERSION_ENTRIES = {
    "pyproject.toml": (r'(\[project\][\s\S]*?\nversion = ")[^"]+', False),
    "Cargo.toml": (r'(\[package\][\s\S]*?\nversion = ")[^"]+', True),
    "Cargo.lock": (r'(\[\[package\]\]\nname = "earth2bufr"\nversion = ")[^"]+', True),
    "uv.lock": (r'(\[\[package\]\]\nname = "earth2bufr"\nversion = ")[^"]+', False),
    "src/earth2bufr/__init__.py": (r'(__version__ = ")[^"]+', False),
    "docs/conf.py": (r'(\nrelease = ")[^"]+', False),
}


def cargo_version(version):
    """Convert a PEP 440 version (0.2.0a0, 0.2.0rc1) to semver (0.2.0-alpha.0, 0.2.0-rc.1)."""
    match = PY_VERSION.fullmatch(version)
    if not match:
        raise ValueError(f"Unsupported version: {version}")
    base = ".".join(match.groups()[:3])
    kind, number = match.group(4), match.group(5)
    if kind is None:
        return base
    return f"{base}-{'alpha' if kind == 'a' else 'rc'}.{number}"


def current_version(root):
    """Read the project version from pyproject.toml."""
    text = (root / "pyproject.toml").read_text()
    return re.search(r'\[project\][\s\S]*?\nversion = "([^"]+)"', text).group(1)


def plan(current, rc_number=1, next_version=None):
    """Return the release plan for an alpha ``current`` version."""
    match = DEV_VERSION.fullmatch(current)
    if not match:
        raise ValueError(f"main must carry an X.Y.Za0 version, found {current}")
    major, minor, patch = (int(g) for g in match.groups())
    release = f"{major}.{minor}.{patch}"
    if next_version is None:
        next_version = f"{major}.{minor + 1}.0a0"
    next_match = DEV_VERSION.fullmatch(next_version)
    if not next_match:
        raise ValueError(f"Next version must have the form X.Y.Za0, found {next_version}")
    if tuple(int(g) for g in next_match.groups()) <= (major, minor, patch):
        raise ValueError(f"Next version {next_version} must be greater than {release}")
    if rc_number < 1:
        raise ValueError("RC number must be at least 1")
    return {
        "current": current,
        "release": release,
        "rc_version": f"{release}rc{rc_number}",
        "rc_branch": f"{release}-rc",
        "next_version": next_version,
    }


def apply_version(root, version):
    """Rewrite each version entry under ``root`` to ``version``."""
    updated = {}
    for name, (pattern, cargo) in VERSION_ENTRIES.items():
        value = cargo_version(version) if cargo else version
        text, count = re.subn(pattern, lambda m, v=value: m.group(1) + v, (root / name).read_text())
        if count != 1:
            raise ValueError(f"Expected one version entry in {name}, found {count}")
        updated[name] = text
    for name, text in updated.items():
        (root / name).write_text(text)


def main():
    """Run the plan or apply command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="Print the release plan as key=value lines")
    plan_parser.add_argument("--rc-number", type=int, default=1)
    plan_parser.add_argument("--next-version", default="")
    apply_parser = sub.add_parser("apply", help="Rewrite the project files to a version")
    apply_parser.add_argument("version")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(current_version(args.root), args.rc_number, args.next_version or None)
        for key, value in result.items():
            print(f"{key}={value}")
    else:
        apply_version(args.root, args.version)


if __name__ == "__main__":
    main()
