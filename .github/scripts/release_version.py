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

"""Print the version a build of the current ref should publish.

On a release candidate branch (``X.Y.Z-rc``) the project files carry
``X.Y.ZrcN`` (``X.Y.Z-rc.N`` for Cargo). Wheels built from that branch are
the final release, so ``--apply`` rewrites every version entry to ``X.Y.Z``
before the build. Other refs are left untouched.
"""

import argparse
import re
from pathlib import Path

RC_BRANCH = re.compile(r"(\d+\.\d+\.\d+)-rc")

# file -> pattern whose first group ends right before the version string
VERSION_ENTRIES = {
    "pyproject.toml": r'(\[project\][\s\S]*?\nversion = ")[^"]+',
    "Cargo.toml": r'(\[package\][\s\S]*?\nversion = ")[^"]+',
    "uv.lock": r'(\[\[package\]\]\nname = "earth2bufr"\nversion = ")[^"]+',
    "src/earth2bufr/__init__.py": r'(__version__ = ")[^"]+',
}


def release_version(ref_name):
    """Return the stripped release version for an RC branch, else None."""
    match = RC_BRANCH.fullmatch(ref_name)
    return match.group(1) if match else None


def apply_version(root, version):
    """Rewrite each version entry under ``root`` to ``version``."""
    updated = {}
    for name, pattern in VERSION_ENTRIES.items():
        text, count = re.subn(pattern, lambda m: m.group(1) + version, (root / name).read_text())
        if count != 1:
            raise ValueError(f"Expected one version entry in {name}, found {count}")
        updated[name] = text
    for name, text in updated.items():
        (root / name).write_text(text)


def current_version(root):
    """Read the project version from pyproject.toml."""
    return re.search(r'\[project\][\s\S]*?\nversion = "([^"]+)"', (root / "pyproject.toml").read_text()).group(1)


def main():
    """Apply the RC version rule and print the resulting version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_name", help="Branch or tag name being built")
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--apply", action="store_true", help="Rewrite the project files")
    args = parser.parse_args()
    version = release_version(args.ref_name)
    if version is None:
        print(current_version(args.root))
        return
    if args.apply:
        apply_version(args.root, version)
    print(version)


if __name__ == "__main__":
    main()
