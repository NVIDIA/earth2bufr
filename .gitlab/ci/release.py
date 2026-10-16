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

"""RC branch preparation for GitLab.

Wheel build, Artifactory upload, and KitMaker publishing live in modulus-ci
(.gitlab-ci/earth2bufr).
"""

import argparse
import base64
import os
import re
import subprocess
from pathlib import Path


def required(name):
    """Read a nonempty CI setting."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in GitLab CI/CD variables or the manual job form")
    return value


def prepare_rc(root, version):
    """Synchronize Python, Rust, and locked project versions for an RC branch."""
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)rc([1-9]\d*)", version)
    if not match:
        raise ValueError("EARTH2BUFR_RC_VERSION must have the form 0.2.0rc1")
    base = ".".join(match.groups()[:3])
    rust_version = f"{base}-rc.{match.group(4)}"
    replacements = {
        "pyproject.toml": (r'(\[project\][\s\S]*?\nversion = ")[^"]+', version),
        "Cargo.toml": (r'(\[package\][\s\S]*?\nversion = ")[^"]+', rust_version),
        "uv.lock": (r'(\[\[package\]\]\nname = "earth2bufr"\nversion = ")[^"]+', version),
        "src/earth2bufr/__init__.py": (r'(__version__ = ")[^"]+', version),
    }
    updated = {}
    for name, (pattern, value) in replacements.items():
        text, count = re.subn(pattern, lambda m, v=value: m.group(1) + v, (root / name).read_text())
        if count != 1:
            raise ValueError(f"Expected one version entry in {name}, found {count}")
        updated[name] = text
    for name, text in updated.items():
        (root / name).write_text(text)
    return f"{base}-rc"


def create_rc():
    """Commit synchronized manifests on a new branch and push to this GitLab project."""
    token = required("EARTH2BUFR_RELEASE_TOKEN")
    version = required("EARTH2BUFR_RC_VERSION")
    email = required("RELEASE_BOT_EMAIL")
    remote = f"{required('CI_SERVER_URL')}/{required('CI_PROJECT_PATH')}.git"
    branch = prepare_rc(Path.cwd(), version)
    auth = base64.b64encode(f"oauth2:{token}".encode()).decode()
    # Supply credentials through the environment, keeping them out of URLs and argv.
    env = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
    }
    existing = subprocess.run(
        ["git", "ls-remote", "--heads", remote, f"refs/heads/{branch}"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if existing.stdout.strip():
        raise ValueError(f"RC branch already exists: {branch}")
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "pyproject.toml", "Cargo.toml", "uv.lock", "src/earth2bufr/__init__.py"], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Earth2BUFR Release Bot",
            "-c",
            f"user.email={email}",
            "commit",
            "--signoff",
            "-m",
            f"Bump version to {version}",
        ],
        check=True,
    )
    subprocess.run(["git", "push", remote, f"HEAD:refs/heads/{branch}"], env=env, check=True)


def main():
    """Dispatch a release operation from the CI job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create-rc"])
    parser.parse_args()
    create_rc()


if __name__ == "__main__":
    main()
