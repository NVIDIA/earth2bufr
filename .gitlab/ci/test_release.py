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

"""Offline release automation tests."""

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import release


class ReleaseTests(unittest.TestCase):
    """Exercise release boundaries without network calls."""

    def test_rc_updates_python_rust_and_lock(self):
        """PEP 440 and Cargo use different RC spellings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/earth2bufr").mkdir(parents=True)
            (root / "pyproject.toml").write_text('[project]\nname = "earth2bufr"\nversion = "0.1.0"\n')
            (root / "Cargo.toml").write_text('[package]\nname = "earth2bufr"\nversion = "0.1.0"\n')
            (root / "uv.lock").write_text(
                '[[package]]\nname = "earth2bufr"\nversion = "0.1.0"\n[[package]]\nname = "other"\nversion = "0.1.0"\n'
            )
            (root / "src/earth2bufr/__init__.py").write_text('__version__ = "0.1.0"\n')
            branch = release.prepare_rc(root, "0.2.0rc1")
            self.assertEqual(branch, "0.2.0-rc")
            self.assertIn('version = "0.2.0-rc.1"', (root / "Cargo.toml").read_text())
            self.assertEqual(tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"], "0.2.0rc1")
            self.assertEqual(tomllib.loads((root / "uv.lock").read_text())["package"][0]["version"], "0.2.0rc1")
            self.assertIn('__version__ = "0.2.0rc1"', (root / "src/earth2bufr/__init__.py").read_text())
            self.assertIn('name = "other"\nversion = "0.1.0"', (root / "uv.lock").read_text())

    def test_invalid_rc_rejected(self):
        """Reject final versions and branch-name injection."""
        for version in ["0.2.0", "../../bad", "0.2.0rc1;bad"]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                release.prepare_rc(Path("unused"), version)

    def test_rc_push_targets_gitlab_without_token_in_arguments(self):
        """Push only the new RC branch using environment-based authentication."""
        settings = {
            "EARTH2BUFR_RELEASE_TOKEN": "secret-token",
            "EARTH2BUFR_RC_VERSION": "0.2.0rc1",
            "RELEASE_BOT_EMAIL": "bot@example.com",
            "CI_SERVER_URL": "https://gitlab.example.com",
            "CI_PROJECT_PATH": "group/earth2bufr",
        }
        with (
            patch.dict(release.os.environ, settings),
            patch.object(release, "prepare_rc", return_value="0.2.0-rc"),
            patch.object(release.subprocess, "run", return_value=Mock(stdout="")) as run,
        ):
            release.create_rc()
            command = run.call_args.args[0]
            self.assertEqual(
                command,
                ["git", "push", "https://gitlab.example.com/group/earth2bufr.git", "HEAD:refs/heads/0.2.0-rc"],
            )
            self.assertNotIn("secret-token", str(run.call_args_list[0].args))
            self.assertIn("GIT_CONFIG_VALUE_0", run.call_args.kwargs["env"])

    def test_existing_rc_branch_is_not_pushed(self):
        """Refuse to update a previously created branch."""
        settings = {
            "EARTH2BUFR_RELEASE_TOKEN": "secret-token",
            "EARTH2BUFR_RC_VERSION": "0.2.0rc1",
            "RELEASE_BOT_EMAIL": "bot@example.com",
            "CI_SERVER_URL": "https://gitlab.example.com",
            "CI_PROJECT_PATH": "group/earth2bufr",
        }
        with (
            patch.dict(release.os.environ, settings),
            patch.object(release, "prepare_rc", return_value="0.2.0-rc"),
            patch.object(release.subprocess, "run", return_value=Mock(stdout="sha refs/heads/0.2.0-rc")) as run,
            self.assertRaises(ValueError),
        ):
            release.create_rc()
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
