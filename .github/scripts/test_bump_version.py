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

import tempfile
import unittest
from pathlib import Path

import bump_version as bv

FILES = {
    "pyproject.toml": '[project]\nname = "earth2bufr"\nversion = "0.2.0a0"\n',
    "Cargo.toml": '[package]\nname = "earth2bufr"\nversion = "0.2.0-alpha.0"\n',
    "Cargo.lock": '[[package]]\nname = "other"\nversion = "0.2.0-alpha.0"\n\n'
    '[[package]]\nname = "earth2bufr"\nversion = "0.2.0-alpha.0"\n',
    "uv.lock": '[[package]]\nname = "other"\nversion = "0.2.0a0"\n\n'
    '[[package]]\nname = "earth2bufr"\nversion = "0.2.0a0"\n',
    "src/earth2bufr/__init__.py": '__version__ = "0.2.0a0"\n',
    "docs/conf.py": 'project = "earth2bufr"\nrelease = "0.2.0a0"\n',
}


class BumpVersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, text in FILES.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cargo_version(self):
        self.assertEqual(bv.cargo_version("0.2.0a0"), "0.2.0-alpha.0")
        self.assertEqual(bv.cargo_version("0.2.0rc1"), "0.2.0-rc.1")
        self.assertEqual(bv.cargo_version("0.2.0"), "0.2.0")
        with self.assertRaises(ValueError):
            bv.cargo_version("0.2.0.dev1")

    def test_plan_defaults(self):
        self.assertEqual(
            bv.plan("0.2.0a0"),
            {
                "current": "0.2.0a0",
                "release": "0.2.0",
                "rc_version": "0.2.0rc1",
                "rc_branch": "0.2.0-rc",
                "next_version": "0.3.0a0",
            },
        )

    def test_plan_options(self):
        result = bv.plan("1.4.2a0", rc_number=2, next_version="1.4.3a0")
        self.assertEqual(result["rc_version"], "1.4.2rc2")
        self.assertEqual(result["next_version"], "1.4.3a0")

    def test_plan_rejects_bad_input(self):
        for current in ("0.2.0", "0.2.0rc1", "0.2.0a1"):
            with self.assertRaises(ValueError):
                bv.plan(current)
        for nxt in ("0.2.0a0", "0.1.0a0", "0.3.0", "0.3.0rc1"):
            with self.assertRaises(ValueError):
                bv.plan("0.2.0a0", next_version=nxt)
        with self.assertRaises(ValueError):
            bv.plan("0.2.0a0", rc_number=0)

    def test_apply_rc_version(self):
        bv.apply_version(self.root, "0.2.0rc1")
        self.assertEqual(bv.current_version(self.root), "0.2.0rc1")
        self.assertIn('version = "0.2.0-rc.1"', (self.root / "Cargo.toml").read_text())
        self.assertIn('release = "0.2.0rc1"', (self.root / "docs/conf.py").read_text())
        self.assertIn('__version__ = "0.2.0rc1"', (self.root / "src/earth2bufr/__init__.py").read_text())

    def test_apply_only_touches_project_lock_entry(self):
        bv.apply_version(self.root, "0.3.0a0")
        for name in ("uv.lock", "Cargo.lock"):
            text = (self.root / name).read_text()
            self.assertIn('name = "other"\nversion = "0.2.0', text)
            self.assertEqual(text.count("0.3.0"), 1)

    def test_apply_fails_on_missing_entry(self):
        (self.root / "Cargo.toml").write_text("[package]\n")
        with self.assertRaises(ValueError):
            bv.apply_version(self.root, "0.3.0a0")


if __name__ == "__main__":
    unittest.main()
