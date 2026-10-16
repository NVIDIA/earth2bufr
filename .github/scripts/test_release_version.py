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

import release_version as rv

FILES = {
    "pyproject.toml": '[project]\nname = "earth2bufr"\nversion = "1.0.0rc1"\n',
    "Cargo.toml": '[package]\nname = "earth2bufr"\nversion = "1.0.0-rc.1"\n',
    "uv.lock": '[[package]]\nname = "earth2bufr"\nversion = "1.0.0rc1"\n',
    "src/earth2bufr/__init__.py": '__version__ = "1.0.0rc1"\n',
}


class ReleaseVersionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, text in FILES.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rc_branch_drops_suffix(self):
        self.assertEqual(rv.release_version("1.0.0-rc"), "1.0.0")

    def test_other_refs_are_unchanged(self):
        for ref in ("main", "ngeneva/feature", "1.0.0", "1.0.0-rc2", "v1.0.0-rc"):
            self.assertIsNone(rv.release_version(ref))

    def test_apply_rewrites_every_file(self):
        rv.apply_version(self.root, "1.0.0")
        for name in FILES:
            text = (self.root / name).read_text()
            self.assertIn('"1.0.0"', text)
            self.assertNotIn("rc", text.replace("earth2bufr", ""))

    def test_apply_fails_on_missing_entry(self):
        (self.root / "Cargo.toml").write_text("[package]\n")
        with self.assertRaises(ValueError):
            rv.apply_version(self.root, "1.0.0")

    def test_current_version(self):
        self.assertEqual(rv.current_version(self.root), "1.0.0rc1")


if __name__ == "__main__":
    unittest.main()
