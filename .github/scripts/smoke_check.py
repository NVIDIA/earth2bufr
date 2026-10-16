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

"""Smoke test a built artifact.

``wheel``: install the single wheel in ``--dist`` into a clean virtualenv and
import it from outside the source tree, checking the version, the Rust
extension, and the bundled BUFR tables.

``sdist``: check the single source distribution in ``--dist`` has the files
needed to rebuild.
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

TABLES = ("table_b.json", "table_d.json")

IMPORT_CHECK = """
import importlib.metadata as md
from pathlib import Path

import earth2bufr

expected = {expected!r}
assert md.version("earth2bufr") == expected, md.version("earth2bufr")
assert earth2bufr.__version__ == expected, earth2bufr.__version__
assert earth2bufr._get_lib() is not None, "Rust extension failed to load"
tables = Path(earth2bufr.__file__).parent / "tables"
for name in {tables!r}:
    assert (tables / name).is_file(), f"missing bundled table {{name}}"
assert earth2bufr._load_default_tables(), "default tables did not load"
print(f"smoke ok: earth2bufr {{expected}} from {{earth2bufr.__file__}}")
"""


def single(dist, pattern):
    """Return the only file matching ``pattern`` in ``dist``."""
    found = sorted(dist.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {dist}, found {[p.name for p in found]}")
    return found[0]


def check_wheel(dist, expected):
    """Install the wheel into a clean venv and import it."""
    wheel = single(dist, "*.whl")
    if not wheel.name.startswith(f"earth2bufr-{expected}-"):
        raise SystemExit(f"wheel {wheel.name} does not match version {expected}")
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", str(wheel.resolve())], check=True)
        # Run away from the checkout so the installed wheel is what gets imported
        subprocess.run([str(python), "-c", IMPORT_CHECK.format(expected=expected, tables=TABLES)], check=True, cwd=tmp)


def check_sdist(dist, expected):
    """Check the sdist name and contents."""
    sdist = single(dist, "*.tar.gz")
    if sdist.name != f"earth2bufr-{expected}.tar.gz":
        raise SystemExit(f"sdist {sdist.name} does not match version {expected}")
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    needed = ["Cargo.toml", "pyproject.toml", *(f"src/earth2bufr/tables/{name}" for name in TABLES)]
    for item in needed:
        if not any(name.endswith("/" + item) for name in names):
            raise SystemExit(f"sdist is missing {item}")
    print(f"smoke ok: {sdist.name}")


def main():
    """Run the requested check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wheel", "sdist"))
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--version", required=True, help="Expected package version")
    args = parser.parse_args()
    {"wheel": check_wheel, "sdist": check_sdist}[args.kind](args.dist, args.version)


if __name__ == "__main__":
    main()
