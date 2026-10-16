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

"""Example: Reading NOAA GDAS PrepBUFR files.

This example demonstrates how to read PrepBUFR files from NOAA's Global Data
Assimilation System (GDAS). These files contain conventional observations used
in numerical weather prediction.

PrepBUFR files are available from:
- NOAA NOMADS: https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/
- AWS S3: s3://noaa-reanalyses-pds/observations/reanalysis/conv/prepbufr/
"""

from __future__ import annotations

import time
import urllib.request
from collections import Counter
from pathlib import Path

import earth2bufr


def download_sample_prepbufr(dest_path: Path) -> bool:
    """Download a sample GDAS PrepBUFR file from AWS S3."""
    url = (
        "https://noaa-reanalyses-pds.s3.amazonaws.com/"
        "observations/reanalysis/conv/prepbufr/2024/01/prepbufr/"
        "gdas.20240101.t00z.prepbufr.nr"
    )
    print("Downloading sample PrepBUFR file...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    """Demonstrate PrepBUFR reading with earth2bufr."""
    # Path for sample data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    prepbufr_file = data_dir / "gdas.20240101.t00z.prepbufr.nr"

    # Download if needed
    if not prepbufr_file.exists() and not download_sample_prepbufr(prepbufr_file):
        print("\nCannot proceed without sample file.")
        return

    print(f"\n{'=' * 60}")
    print("Example 1: Inspect message headers")
    print("=" * 60)

    info = earth2bufr.get_metadata(prepbufr_file)
    print(f"messages: {len(info.messages)}, with data: {len(info.data_messages)}")
    print(f"embedded DX table blocks: {sum(e.embedded_table_sha256 is not None for e in info.table_epochs)}")

    print(f"\n{'=' * 60}")
    print("Example 2: List available data categories")
    print("=" * 60)

    subsets = Counter()
    for message in info.data_messages:
        subsets[message.data_category] += message.subsets
    print(f"{'Code':>4}  {'Subsets':>9s}")
    print("-" * 60)
    for category, count in sorted(subsets.items()):
        print(f"{category:4d}  {count:>9,}")

    print(f"\n{'=' * 60}")
    print("Example 3: Read surface land observations (ADPSFC)")
    print("=" * 60)

    # read_bufr returns one RecordBatch per table epoch, data category and run
    # of one descriptor template, so a category can span several batches.
    t0 = time.perf_counter()
    batches = earth2bufr.read_bufr(
        str(prepbufr_file),
        data_category_filter=[109],  # ADPSFC only
    )
    t_decode = time.perf_counter() - t0
    batch = batches[0]
    print(f"Batches: {len(batches)}; first batch rows: {batch.num_rows:,}")
    print(f"Columns: {batch.num_columns}")
    print(f"  Decode time: {t_decode:.3f}s")

    # Convert to pandas for display
    t0 = time.perf_counter()
    df_surface = batch.to_pandas()
    t_pandas = time.perf_counter() - t0
    print(f"  Pandas conversion: {t_pandas:.3f}s")

    print("\nFirst 10 column names:")
    for col in list(df_surface.columns)[:10]:
        print(f"  - {col}")

    print("\nSample data (first 3 rows, selected columns):")
    key_cols = ["_data_category", *[c for c in ["SID", "XOB", "YOB", "DHR", "ELV", "TYP"] if c in df_surface.columns]]
    print(df_surface[key_cols].head(3).to_string())

    print(f"\n{'=' * 60}")
    print("Example 4: Read aircraft observations (AIRCFT)")
    print("=" * 60)

    t0 = time.perf_counter()
    batches = earth2bufr.read_bufr(
        str(prepbufr_file),
        data_category_filter=[104],  # AIRCFT only
    )
    t_decode = time.perf_counter() - t0
    batch = batches[0]
    print(f"Aircraft observations: {batch.num_rows:,} rows")
    print(f"  Decode time: {t_decode:.3f}s")

    print(f"\n{'=' * 60}")
    print("Example 5: One row per level with flatten=True")
    print("=" * 60)

    # flatten=True expands the PrepBUFR level list (the group carrying CAT) into
    # one row per level, repeating the report's scalars; other lists keep their
    # first entry. Upper-air reports (ADPUPA) have many levels.
    t0 = time.perf_counter()
    results = earth2bufr.read_bufr(
        str(prepbufr_file),
        data_category_filter=[102],
        flatten=True,
    )
    t_flatten = time.perf_counter() - t0
    levels = results[0]
    print(f"Upper-air levels: {levels.num_rows:,} rows, {levels.num_columns} columns")
    print(f"  Decode + flatten time: {t_flatten:.3f}s")
    print(f"Columns: {levels.schema.names}")
    print("\nSample levels (first 5 rows):")
    level_cols = [c for c in levels.schema.names if c in ["SID", "XOB", "YOB", "POB", "TOB", "CAT"]]
    if level_cols:
        print(levels.select(level_cols).slice(0, 5).to_pandas().to_string())

    print(f"\n{'=' * 60}")
    print("Example 6: Read multiple categories")
    print("=" * 60)

    # Each category returns at least one RecordBatch, one per template run
    t0 = time.perf_counter()
    batches = earth2bufr.read_bufr(
        str(prepbufr_file),
        data_category_filter=[109, 110],  # Surface land + marine
    )
    t_decode = time.perf_counter() - t0
    print(f"Batches returned: {len(batches)}")
    print(f"  Decode time: {t_decode:.3f}s")
    print("\nRows by category:")
    for batch in batches:
        cat = int(batch.column("_data_category")[0].as_py())
        print(f"  {cat}: {batch.num_rows:,} rows")

    print(f"\n{'=' * 60}")
    print("Example 7: Read all categories")
    print("=" * 60)

    t0 = time.perf_counter()
    batches = earth2bufr.read_bufr(str(prepbufr_file))
    t_decode = time.perf_counter() - t0
    total_rows = sum(b.num_rows for b in batches)
    categories = {int(b.column("_data_category")[0].as_py()) for b in batches}
    print(f"Batches: {len(batches)} across {len(categories)} categories")
    print(f"Total rows: {total_rows:,}")
    print(f"  Decode time: {t_decode:.3f}s")
    for batch in batches:
        cat = int(batch.column("_data_category")[0].as_py())
        print(f"  {cat}: {batch.num_rows:,} rows")

    print("\nDone!")


if __name__ == "__main__":
    main()
