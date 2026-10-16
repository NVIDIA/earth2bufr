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

"""Example: Reading NOAA JPSS ATMS satellite radiance BUFR files.

This example demonstrates how to read Advanced Technology Microwave Sounder
(ATMS) BUFR files from NOAA's Joint Polar Satellite System (JPSS). ATMS is
a 22-channel microwave radiometer on the NOAA-20, NOAA-21, and Suomi NPP
satellites, used for temperature and moisture profiling of the atmosphere.

ATMS BUFR files are available from NOAA public S3 buckets:
- noaa-nesdis-n20-pds   (NOAA-20 / JPSS-1)
- noaa-nesdis-n21-pds   (NOAA-21 / JPSS-2)
- noaa-nesdis-snpp-pds  (Suomi NPP)

Path layout: s3://<bucket>/ATMS_BUFR/{YYYY}/{MM}/{DD}/*.bufr

Reference: https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/data/jpss_atms.py
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

import earth2bufr

# NOAA JPSS ATMS file on public S3 (NOAA-20, Jan 15 2024, first granule)
ATMS_S3_URL = (
    "https://noaa-nesdis-n20-pds.s3.amazonaws.com/"
    "ATMS_BUFR/2024/01/15/"
    "ATMS_v1r0_j01_s202401150000053_e202401150000369_c202401150028230.bufr"
)

ATMS_FILENAME = "ATMS_v1r0_j01_s202401150000053_e202401150000369_c202401150028230.bufr"


def download_atms_bufr(dest_path: Path) -> bool:
    """Download a sample ATMS BUFR file from NOAA S3."""
    print("Downloading sample ATMS BUFR file...")
    print(f"  URL: {ATMS_S3_URL}")
    print(f"  Destination: {dest_path}")

    try:
        urllib.request.urlretrieve(ATMS_S3_URL, dest_path)
        size_kb = dest_path.stat().st_size / 1024
        print(f"  Downloaded: {size_kb:.1f} KB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    """Demonstrate ATMS BUFR reading with earth2bufr."""
    # Path for sample data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    atms_file = data_dir / ATMS_FILENAME

    # Download if needed
    if not atms_file.exists() and not download_atms_bufr(atms_file):
        print("\nCannot proceed without sample file.")
        return

    print(f"\n{'=' * 60}")
    print("Example 1: Inspect file metadata")
    print("=" * 60)

    meta = earth2bufr.get_metadata(str(atms_file))
    print(f"File: {atms_file.name}")
    print(f"Messages: {len(meta.messages)}")
    print(f"Embedded DX table blocks: {sum(e.embedded_table_sha256 is not None for e in meta.table_epochs)}")

    print("\nMessage details:")
    for i, m in enumerate(meta.messages):
        print(
            f"  [{i}] Edition {m.edition}, "
            f"Category {m.data_category}, "
            f"Subsets: {m.subsets}, "
            f"Compressed: {m.compressed}, "
            f"Time: {m.year}-{m.month:02d}-{m.day:02d} "
            f"{m.hour:02d}:{m.minute:02d}"
        )

    print("\nDescriptors in first message:")
    for desc in meta.messages[0].descriptors[:10]:
        print(f"  {desc}")
    if len(meta.messages[0].descriptors) > 10:
        print(f"  ... and {len(meta.messages[0].descriptors) - 10} more")

    print(f"\n{'=' * 60}")
    print("Example 2: Decode all observations")
    print("=" * 60)

    t0 = time.perf_counter()
    batches = earth2bufr.read_bufr(str(atms_file))
    t_decode = time.perf_counter() - t0

    total_rows = sum(b.num_rows for b in batches)
    print(f"Decoded {len(batches)} batch(es)")
    print(f"Total rows (scan lines): {total_rows:,}")
    print(f"  Decode time: {t_decode:.3f}s")

    batch = batches[0]
    print(f"\nSchema ({batch.num_columns} columns):")
    for i, field in enumerate(batch.schema):
        print(f"  {field.name}: {field.type}")
        if i >= 14:
            print(f"  ... and {batch.num_columns - 15} more columns")
            break

    print(f"\n{'=' * 60}")
    print("Example 3: Extract geolocation and brightness temperatures")
    print("=" * 60)

    # Convert to pandas for convenient display
    df = batch.to_pandas()

    # Key ATMS columns (WMO mnemonics):
    #   B005001 = Latitude (high accuracy)
    #   B006001 = Longitude (high accuracy)
    #   B004001-B004006 = Year, Month, Day, Hour, Minute, Second
    #   B010001 = Height of station (altitude)
    #   B002019 = Satellite instruments (instrument ID)
    #   B001007 = Satellite identifier
    print(f"Total columns: {len(df.columns)}")
    print(f"Rows (scan lines): {len(df)}")

    # Show scalar columns (non-list) for overview
    scalar_cols = [c for c in df.columns if not str(df[c].dtype).startswith("object")]
    print(f"\nScalar columns ({len(scalar_cols)}):")
    for col in scalar_cols[:15]:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(df)} non-null, dtype={df[col].dtype}")

    print(f"\n{'=' * 60}")
    print("Example 4: One row per channel")
    print("=" * 60)

    # A replicated group decodes to one List<Struct> column with one list per
    # scan position. pyarrow turns it into a long table with one row per channel
    # without leaving Arrow: list_parent_indices gives each entry's scan position.
    batch = batches[0]
    list_index = next(i for i, field in enumerate(batch.schema) if pa.types.is_list(field.type))
    lists = batch.column(list_index)
    entries = lists.flatten()
    long = pa.Table.from_arrays(
        [pc.list_parent_indices(lists), *entries.flatten()],
        names=["scan position", *[field.name for field in entries.type]],
    )
    print(f"{batch.schema.field(list_index).name}: {batch.num_rows:,} scan positions -> {long.num_rows:,} channels")
    print(long.slice(0, 5).to_pandas().to_string())

    print(f"\n{'=' * 60}")
    print("Example 5: Work with PyArrow directly (zero-copy)")
    print("=" * 60)

    # For large-scale processing, stay in PyArrow for zero-copy operations
    batch = batches[0]
    print(f"RecordBatch: {batch.num_rows} rows x {batch.num_columns} columns")
    print(f"Memory usage: {batch.nbytes / 1024:.1f} KB")

    # Access columns by name
    col_names = batch.schema.names
    print(f"\nAll column names: {col_names}")

    # Get a specific column as a PyArrow array
    if "B001007" in col_names:
        sat_id = batch.column("B001007")
        print(f"\nSatellite ID (B001007): {sat_id[0].as_py()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
