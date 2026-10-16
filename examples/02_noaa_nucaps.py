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

"""Example: Reading NOAA NUCAPS atmospheric sounding BUFR files.

This example demonstrates how to read NUCAPS (NOAA Unique Combined
Atmospheric Processing System) BUFR files. NUCAPS produces retrieved
temperature and moisture profiles from CrIS and ATMS measurements on
the Suomi NPP and NOAA-20/21 satellites.

NUCAPS BUFR files are available from NOAA public S3 buckets:
- noaa-nesdis-snpp-pds   (Suomi NPP)
- noaa-nesdis-n20-pds    (NOAA-20 / JPSS-1)
- noaa-nesdis-n21-pds    (NOAA-21 / JPSS-2)

Path layout: s3://<bucket>/NUCAPS_C2211_BUFR/{YYYY}/{MM}/{DD}/*.bufr

Reference: https://www.nesdis.noaa.gov/current-satellite-missions/currently-flying/joint-polar-satellite-system/nucaps
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

import earth2bufr

# NOAA NUCAPS file on public S3 (Suomi NPP, Jan 15 2024, first granule)
NUCAPS_S3_URL = (
    "https://noaa-nesdis-snpp-pds.s3.amazonaws.com/"
    "NUCAPS_C2211_BUFR/2024/01/15/"
    "NUCAPS-C2211_v3r0_npp_s202401150000079_e202401150000377_c202401150145320.bufr"
)

NUCAPS_FILENAME = "NUCAPS-C2211_v3r0_npp_s202401150000079_e202401150000377_c202401150145320.bufr"


def download_nucaps_bufr(dest_path: Path) -> bool:
    """Download a sample NUCAPS BUFR file from NOAA S3."""
    print("Downloading sample NUCAPS BUFR file...")
    print(f"  URL: {NUCAPS_S3_URL}")
    print(f"  Destination: {dest_path}")

    try:
        urllib.request.urlretrieve(NUCAPS_S3_URL, dest_path)
        size_kb = dest_path.stat().st_size / 1024
        print(f"  Downloaded: {size_kb:.1f} KB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    """Demonstrate NUCAPS BUFR reading with earth2bufr."""
    # Path for sample data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    nucaps_file = data_dir / NUCAPS_FILENAME

    # Download if needed
    if not nucaps_file.exists() and not download_nucaps_bufr(nucaps_file):
        print("\nCannot proceed without sample file.")
        return

    print(f"\n{'=' * 60}")
    print("Example 1: Inspect file metadata")
    print("=" * 60)

    meta = earth2bufr.get_metadata(str(nucaps_file))
    print(f"File: {nucaps_file.name}")
    print(f"Messages: {len(meta.messages)}")
    print(f"Embedded DX table blocks: {sum(e.embedded_table_sha256 is not None for e in meta.table_epochs)}")

    print("\nMessage details (first 5):")
    for i, m in enumerate(meta.messages[:5]):
        print(
            f"  [{i}] Edition {m.edition}, "
            f"Category {m.data_category}, "
            f"Subsets: {m.subsets}, "
            f"Compressed: {m.compressed}, "
            f"Time: {m.year}-{m.month:02d}-{m.day:02d} "
            f"{m.hour:02d}:{m.minute:02d}"
        )

    print("\nDescriptors in first message:")
    for desc in meta.messages[0].descriptors:
        print(f"  {desc}")

    print(f"\n{'=' * 60}")
    print("Example 2: Decode all soundings")
    print("=" * 60)

    t0 = time.perf_counter()
    batches = earth2bufr.read_bufr(str(nucaps_file))
    t_decode = time.perf_counter() - t0

    total_rows = sum(b.num_rows for b in batches)
    print(f"Decoded {len(batches)} batch(es)")
    print(f"Total rows (soundings): {total_rows:,}")
    print(f"  Decode time: {t_decode:.3f}s")

    batch = batches[0]
    print(f"\nSchema ({batch.num_columns} columns):")
    for i, field in enumerate(batch.schema):
        type_str = str(field.type)
        if len(type_str) > 60:
            type_str = type_str[:57] + "..."
        print(f"  {field.name}: {type_str}")
        if i >= 19:
            print(f"  ... and {batch.num_columns - 20} more columns")
            break

    print(f"\n{'=' * 60}")
    print("Example 3: Explore scalar and replicated columns")
    print("=" * 60)

    # NUCAPS data has scalar columns (lat, lon, time, satellite info)
    # and replicated columns (vertical profile levels as List<Struct>)

    scalar_indices = []
    list_indices = []
    for i, field in enumerate(batch.schema):
        if pa.types.is_list(field.type):
            list_indices.append(i)
        else:
            scalar_indices.append(i)

    print(f"Scalar columns ({len(scalar_indices)}): geolocation, time, satellite info")
    for idx in scalar_indices[:15]:
        field = batch.schema.field(idx)
        arr = batch.column(idx)
        non_null = batch.num_rows - arr.null_count
        print(f"  {field.name}: {non_null}/{batch.num_rows} non-null")
    if len(scalar_indices) > 15:
        print(f"  ... and {len(scalar_indices) - 15} more")

    print(f"\nList columns ({len(list_indices)}): replicated vertical levels")
    for idx in list_indices[:5]:
        field = batch.schema.field(idx)
        arr = batch.column(idx)
        type_str = str(field.type)
        if len(type_str) > 50:
            type_str = type_str[:47] + "..."
        print(f"  {field.name}: {type_str}")
        # Show how many items in first non-null list
        for i in range(min(5, batch.num_rows)):
            if arr[i].is_valid:
                print(f"    First non-null row has {len(arr[i].as_py())} levels")
                break
    if len(list_indices) > 5:
        print(f"  ... and {len(list_indices) - 5} more")

    print(f"\n{'=' * 60}")
    print("Example 4: One row per vertical level")
    print("=" * 60)

    # A replicated group decodes to one List<Struct> column with one list per
    # sounding. pyarrow turns it into a long table with one row per level
    # without leaving Arrow: list_parent_indices gives each entry's sounding.
    batch = batches[0]
    list_index = next(i for i, field in enumerate(batch.schema) if pa.types.is_list(field.type))
    lists = batch.column(list_index)
    entries = lists.flatten()
    long = pa.Table.from_arrays(
        [pc.list_parent_indices(lists), *entries.flatten()],
        names=["sounding", *[field.name for field in entries.type]],
    )
    print(f"{batch.schema.field(list_index).name}: {batch.num_rows:,} soundings -> {long.num_rows:,} levels")
    print(long.slice(0, 5).to_pandas().to_string())

    print(f"\n{'=' * 60}")
    print("Example 5: Work with PyArrow directly (zero-copy)")
    print("=" * 60)

    batch = batches[0]
    print(f"RecordBatch: {batch.num_rows} rows x {batch.num_columns} columns")
    print(f"Memory usage: {batch.nbytes / 1024:.1f} KB")

    # Access columns by index to handle duplicate names safely
    col_names = batch.schema.names
    print(f"\nFirst 10 column names: {col_names[:10]}")

    # Find first occurrence of a column by name
    def get_col(name):
        """Get the first column matching the given name."""
        idx = col_names.index(name)
        return batch.column(idx)

    if "B001007" in col_names:
        sat_id = get_col("B001007")
        unique_sats = {v.as_py() for v in sat_id if v.is_valid}
        print(f"Satellite IDs (B001007): {unique_sats}")

    if "B005001" in col_names:
        lats = get_col("B005001")
        valid_lats = [v.as_py() for v in lats if v.is_valid]
        if valid_lats:
            print(f"Latitude range: [{min(valid_lats):.2f}, {max(valid_lats):.2f}]")

    if "B006001" in col_names:
        lons = get_col("B006001")
        valid_lons = [v.as_py() for v in lons if v.is_valid]
        if valid_lons:
            print(f"Longitude range: [{min(valid_lons):.2f}, {max(valid_lons):.2f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
