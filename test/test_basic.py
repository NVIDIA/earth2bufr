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

"""Unit tests for earth2bufr."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import pytest
from synthetic.builders import compressed_numeric_case, surface_observation_case

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_import():
    """Test that the package can be imported."""
    import earth2bufr

    assert earth2bufr.__version__ == version("earth2bufr")


@pytest.mark.unit
def test_exports():
    """The public API is exactly these names."""
    import earth2bufr

    assert set(earth2bufr.__all__) == {
        "__version__",
        "BatchIterator",
        "BufrFile",
        "DecodeFailure",
        "FileInfo",
        "ListValueFilter",
        "MessageInfo",
        "TableEpoch",
        "TableRedefinition",
        "create_local_tables",
        "flatten_batch",
        "get_metadata",
        "get_num_threads",
        "load_tables",
        "open_bufr",
        "read_bufr",
        "set_num_threads",
    }
    assert all(hasattr(earth2bufr, name) for name in earth2bufr.__all__)


@pytest.mark.unit
def test_load_tables():
    """Test loading default tables."""
    import earth2bufr

    table_b, table_d = earth2bufr.load_tables()

    # Should be valid JSON strings
    import json

    b_data = json.loads(table_b)
    d_data = json.loads(table_d)

    # Should have some entries
    assert len(b_data) > 0
    assert len(d_data) > 0

    # Check table B entry structure
    first_key = next(iter(b_data.keys()))
    entry = b_data[first_key]
    assert "mnemonic" in entry
    assert "name" in entry
    assert "units" in entry
    assert "scale" in entry
    assert "reference_value" in entry
    assert "bit_width" in entry


@pytest.mark.unit
def test_create_local_tables():
    """Test creating local table overrides."""
    import json

    import earth2bufr

    local = earth2bufr.create_local_tables(
        table_b_overrides={
            "0-01-192": {
                "mnemonic": "CUSTOM",
                "name": "Custom descriptor",
                "units": "NUMERIC",
                "scale": 0,
                "reference_value": 0,
                "bit_width": 16,
            }
        },
        table_d_overrides={
            "3-60-001": {
                "mnemonic": "CUSTOM_SEQ",
                "descriptors": [1001, 1002],
            }
        },
    )

    data = json.loads(local)
    assert "table_b" in data
    assert "table_d" in data
    assert "0-01-192" in data["table_b"]
    assert "3-60-001" in data["table_d"]


@pytest.mark.rust
def test_file_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    import earth2bufr

    with pytest.raises(FileNotFoundError):
        earth2bufr.read_bufr("/nonexistent/path/to/file.bufr")


@pytest.mark.rust
def test_metadata_file_not_found():
    """Test that get_metadata raises FileNotFoundError for missing files."""
    import earth2bufr

    with pytest.raises(FileNotFoundError):
        earth2bufr.get_metadata("/nonexistent/path/to/file.bufr")


@pytest.mark.rust
def test_invalid_bufr_file(tmp_path: Path):
    """read_bufr raises ValueError on a file without BUFR messages."""
    import earth2bufr

    path = tmp_path / "not_bufr.bufr"
    path.write_bytes(b"This is not a BUFR file")
    with pytest.raises(ValueError):
        earth2bufr.read_bufr(path)


@pytest.mark.rust
def test_read_bufr_returns_dataframe(tmp_path: Path):
    """Test that read_bufr with as_pandas=True returns DataFrame-like object."""
    import earth2bufr

    case = surface_observation_case()
    path = case.write(tmp_path)
    frames = earth2bufr.read_bufr(path, as_pandas=True)

    assert len(frames) == 1
    assert len(frames[0]) == len(case.rows)
    assert frames[0]["B004001"].tolist() == [2024.0, 2024.0]
    assert frames[0]["_data_category"].tolist() == [float(case.data_category)] * len(case.rows)


@pytest.mark.rust
def test_descriptor_expansion(tmp_path: Path):
    """Test that message descriptors are exposed in F-X-Y order."""
    import earth2bufr

    case = surface_observation_case()
    path = case.write(tmp_path)

    assert earth2bufr.get_metadata(path).messages[0].descriptors == tuple(
        field.descriptor_name for field in case.fields
    )


@pytest.mark.rust
def test_compressed_data_decode(tmp_path: Path):
    """Test decoding of compressed BUFR scalar data."""
    import earth2bufr

    case = compressed_numeric_case()
    path = case.write(tmp_path)
    frame = earth2bufr.read_bufr(path, as_pandas=True)[0]

    assert earth2bufr.get_metadata(path).messages[0].compressed is True
    assert frame["B012101"].tolist() == [280.0, 280.25, 280.5]
    assert frame["B011002"].tolist() == [1.0, 1.1, 1.2]


@pytest.mark.rust
def test_uncompressed_data_decode(tmp_path: Path):
    """Test decoding of uncompressed BUFR scalar data."""
    import earth2bufr

    case = surface_observation_case()
    path = case.write(tmp_path)
    frame = earth2bufr.read_bufr(path, as_pandas=True)[0]

    assert earth2bufr.get_metadata(path).messages[0].compressed is False
    assert frame["B005001"].tolist() == [35.12345, -12.5]
    assert frame["B006001"].tolist() == [-97.12345, 130.25]


@pytest.mark.rust
def test_projection_keeps_data_category(tmp_path: Path):
    """A projected read keeps the requested fields and _data_category."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)
    batch = earth2bufr.read_bufr(path, projection=["B005001"])[0]

    assert batch.schema.names == ["B005001", "_data_category"]


@pytest.mark.rust
def test_projection_of_unknown_field_raises(tmp_path: Path):
    """A projected field the template lacks raises."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)

    with pytest.raises(ValueError, match="B999999"):
        earth2bufr.read_bufr(path, projection=["B999999"])


@pytest.mark.rust
def test_flatten_requires_flat_layout(tmp_path: Path):
    """flatten with the structural layout raises before decoding."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)

    with pytest.raises(ValueError, match="flatten=True requires fixed_replication='flat'"):
        earth2bufr.read_bufr(path, flatten=True, fixed_replication="structural")


@pytest.mark.rust
def test_data_category_filter_no_match(tmp_path: Path):
    """Test filtering with no matching categories."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)

    with pytest.raises(ValueError):
        earth2bufr.read_bufr(path, data_category_filter=[99])
