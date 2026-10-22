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

"""Synthetic WMO BUFR cross-validation against reference decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from . import adapters
from .assertions import assert_nested_rows_match, assert_reference_frames_match
from .builders import (
    BITMAP_MARKER_KEYS,
    bitmap_marker_value,
    build_bitmap_marker_message,
    build_code_flag_combined_operator_message,
    build_code_flag_operator_message,
    build_quality_bitmap_message,
    build_quality_section_message,
    code_flag_combined_operator_case,
    code_flag_operator_case,
    compressed_numeric_case,
    delayed_replication_case,
    edition3_surface_case,
    missing_value_case,
    quality_section_values,
    surface_observation_case,
)

pytestmark = [pytest.mark.unit, pytest.mark.rust]


def _decode_reference(library: str, decoder, *args):
    try:
        return decoder(*args)
    except adapters.ReferenceDecoderUnavailableError as exc:
        raise AssertionError(f"{library} reference decoder unavailable: {exc}") from exc


@pytest.mark.parametrize(
    "case_factory",
    [surface_observation_case, edition3_surface_case, missing_value_case, compressed_numeric_case],
    ids=lambda factory: factory().name,
)
def test_wmo_flat_cases_match_eccodes_and_pybufrkit(tmp_path: Path, case_factory):
    """Flat WMO fixtures decode to the same DataFrame in all scalar decoders.

    ecCodes is the gold-standard reference for these non-PrepBUFR cases.
    PyBufrKit is included as a second independent decoder because it exposes
    descriptor/value pairs in an easy-to-audit structure.
    """
    case = case_factory()
    path = case.write(tmp_path)
    expected = adapters.expected_frame(case)

    frames = {
        "earth2bufr": adapters.decode_earth2bufr_frame(path, case.fields),
        "eccodes": _decode_reference("ecCodes", adapters.decode_eccodes_frame, path, case.fields),
        "pybufrkit": _decode_reference("PyBufrKit", adapters.decode_pybufrkit_frame, path, case.fields),
    }

    assert_reference_frames_match(case, frames, expected)


def test_wmo_numeric_operators_exempt_code_and_flag_tables(tmp_path: Path):
    """Code/flag operator exemptions match ecCodes and explicit values."""
    case = code_flag_operator_case()
    path = tmp_path / f"{case.name}.bufr"
    path.write_bytes(build_code_flag_operator_message(case))
    expected = adapters.expected_frame(case)

    frames = {
        "earth2bufr": adapters.decode_earth2bufr_frame(path, case.fields),
        "eccodes": _decode_reference("ecCodes", adapters.decode_eccodes_frame, path, case.fields),
    }

    assert_reference_frames_match(case, frames, expected)


def test_wmo_combined_numeric_operator_exempts_code_and_flag_tables(tmp_path: Path):
    """Combined scale/width operator exemptions match ecCodes."""
    case = code_flag_combined_operator_case()
    path = tmp_path / f"{case.name}.bufr"
    path.write_bytes(build_code_flag_combined_operator_message(case))
    expected = adapters.expected_frame(case)

    frames = {
        "earth2bufr": adapters.decode_earth2bufr_frame(path, case.fields),
        "eccodes": _decode_reference("ecCodes", adapters.decode_eccodes_frame, path, case.fields),
    }

    assert_reference_frames_match(case, frames, expected)


def test_wmo_delayed_replication_nested_structure_matches_references(tmp_path: Path):
    """A delayed-replication level list matches ecCodes and PyBufrKit."""
    case = delayed_replication_case()
    path = case.write(tmp_path)
    expected = adapters.delayed_replication_expected_rows(case)

    rows_by_library = {
        "earth2bufr": adapters.decode_earth2bufr_delayed_replication(path),
        "eccodes": _decode_reference("ecCodes", adapters.decode_eccodes_delayed_replication, path),
        "pybufrkit": _decode_reference("PyBufrKit", adapters.decode_pybufrkit_delayed_replication, path),
    }

    for library, rows in rows_by_library.items():
        assert_nested_rows_match(case, library, rows, expected)


def test_wmo_quality_bitmap_operators_match_eccodes(tmp_path: Path):
    """Quality, bitmap and bitmap-reuse operators occupy no bits: values after them match ecCodes."""
    import earth2bufr

    path = tmp_path / "quality_bitmap.bufr"
    path.write_bytes(build_quality_bitmap_message())
    keys = ["year", "airTemperature", "dataPresentIndicator", "#1#airTemperature->percentConfidence", "windSpeed"]
    reference = _decode_reference("ecCodes", adapters.decode_eccodes_keys, path, keys)
    batch = next(iter(earth2bufr.read_bufr(path)))
    values = [(name, column[0].as_py()) for name, column in zip(batch.schema.names, batch.columns, strict=True)]

    assert reference == {
        "year": [2024],
        "airTemperature": [281.5],
        "dataPresentIndicator": [1, 0],
        "#1#airTemperature->percentConfidence": [70],
        "windSpeed": [3.5],
    }
    assert values[:7] == [
        ("B004001", 2024.0),
        ("B012101", 281.5),
        ("B031031", 1.0),
        ("B031031", 0.0),
        ("B033007", 70.0),
        ("B033007", 90.0),
        ("B011002", 3.5),
    ]


@pytest.mark.parametrize("compressed", [False, True], ids=["uncompressed", "compressed"])
@pytest.mark.parametrize("operator", sorted(BITMAP_MARKER_KEYS), ids=lambda operator: f"2-{operator:02d}-255")
def test_wmo_bitmap_marker_operators_match_eccodes(tmp_path: Path, operator: int, compressed: bool):
    """Marker operators with YYY=255 read the element the bitmap points at, as ecCodes does."""
    import earth2bufr

    path = tmp_path / "bitmap_marker.bufr"
    path.write_bytes(build_bitmap_marker_message(operator, compressed=compressed, num_subsets=2, passes=2))
    key = BITMAP_MARKER_KEYS[operator]
    reference = _decode_reference("ecCodes", adapters.decode_eccodes_keys, path, [key])[key]
    expected = [
        [bitmap_marker_value(operator, subset, repeat, marker) for repeat in range(2) for marker in range(2)]
        for subset in range(2)
    ]
    # ecCodes lists uncompressed values subset by subset and compressed values marker by marker.
    if compressed:
        reference_rows = [reference[subset::2] for subset in range(2)]
    else:
        reference_rows = [reference[subset * 4 : (subset + 1) * 4] for subset in range(2)]
    batch = next(iter(earth2bufr.read_bufr(path)))
    suffix = f"_2{operator:02d}255"
    markers = [name for name in batch.schema.names if name.endswith(suffix)]
    rows = [
        [batch.column(index)[subset].as_py() for index, name in enumerate(batch.schema.names) if name in markers]
        for subset in range(2)
    ]

    assert reference_rows == [pytest.approx(row) for row in expected]
    assert markers == [f"B012101{suffix}", f"B011002{suffix}"] * 2
    assert rows == [pytest.approx(row) for row in expected]


@pytest.mark.parametrize("compressed", [False, True], ids=["uncompressed", "compressed"])
def test_wmo_quality_section_with_bitmap_reuse_matches_eccodes(tmp_path: Path, compressed: bool):
    """Confidences, reused-bitmap statistics and a substituted value on a second bitmap match ecCodes,
    in strict mode and in both fixed replication layouts."""
    import pyarrow as pa

    import earth2bufr

    path = tmp_path / "quality_section.bufr"
    path.write_bytes(build_quality_section_message(compressed=compressed))
    expected = [quality_section_values(subset) for subset in range(2)]
    targets = ["#1#airTemperature", "#2#airTemperature", "#2#windSpeed"]
    keys = {
        "percentConfidence": [f"{target}->percentConfidence" for target in targets],
        "firstOrderStatisticalValue": [f"{target}->firstOrderStatisticalValue" for target in targets],
        "substitutedValue": ["#1#year->substitutedValue"],
    }
    reference = _decode_reference(
        "ecCodes", adapters.decode_eccodes_keys, path, [key for group in keys.values() for key in group]
    )
    # ecCodes attaches these values to their elements in the first subset only when uncompressed.
    for subset in range(2 if compressed else 1):
        for name, group in keys.items():
            assert [reference[key][subset] for key in group] == pytest.approx(expected[subset][name]), name

    suffixes = {"firstOrderStatisticalValue": "_224255", "substitutedValue": "_223255"}
    marker_names = {
        "flat": ["B012101_224255", "B012101_224255", "B011002_224255", "B004001_223255"],
        "structural": ["B012101_224255", "B012101_224255", "B011002_224255", "B004001_223255"],
    }
    for layout, names in marker_names.items():
        with earth2bufr.open_bufr(path, strict=True) as source:
            batch = pa.Table.from_batches(list(source.iter_batches(fixed_replication=layout)))
        markers = [name for name in batch.column_names if name.endswith(tuple(suffixes.values()))]
        assert markers == names, layout
        for key, suffix in suffixes.items():
            columns = [batch.column(i) for i, name in enumerate(batch.column_names) if name.endswith(suffix)]
            rows = [[column[subset].as_py() for column in columns] for subset in range(2)]
            assert rows == [pytest.approx(row[key]) for row in expected], (layout, key)
    flat = pa.Table.from_batches(earth2bufr.read_bufr(path, strict=True))
    confidences = [flat.column(i) for i, name in enumerate(flat.column_names) if name == "B033007"]
    assert [[column[subset].as_py() for column in confidences] for subset in range(2)] == [
        row["percentConfidence"] for row in expected
    ]


def test_satpy_adapter_reports_product_reader_limit_for_generic_wmo(tmp_path: Path):
    """SatPy BUFR readers are product-specific, not generic WMO decoders."""
    case = surface_observation_case()
    path = case.write(tmp_path)

    try:
        adapters.decode_satpy_frame(path)
    except adapters.ReferenceDecoderUnavailableError as exc:
        if "not installed" in str(exc):
            raise AssertionError(str(exc)) from exc
        assert "product-specific BUFR reader" in str(exc)
    else:
        pytest.fail("SatPy unexpectedly decoded a generic WMO fixture with a product-specific reader")
