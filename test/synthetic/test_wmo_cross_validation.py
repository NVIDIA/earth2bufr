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
    build_code_flag_combined_operator_message,
    build_code_flag_operator_message,
    build_quality_bitmap_message,
    code_flag_combined_operator_case,
    code_flag_operator_case,
    compressed_numeric_case,
    delayed_replication_case,
    edition3_surface_case,
    missing_value_case,
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
