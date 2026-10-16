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

"""Deterministic tiny BUFR fixtures for parser-mode tests.

These tests generate very small BUFR Edition 4 messages in-place. The point is
not to mimic an operational product; it is to make each parser mode visible and
fully deterministic. Every raw bit value written below has an expected decoded
value asserted in the test.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
from synthetic.builders import BitWriter, bufr_message, desc, desc_name

pytestmark = [pytest.mark.unit, pytest.mark.rust]

if TYPE_CHECKING:
    from pathlib import Path

_bufr = partial(bufr_message, data_category=7)


def _write_bufr(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _rows(record_batch) -> list[dict]:
    columns = record_batch.to_pydict()
    return [dict(zip(columns.keys(), values, strict=True)) for values in zip(*columns.values(), strict=True)]


def _local_tables() -> str:
    import earth2bufr

    table_b = {
        desc_name(0, 31, 1): {
            "mnemonic": "LEVEL_COUNT",
            "name": "Synthetic outer delayed-replication count",
            "units": "NUMERIC",
            "scale": 0,
            "reference_value": 0,
            "bit_width": 8,
        },
        desc_name(0, 31, 2): {
            "mnemonic": "OBS_COUNT",
            "name": "Synthetic inner delayed-replication count",
            "units": "NUMERIC",
            "scale": 0,
            "reference_value": 0,
            "bit_width": 4,
        },
        desc_name(0, 63, 1): {
            "mnemonic": "SYN_TEMP",
            "name": "Synthetic temperature",
            "units": "K",
            "scale": 1,
            "reference_value": 2500,
            "bit_width": 8,
        },
        desc_name(0, 63, 2): {
            "mnemonic": "SYN_ID",
            "name": "Synthetic station id",
            "units": "CCITT IA5",
            "scale": 0,
            "reference_value": 0,
            "bit_width": 32,
        },
        desc_name(0, 63, 3): {
            "mnemonic": "SYN_FLAG",
            "name": "Synthetic flag",
            "units": "CODE TABLE",
            "scale": 0,
            "reference_value": 0,
            "bit_width": 4,
        },
        desc_name(0, 63, 4): {
            "mnemonic": "LEVEL_PRESSURE",
            "name": "Synthetic level pressure",
            "units": "hPa",
            "scale": 0,
            "reference_value": 800,
            "bit_width": 8,
        },
        desc_name(0, 63, 5): {
            "mnemonic": "LEVEL_TEMP",
            "name": "Synthetic level temperature",
            "units": "C",
            "scale": 1,
            "reference_value": 200,
            "bit_width": 8,
        },
        desc_name(0, 63, 6): {
            "mnemonic": "SYN_MISSING",
            "name": "Synthetic nullable value",
            "units": "NUMERIC",
            "scale": 0,
            "reference_value": 0,
            "bit_width": 4,
        },
        desc_name(0, 63, 7): {
            "mnemonic": "CAT",
            "name": "Synthetic PrepBUFR level category marker",
            "units": "CODE TABLE",
            "scale": 0,
            "reference_value": 0,
            "bit_width": 4,
        },
    }
    table_d = {
        desc_name(3, 63, 1): {
            "mnemonic": "SYN_TOP_LEVEL",
            "descriptors": [desc(0, 63, 1), desc(0, 63, 2), desc(0, 63, 3)],
        }
    }
    return earth2bufr.create_local_tables(table_b_overrides=table_b, table_d_overrides=table_d)


def test_synthetic__flat_uncompressed__decodes_sequence_scalars_strings_and_missing(tmp_path: Path):
    """A tiny Table-D sequence expands into readable scalar/string columns."""
    import earth2bufr

    descriptors = [desc(3, 63, 1), desc(0, 63, 6)]

    bits = BitWriter()
    bits.write_bits(31, 8)  # SYN_TEMP: (2500 + 31) / 10 = 253.1
    bits.write_ascii("ABCD", 4)
    bits.write_bits(3, 4)
    bits.write_bits(7, 4)  # ordinary value, not missing
    bits.write_bits(82, 8)  # SYN_TEMP: (2500 + 82) / 10 = 258.2
    bits.write_ascii("WXYZ", 4)
    bits.write_bits(5, 4)
    bits.write_bits(15, 4)  # all bits set: missing for a non-class-31 descriptor

    path = _write_bufr(
        tmp_path,
        "synthetic_flat_uncompressed.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=2),
    )

    message = earth2bufr.get_metadata(path).messages[0]
    assert (message.edition, message.center, message.sub_center, message.data_category) == (4, 98, 0, 7)
    assert (message.subsets, message.compressed) == (2, False)
    assert (message.year, message.month, message.day, message.hour, message.minute) == (2024, 1, 2, 3, 4)
    assert message.descriptors == ("3-63-001", "0-63-006")

    batches = earth2bufr.read_bufr(path, local_tables_json=_local_tables())
    assert len(batches) == 1
    rows = _rows(batches[0])

    assert rows[0]["SYN_TEMP"] == pytest.approx(253.1)
    assert rows[0]["SYN_ID"] == "ABCD"
    assert rows[0]["SYN_FLAG"] == 3.0
    assert rows[0]["SYN_MISSING"] == 7.0
    assert rows[0]["_data_category"] == 7.0

    assert rows[1]["SYN_TEMP"] == pytest.approx(258.2)
    assert rows[1]["SYN_ID"] == "WXYZ"
    assert rows[1]["SYN_FLAG"] == 5.0
    assert rows[1]["SYN_MISSING"] is None
    assert rows[1]["_data_category"] == 7.0


def test_synthetic__mixed_templates_in_one_category__return_source_order_groups(tmp_path: Path):
    """Mixed templates split in source order while zero-subset messages are ignored."""
    import earth2bufr

    first = BitWriter()
    first.write_bits(31, 8)  # SYN_TEMP: 253.1 K

    second = BitWriter()
    second.write_ascii("DIFF", 4)
    second.write_bits(3, 4)

    third = BitWriter()
    third.write_bits(42, 8)  # SYN_TEMP: 254.2 K

    path = _write_bufr(
        tmp_path,
        "synthetic_mixed_templates.bufr",
        b"".join(
            [
                _bufr([desc(0, 63, 2)], b"", num_subsets=0),
                _bufr([desc(0, 63, 1)], first.to_bytes(), num_subsets=1),
                _bufr(
                    [desc(0, 63, 2), desc(0, 63, 3)],
                    second.to_bytes(),
                    num_subsets=1,
                ),
                _bufr([desc(0, 63, 1)], third.to_bytes(), num_subsets=1),
            ]
        ),
    )

    batches = earth2bufr.read_bufr(path, local_tables_json=_local_tables())

    assert [batch.schema.names for batch in batches] == [
        ["SYN_TEMP", "_data_category"],
        ["SYN_ID", "SYN_FLAG", "_data_category"],
        ["SYN_TEMP", "_data_category"],
    ]
    assert [_rows(batch) for batch in batches] == [
        [{"SYN_TEMP": pytest.approx(253.1), "_data_category": 7.0}],
        [{"SYN_ID": "DIFF", "SYN_FLAG": 3.0, "_data_category": 7.0}],
        [{"SYN_TEMP": pytest.approx(254.2), "_data_category": 7.0}],
    ]


def test_synthetic__nested_delayed_replication__decodes_human_readable_tree(tmp_path: Path):
    """A deterministic nested delayed-replication tree decodes to List<Struct>."""
    import earth2bufr

    descriptors = [
        desc(0, 63, 2),  # SYN_ID
        desc(1, 4, 0),  # delayed replication: four following raw descriptors
        desc(0, 31, 1),  # LEVEL_COUNT
        desc(0, 63, 4),  # LEVEL_PRESSURE
        desc(1, 1, 0),  # nested delayed replication: one following descriptor
        desc(0, 31, 2),  # OBS_COUNT
        desc(0, 63, 5),  # LEVEL_TEMP
    ]

    bits = BitWriter()
    bits.write_ascii("NEST", 4)
    bits.write_bits(2, 8)  # two outer levels

    bits.write_bits(0, 8)  # LEVEL_PRESSURE: 800 + 0 = 800 hPa
    bits.write_bits(2, 4)  # two temperatures at this level
    bits.write_bits(10, 8)  # LEVEL_TEMP: (200 + 10) / 10 = 21.0 C
    bits.write_bits(20, 8)  # LEVEL_TEMP: (200 + 20) / 10 = 22.0 C

    bits.write_bits(50, 8)  # LEVEL_PRESSURE: 800 + 50 = 850 hPa
    bits.write_bits(1, 4)  # one temperature at this level
    bits.write_bits(30, 8)  # LEVEL_TEMP: (200 + 30) / 10 = 23.0 C

    path = _write_bufr(
        tmp_path,
        "synthetic_nested_delayed_replication.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=1),
    )

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables())[0])

    assert rows == [
        {
            "SYN_ID": "NEST",
            "LEVEL_COUNT": [
                {
                    "LEVEL_PRESSURE": 800.0,
                    "OBS_COUNT": [{"LEVEL_TEMP": 21.0}, {"LEVEL_TEMP": 22.0}],
                },
                {
                    "LEVEL_PRESSURE": 850.0,
                    "OBS_COUNT": [{"LEVEL_TEMP": 23.0}],
                },
            ],
            "_data_category": 7.0,
        }
    ]


def test_synthetic__compressed_constant_scalars__decodes_all_subsets(tmp_path: Path):
    """Compressed mode with zero increments repeats each reference value."""
    import earth2bufr

    descriptors = [desc(0, 63, 1), desc(0, 63, 2), desc(0, 63, 3)]

    bits = BitWriter()
    bits.write_bits(42, 8)  # SYN_TEMP reference: (2500 + 42) / 10 = 254.2
    bits.write_bits(0, 6)  # zero diff bits: same value for every subset
    bits.write_ascii("SAME", 4)
    bits.write_bits(0, 6)  # zero byte increment: same string for every subset
    bits.write_bits(6, 4)  # SYN_FLAG reference
    bits.write_bits(0, 6)  # zero diff bits: same value for every subset

    path = _write_bufr(
        tmp_path,
        "synthetic_compressed_constant_scalars.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=3, compressed=True),
    )

    message = earth2bufr.get_metadata(path).messages[0]
    assert message.compressed is True
    assert message.subsets == 3

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables())[0])
    assert rows == [
        {"SYN_TEMP": pytest.approx(254.2), "SYN_ID": "SAME", "SYN_FLAG": 6.0, "_data_category": 7.0},
        {"SYN_TEMP": pytest.approx(254.2), "SYN_ID": "SAME", "SYN_FLAG": 6.0, "_data_category": 7.0},
        {"SYN_TEMP": pytest.approx(254.2), "SYN_ID": "SAME", "SYN_FLAG": 6.0, "_data_category": 7.0},
    ]


def test_synthetic__operator_character_descriptor__decodes_inline_text(tmp_path: Path):
    """Operator 2-05-YYY creates a fixed-width inline character field."""
    import earth2bufr

    descriptors = [desc(2, 5, 4), desc(0, 63, 1)]

    bits = BitWriter()
    bits.write_ascii("TEXT", 4)
    bits.write_bits(42, 8)  # SYN_TEMP: (2500 + 42) / 10 = 254.2

    path = _write_bufr(
        tmp_path,
        "synthetic_operator_character_descriptor.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=1),
    )

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables())[0])
    assert rows == [{"CHAR_DATA": "TEXT", "SYN_TEMP": pytest.approx(254.2), "_data_category": 7.0}]


# ─────────────────────────────────────────────────────────────────────────────
# Flatten coverage: only CAT lists expand into rows.
#
# read_bufr(flatten=True) row-expands ONLY a CAT-bearing multi-element
# list<struct> (the PrepBUFR vertical-level replication). Any other
# multi-element list<struct> must NOT expand into rows (that would duplicate
# observations) and subsets whose optional non-CAT list is empty must NOT be
# dropped (that would silently lose single-level stations).
# ─────────────────────────────────────────────────────────────────────────────


def test_flatten__cat_multi_level_list__expands_one_row_per_level(tmp_path: Path):
    """A multi-element CAT list expands into one flat row per level."""
    import earth2bufr

    descriptors = [
        desc(0, 63, 2),  # SYN_ID (broadcast scalar)
        desc(1, 2, 0),  # delayed replication of the next two descriptors
        desc(0, 31, 1),  # LEVEL_COUNT (delayed count; names the list column)
        desc(0, 63, 7),  # CAT (marks this list as the vertical-level replication)
        desc(0, 63, 4),  # LEVEL_PRESSURE
    ]

    bits = BitWriter()
    bits.write_ascii("STAT", 4)
    bits.write_bits(2, 8)  # two levels
    bits.write_bits(1, 4)  # CAT = 1
    bits.write_bits(0, 8)  # LEVEL_PRESSURE: 800 + 0 = 800
    bits.write_bits(2, 4)  # CAT = 2
    bits.write_bits(50, 8)  # LEVEL_PRESSURE: 800 + 50 = 850

    path = _write_bufr(
        tmp_path,
        "flatten_cat_multi_level.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=1),
    )

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables(), flatten=True)[0])

    assert rows == [
        {"SYN_ID": "STAT", "CAT": 1.0, "LEVEL_PRESSURE": 800.0, "_data_category": 7.0},
        {"SYN_ID": "STAT", "CAT": 2.0, "LEVEL_PRESSURE": 850.0, "_data_category": 7.0},
    ]


def test_flatten__non_cat_multi_element_list__does_not_multiply_rows(tmp_path: Path):
    """A multi-element non-CAT list is scalarized (element 0), not row-expanded."""
    import earth2bufr

    descriptors = [
        desc(0, 63, 2),  # SYN_ID
        desc(1, 1, 0),  # delayed replication of the next descriptor
        desc(0, 31, 2),  # OBS_COUNT (delayed count; inner struct has NO CAT field)
        desc(0, 63, 4),  # LEVEL_PRESSURE
    ]

    bits = BitWriter()
    bits.write_ascii("STAT", 4)
    bits.write_bits(2, 4)  # two elements in the non-CAT list
    bits.write_bits(0, 8)  # LEVEL_PRESSURE: 800
    bits.write_bits(50, 8)  # LEVEL_PRESSURE: 850

    path = _write_bufr(
        tmp_path,
        "flatten_non_cat_multi.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=1),
    )

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables(), flatten=True)[0])

    # One subset in, one row out (no multiplication); element 0 is kept.
    assert rows == [{"SYN_ID": "STAT", "LEVEL_PRESSURE": 800.0, "_data_category": 7.0}]


def test_flatten__empty_optional_non_cat_list__row_retained(tmp_path: Path):
    """A subset whose optional non-CAT list is empty is retained, not dropped."""
    import earth2bufr

    descriptors = [
        desc(0, 63, 2),  # SYN_ID
        desc(1, 1, 0),  # delayed replication of the next descriptor
        desc(0, 31, 2),  # OBS_COUNT (non-CAT optional group)
        desc(0, 63, 4),  # LEVEL_PRESSURE
    ]

    bits = BitWriter()
    bits.write_ascii("AAAA", 4)  # subset 0: one element
    bits.write_bits(1, 4)
    bits.write_bits(0, 8)  # LEVEL_PRESSURE: 800
    bits.write_ascii("BBBB", 4)  # subset 1: empty optional list
    bits.write_bits(0, 4)

    path = _write_bufr(
        tmp_path,
        "flatten_empty_optional_list.bufr",
        _bufr(descriptors, bits.to_bytes(), num_subsets=2),
    )

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables(), flatten=True)[0])

    # Both subsets survive; the empty-list subset is kept with a null value.
    assert len(rows) == 2
    assert [row["SYN_ID"] for row in rows] == ["AAAA", "BBBB"]
    assert rows[0]["LEVEL_PRESSURE"] == 800.0
    assert rows[1]["LEVEL_PRESSURE"] is None


def test_flatten__promoted_name_collision__gets_numbered_suffix(tmp_path: Path):
    """A promoted list field whose name is already a column is renamed with a _2 suffix."""
    import earth2bufr

    descriptors = [
        desc(0, 63, 4),  # LEVEL_PRESSURE
        desc(1, 1, 0),  # delayed replication of the next descriptor
        desc(0, 31, 2),  # OBS_COUNT
        desc(0, 63, 4),  # LEVEL_PRESSURE inside the list
    ]
    bits = BitWriter()
    bits.write_bits(10, 8)  # LEVEL_PRESSURE: 810
    bits.write_bits(1, 4)  # one list element
    bits.write_bits(50, 8)  # LEVEL_PRESSURE: 850

    path = _write_bufr(tmp_path, "flatten_collision.bufr", _bufr(descriptors, bits.to_bytes(), num_subsets=1))

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_local_tables(), flatten=True)[0])
    assert rows == [{"LEVEL_PRESSURE": 810.0, "LEVEL_PRESSURE_2": 850.0, "_data_category": 7.0}]


def test_flatten__list_empty_in_every_row__promotes_null_columns(tmp_path: Path):
    """A list column with no entries in any row becomes null columns of its field types."""
    import earth2bufr

    descriptors = [
        desc(0, 63, 2),  # SYN_ID
        desc(1, 1, 0),
        desc(0, 31, 2),  # OBS_COUNT list, empty in every row
        desc(0, 63, 4),  # LEVEL_PRESSURE
    ]
    bits = BitWriter()
    for station in ("AAAA", "BBBB"):
        bits.write_ascii(station, 4)
        bits.write_bits(0, 4)  # no OBS_COUNT entries

    path = _write_bufr(tmp_path, "flatten_all_empty.bufr", _bufr(descriptors, bits.to_bytes(), num_subsets=2))

    flat = earth2bufr.read_bufr(path, local_tables_json=_local_tables(), flatten=True)[0]
    assert str(flat.schema.field("LEVEL_PRESSURE").type) == "double"
    assert _rows(flat) == [
        {"SYN_ID": "AAAA", "LEVEL_PRESSURE": None, "_data_category": 7.0},
        {"SYN_ID": "BBBB", "LEVEL_PRESSURE": None, "_data_category": 7.0},
    ]


def test_synthetic__64_bit_element__all_ones_is_null(tmp_path: Path):
    """A 64-bit element decodes all ones as missing and other codes as values."""
    import earth2bufr

    local = earth2bufr.create_local_tables(
        table_b_overrides={
            desc_name(0, 63, 40): {
                "mnemonic": "WIDE",
                "name": "Synthetic 64-bit value",
                "units": "NUMERIC",
                "scale": 0,
                "reference_value": 0,
                "bit_width": 64,
            }
        }
    )
    bits = BitWriter()
    bits.write_bits((1 << 64) - 1, 64)
    bits.write_bits(5, 64)

    path = _write_bufr(tmp_path, "wide.bufr", _bufr([desc(0, 63, 40)], bits.to_bytes(), num_subsets=2))

    assert earth2bufr.read_bufr(path, local_tables_json=local)[0].column("WIDE").to_pylist() == [None, 5.0]


# WMO Table C width, scale and combined operators (2-01, 2-02, 2-07) do not apply
# to code- or flag-table descriptors, including ones from local_tables_json.


def _code_flag_local_tables() -> str:
    import earth2bufr

    return earth2bufr.create_local_tables(
        table_b_overrides={
            desc_name(0, 63, 30): {
                "mnemonic": "LOC_CODE",
                "name": "Local code-table value",
                "units": "CODE TABLE",
                "scale": 0,
                "reference_value": 0,
                "bit_width": 4,
            },
            desc_name(0, 63, 31): {
                "mnemonic": "LOC_FLAG",
                "name": "Local flag-table value",
                "units": "FLAG TABLE",
                "scale": 0,
                "reference_value": 0,
                "bit_width": 4,
            },
            desc_name(0, 63, 32): {
                "mnemonic": "LOC_NUM",
                "name": "Local numeric value",
                "units": "m",
                "scale": 0,
                "reference_value": 0,
                "bit_width": 7,
            },
        }
    )


@pytest.mark.parametrize(
    ("operators", "cancels", "numeric_width"),
    [
        # 2-01-131 adds 3 bits and 2-02-129 one scale digit.
        pytest.param([desc(2, 1, 131), desc(2, 2, 129)], [desc(2, 2, 0), desc(2, 1, 0)], 7 + 3, id="2-01_2-02"),
        # 2-07-001 adds one scale digit and ceil(10 / 3) = 4 bits.
        pytest.param([desc(2, 7, 1)], [desc(2, 7, 0)], 7 + 4, id="2-07"),
    ],
)
def test_synthetic__local_code_flag__exempt_from_numeric_operators(
    tmp_path: Path, operators: list[int], cancels: list[int], numeric_width: int
):
    """Width and scale operators span local code/flag/numeric fields; only the numeric field is modified."""
    import earth2bufr

    fields = [desc(0, 63, 30), desc(0, 63, 31), desc(0, 63, 32)]  # LOC_CODE, LOC_FLAG, LOC_NUM
    descriptors = [*operators, *fields, *cancels]
    bits = BitWriter()
    bits.write_bits(5, 4)  # LOC_CODE = 5 at its native 4 bits
    bits.write_bits(3, 4)  # LOC_FLAG = 3 at its native 4 bits
    bits.write_bits(123, numeric_width)  # LOC_NUM: 123 / 10^1 = 12.3

    path = _write_bufr(tmp_path, "local_code_flag_operator.bufr", _bufr(descriptors, bits.to_bytes(), num_subsets=1))

    rows = _rows(earth2bufr.read_bufr(path, local_tables_json=_code_flag_local_tables())[0])
    assert rows == [{"LOC_CODE": 5.0, "LOC_FLAG": 3.0, "LOC_NUM": pytest.approx(12.3), "_data_category": 7.0}]
