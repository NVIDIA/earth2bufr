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

"""Synthetic PrepBUFR DX-table validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from . import adapters
from .assertions import assert_frame_matches
from .builders import (
    LEGACY_MISSING_STRING,
    build_embedded_code_flag_operator_message,
    build_ia5_missing_prepbufr,
    build_two_epoch_prepbufr_messages,
    synthetic_prepbufr_dx_case,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.unit, pytest.mark.rust]


def test_synthetic_prepbufr_dx_table_decodes_local_sequence(tmp_path: Path):
    """A tiny DX table defines local descriptors that read_bufr decodes."""
    import pyarrow as pa

    import earth2bufr

    case = synthetic_prepbufr_dx_case()
    path = case.write(tmp_path)

    info = earth2bufr.get_metadata(path)
    assert [message.data_category for message in info.messages] == [11, case.data_category]
    assert [message.subsets for message in info.data_messages] == [1]

    batches = earth2bufr.read_bufr(path)
    actual = pa.Table.from_batches(batches).to_pandas()[case.columns]
    expected = adapters.expected_prepbufr_frame(case)

    assert len(batches) == 1
    assert_frame_matches(case, "earth2bufr.read_bufr", actual, expected)

    overrides = earth2bufr.create_local_tables(
        table_b_overrides={
            "0-63-002": {
                "mnemonic": "CUSTOM_TOB",
                "name": "Caller-defined temperature observation",
                "units": "K",
                "scale": 1,
                "reference_value": 2500,
                "bit_width": 12,
            }
        }
    )
    overridden = pa.Table.from_batches(earth2bufr.read_bufr(path, local_tables_json=overrides)).to_pandas()

    assert "TOB" not in overridden
    assert overridden["CUSTOM_TOB"].tolist() == [pytest.approx(273.1)]
    assert overridden["_data_category"].tolist() == [case.data_category]


def _epoch_rows(batches) -> list[dict]:
    """Flatten decoded batches into per-row dicts keyed by column name."""
    rows = []
    for batch in batches:
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            rows.append({name: values[index] for name, values in columns.items()})
    return rows


@pytest.mark.parametrize("reader", ["read_bufr", "open_bufr_iter_batches"])
def test_two_dx_epochs__each_decodes_under_its_own_table(tmp_path: Path, reader: str):
    """A redefined descriptor must be read under the table in force at that point.

    Both epochs use sequence 3-63-001, but 0-63-002 is TOB/12-bit in the first
    DX block and WSP/8-bit in the second. Merging the blocks file-wide makes one
    epoch read the wrong width and mis-align.
    """
    import earth2bufr

    path = tmp_path / "two_epochs.prepbufr"
    path.write_bytes(build_two_epoch_prepbufr_messages())

    if reader == "read_bufr":
        rows = _epoch_rows(earth2bufr.read_bufr(path))
    else:
        with earth2bufr.open_bufr(path) as source:
            rows = _epoch_rows(source.iter_batches())

    assert len(rows) == 2, "each epoch contributes exactly one subset"
    first = next(row for row in rows if row["SID"] == "KDEN")
    second = next(row for row in rows if row["SID"] == "KSEA")

    # The first epoch keeps the 12-bit TOB meaning even though a later block
    # redefines the same descriptor.
    assert first["TOB"] == pytest.approx(273.1)
    assert first.get("WSP") is None

    # The second epoch sees only its own 8-bit WSP definition.
    assert second["WSP"] == pytest.approx(42.0)
    assert second.get("TOB") is None


@pytest.mark.parametrize(
    ("raw", "expected", "note"),
    [
        pytest.param(b"\xff" * 6, None, "standard all-ones missing", id="all_ones"),
        pytest.param(LEGACY_MISSING_STRING[:6], None, "legacy 10E10 placeholder", id="legacy_10e10"),
        pytest.param(LEGACY_MISSING_STRING[:4], None, "placeholder in a 4-byte field", id="legacy_10e10_4_bytes"),
        pytest.param(LEGACY_MISSING_STRING[:5], None, "placeholder in a 5-byte field", id="legacy_10e10_5_bytes"),
        pytest.param(LEGACY_MISSING_STRING, None, "placeholder with NUL tail", id="legacy_10e10_nul_tail"),
        pytest.param(LEGACY_MISSING_STRING[:5] + b"   ", None, "blank tail", id="legacy_10e10_blank_tail"),
        pytest.param(LEGACY_MISSING_STRING[:5] + b"\x00 \x00", None, "mixed tail", id="legacy_10e10_mixed_tail"),
        pytest.param(b"B7Hv-real", "B7Hv-real", "placeholder prefix in a 9-byte field", id="legacy_prefix_too_long"),
        pytest.param(b"B7Hv\xe1   ", "B7Hv", "fifth byte breaks the placeholder", id="legacy_prefix_wrong_fifth_byte"),
        pytest.param(b"TWNA01", "TWNA01", "ordinary printable text", id="printable"),
        pytest.param(b"AB\x00CD\x00", "AB CD", "internal non-printables", id="internal_nonprintable"),
    ],
)
def test_ia5_missing_and_invalid_bytes(tmp_path: Path, raw: bytes, expected, note: str):
    """Character fields must report missing, and must never be silently shortened.

    NCEP files written before NCEPLIBS 10.2 store the 10E10 missing placeholder
    directly in short character fields; its printable prefix ('B7Hv') is not a
    value. Deleting non-printable bytes would turn malformed data into a short
    string that nothing downstream can tell from real text.
    """
    import earth2bufr

    path = tmp_path / "ia5.prepbufr"
    path.write_bytes(build_ia5_missing_prepbufr(raw))

    rows = _epoch_rows(earth2bufr.read_bufr(path))
    assert len(rows) == 1, note
    assert rows[0]["TXT"] == expected, note
    # The preceding field must be untouched: the six bytes are always consumed
    # whole, so nothing shifts whichever way the content is interpreted.
    assert rows[0]["SID"] == "KDEN", note


def test_file_info_locates_messages_and_their_table_epochs(tmp_path: Path):
    """One metadata object covers every message, its bytes, and the tables that decode it."""
    import earth2bufr

    raw = build_two_epoch_prepbufr_messages()
    path = tmp_path / "two_epochs.prepbufr"
    path.write_bytes(raw)

    info = earth2bufr.get_metadata(path)

    assert [message.data_category for message in info.messages] == [11, 102, 11, 102]
    assert [message.table_epoch for message in info.messages] == [None, 0, None, 1]
    assert [message.source_message_index for message in info.data_messages] == [1, 3]
    for message in info.messages:
        assert raw[message.offset : message.offset + message.size_bytes].startswith(b"BUFR")
    assert sum(message.size_bytes for message in info.messages) == len(raw)
    first, second = info.table_epochs
    assert first.embedded_table_sha256 and second.embedded_table_sha256
    assert first.embedded_table_sha256 != second.embedded_table_sha256
    assert first.redefines == ()
    (redefinition,) = second.redefines
    assert (redefinition.descriptor, redefinition.table) == ("0-63-002", "B")
    assert (redefinition.before["bit_width"], redefinition.after["bit_width"]) == (12, 8)


def test_file_info_built_from_messages_equals_the_file_info_of_the_file(tmp_path: Path):
    """FileInfo(messages=..., table_epochs=...) holds the same table, messages and epochs as the opened file."""
    import earth2bufr

    path = tmp_path / "two_epochs.prepbufr"
    path.write_bytes(build_two_epoch_prepbufr_messages())
    with earth2bufr.open_bufr(path) as source:
        opened = source.info

    built = earth2bufr.FileInfo(messages=opened.messages, table_epochs=opened.table_epochs)

    assert built == opened
    assert built.message_table.schema == opened.message_table.schema
    assert built.messages == opened.messages
    assert built.data_messages == opened.data_messages
    assert earth2bufr.FileInfo(messages=(), table_epochs=()).message_table.num_rows == 0


def test_get_metadata_rejects_a_file_without_bufr_messages(tmp_path: Path):
    import earth2bufr

    path = tmp_path / "not_bufr.bin"
    path.write_bytes(b"no messages here")
    with pytest.raises(ValueError):
        earth2bufr.get_metadata(path)


def test_source_structure_reports_each_messages_table_epoch_hash(tmp_path: Path):
    """`embedded_table_sha256` identifies the message's table epoch."""
    import earth2bufr

    path = tmp_path / "two_epochs.prepbufr"
    path.write_bytes(build_two_epoch_prepbufr_messages())
    info = earth2bufr.get_metadata(path)

    with earth2bufr.open_bufr(path) as source:
        for message in info.data_messages:
            assert message.table_epoch is not None
            structure = source.source_structure(message.source_message_index)
            expected = info.table_epochs[message.table_epoch].embedded_table_sha256
            assert structure["embedded_table_sha256"] == expected


def test_embedded_dx_code_flag_descriptors_are_exempt_from_numeric_operators(tmp_path: Path):
    """Code/flag descriptors from an embedded DX table are exempt from 2-01/2-02.

    read_bufr loads the embedded DX table, classifies the DX-defined
    LOC_CODE/LOC_FLAG descriptors as code/flag, and therefore reads them at their
    native width while applying the width/scale operators only to the numeric
    LOC_NUM field.
    """
    import pyarrow as pa

    import earth2bufr

    path = tmp_path / "embedded_code_flag_operator.bufr"
    path.write_bytes(build_embedded_code_flag_operator_message())

    table = pa.Table.from_batches(earth2bufr.read_bufr(path)).to_pandas()

    assert table["LOC_CODE"].tolist() == [5.0]
    assert table["LOC_FLAG"].tolist() == [3.0]
    assert table["LOC_NUM"].tolist() == [pytest.approx(12.3)]
