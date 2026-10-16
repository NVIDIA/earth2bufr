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


"""Small, deterministic BUFR builders for cross-decoder tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


def desc(f: int, x: int, y: int) -> int:
    """Return a packed 16-bit F-X-Y descriptor id."""
    return (f << 14) | (x << 8) | y


def desc_name(f: int, x: int, y: int) -> str:
    """Return a human-readable descriptor string."""
    return f"{f}-{x:02}-{y:03}"


def _u24(value: int) -> bytes:
    return value.to_bytes(3, "big")


class BitWriter:
    """Tiny MSB-first writer matching BUFR Section 4 bit packing."""

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current = 0
        self._used = 0

    def write_bits(self, value: int, width: int) -> None:
        """Write ``width`` bits from ``value`` most-significant bit first."""
        assert width >= 0
        assert value == 0 if width == 0 else 0 <= value < (1 << width), (value, width)
        for bit_index in range(width - 1, -1, -1):
            bit = (value >> bit_index) & 1
            self._current = (self._current << 1) | bit
            self._used += 1
            if self._used == 8:
                self._bytes.append(self._current)
                self._current = 0
                self._used = 0

    def write_ascii(self, value: str, byte_count: int) -> None:
        """Write a fixed-width ASCII field."""
        raw = value.encode("ascii")
        assert len(raw) == byte_count, (value, byte_count)
        for byte in raw:
            self.write_bits(byte, 8)

    def to_bytes(self) -> bytes:
        """Return byte-aligned encoded bits."""
        if self._used:
            self._bytes.append(self._current << (8 - self._used))
            self._current = 0
            self._used = 0
        return bytes(self._bytes)


@dataclass(frozen=True)
class WmoField:
    """A scalar WMO Table-B field used by a synthetic test case."""

    name: str
    f: int
    x: int
    y: int
    earth2bufr_column: str
    eccodes_key: str
    scale: int
    reference_value: int
    bit_width: int
    units: str

    @property
    def descriptor(self) -> int:
        """Return the packed F-X-Y integer."""
        return desc(self.f, self.x, self.y)

    @property
    def descriptor_name(self) -> str:
        """Return the F-XX-YYY text representation."""
        return desc_name(self.f, self.x, self.y)

    @property
    def pybufrkit_id(self) -> int:
        """Return PyBufrKit's integer descriptor id format."""
        return self.f * 100000 + self.x * 1000 + self.y

    def encode_raw(self, value: Any) -> int:
        """Encode a display value into raw BUFR bits."""
        if value is None:
            return (1 << self.bit_width) - 1
        return round(float(value) * (10**self.scale) - self.reference_value)


YEAR = WmoField("year", 0, 4, 1, "B004001", "year", 0, 0, 12, "year")
MONTH = WmoField("month", 0, 4, 2, "B004002", "month", 0, 0, 4, "month")
DAY = WmoField("day", 0, 4, 3, "B004003", "day", 0, 0, 6, "day")
HOUR = WmoField("hour", 0, 4, 4, "B004004", "hour", 0, 0, 5, "hour")
MINUTE = WmoField("minute", 0, 4, 5, "B004005", "minute", 0, 0, 6, "minute")
LATITUDE = WmoField("latitude", 0, 5, 1, "B005001", "latitude", 5, -9_000_000, 25, "deg")
LONGITUDE = WmoField("longitude", 0, 6, 1, "B006001", "longitude", 5, -18_000_000, 26, "deg")
AIR_TEMPERATURE = WmoField("air_temperature", 0, 12, 101, "B012101", "airTemperature", 2, 0, 16, "K")
WIND_DIRECTION = WmoField("wind_direction", 0, 11, 1, "B011001", "windDirection", 0, 0, 9, "deg")
WIND_SPEED = WmoField("wind_speed", 0, 11, 2, "B011002", "windSpeed", 1, 0, 12, "m/s")
RELATIVE_HUMIDITY = WmoField("relative_humidity", 0, 13, 3, "B013003", "relativeHumidity", 0, 0, 7, "%")
VERTICAL_SOUNDING_SIGNIFICANCE = WmoField(
    "vertical_sounding_significance",
    0,
    8,
    1,
    "B008001",
    "verticalSoundingSignificance",
    0,
    0,
    7,
    "FLAG TABLE",
)
FIRST_ORDER_STATISTICS = WmoField(
    "first_order_statistics", 0, 8, 23, "B008023", "firstOrderStatistics", 0, 0, 6, "CODE TABLE"
)
ALBEDO = WmoField("albedo", 0, 14, 27, "B014027", "albedo", 0, 0, 7, "%")
NON_COORDINATE_PRESSURE = WmoField(
    "non_coordinate_pressure", 0, 10, 4, "B010004", "nonCoordinatePressure", -1, 0, 14, "Pa"
)
DELAYED_REPLICATION_FACTOR = WmoField(
    "delayed_replication_factor", 0, 31, 1, "B031001", "delayedDescriptorReplicationFactor", 0, 0, 8, "numeric"
)
WMO_REGION = WmoField("wmo_region", 0, 1, 3, "B001003", "wmoRegion", 0, 0, 3, "CODE TABLE")
AIRCRAFT_FLIGHT_NUMBER = WmoField(
    "aircraft_flight_number",
    0,
    1,
    6,
    "B001006",
    "aircraftFlightNumber",
    0,
    0,
    64,
    "CCITT IA5",
)


@dataclass(frozen=True)
class SyntheticBufrCase:
    """A complete flat BUFR case with expected row values."""

    name: str
    description: str
    fields: Sequence[WmoField]
    rows: Sequence[dict[str, Any]]
    compressed: bool = False
    edition: int = 4
    data_category: int = 0

    def describe(self) -> str:
        """Return a compact human-readable case description for assertion output."""
        descriptors = [
            f"  {field.name}: {field.descriptor_name} {field.eccodes_key} [{field.units}]" for field in self.fields
        ]
        rows = [f"  row {idx}: {dict(row)}" for idx, row in enumerate(self.rows)]
        mode = "compressed" if self.compressed else "uncompressed"
        return "\n".join(
            [
                f"Synthetic BUFR case: {self.name}",
                f"Edition: {self.edition}",
                f"Mode: {mode}",
                self.description,
                "Descriptors:",
                *descriptors,
                "Rows:",
                *rows,
            ]
        )

    def write(self, directory: Path) -> Path:
        """Write the synthetic BUFR payload into ``directory`` and return its path."""
        path = directory / f"{self.name}.bufr"
        path.write_bytes(build_flat_wmo_message(self))
        return path


@dataclass(frozen=True)
class DelayedReplicationCase:
    """A tiny one-subset WMO delayed-replication fixture."""

    name: str
    description: str
    year: int
    levels: Sequence[dict[str, float]]
    data_category: int = 0

    def describe(self) -> str:
        """Return the descriptor tree in a readable form."""
        return "\n".join(
            [
                f"Synthetic BUFR case: {self.name}",
                self.description,
                "Descriptor tree:",
                "  B004001 year",
                "  1-02-000 delayed replication of two descriptors",
                "    B031001 replication count",
                "    B010004 non_coordinate_pressure [Pa]",
                "    B012101 air_temperature [K]",
                f"Rows: year={self.year}, levels={list(self.levels)}",
            ]
        )

    def write(self, directory: Path) -> Path:
        """Write the delayed-replication BUFR payload."""
        path = directory / f"{self.name}.bufr"
        path.write_bytes(build_delayed_replication_message(self))
        return path


@dataclass(frozen=True)
class SyntheticPrepbufrCase:
    """A tiny PrepBUFR file with an embedded DX table and one data message."""

    name: str
    description: str
    rows: Sequence[dict[str, Any]]
    data_category: int = 102

    @property
    def columns(self) -> list[str]:
        """Return the comparison columns in human-readable order."""
        return list(self.rows[0])

    def describe(self) -> str:
        """Return a readable summary of the embedded DX table and data rows."""
        return "\n".join(
            [
                f"Synthetic BUFR case: {self.name}",
                "Format: PrepBUFR with embedded DX table",
                self.description,
                "DX Table B:",
                "  0-63-001 SID [CCITT IA5, 32 bits]",
                "  0-63-002 TOB [K, scale=1, reference=2500, 12 bits]",
                "DX Table D:",
                "  3-63-001 SYNOBS -> SID, TOB",
                f"Data category: {self.data_category}",
                "Rows:",
                *[f"  row {idx}: {dict(row)}" for idx, row in enumerate(self.rows)],
            ]
        )

    def write(self, directory: Path) -> Path:
        """Write the synthetic PrepBUFR payload."""
        path = directory / f"{self.name}.prepbufr"
        path.write_bytes(build_synthetic_prepbufr_dx_messages(self))
        return path


def section1_ed4(*, data_category: int = 0) -> bytes:
    """Build a fixed BUFR Edition 4 Section 1."""
    section = bytearray(_u24(22))
    section.extend([0, 0, 98, 0, 0, 0, 0, data_category, 0, 0, 41, 0, 0x07, 0xE8, 1, 2, 3, 4, 5])
    assert len(section) == 22
    return bytes(section)


def section1_ed3(*, data_category: int = 0) -> bytes:
    """Build a fixed BUFR Edition 3 Section 1."""
    section = bytearray(_u24(18))
    section.extend([0, 0, 98, 0, 0, data_category, 0, 41, 0, 24, 1, 2, 3, 4, 0])
    assert len(section) == 18
    return bytes(section)


def section3(descriptors: Iterable[int], num_subsets: int, *, compressed: bool = False) -> bytes:
    """Build BUFR Section 3."""
    descriptor_bytes = b"".join(descriptor.to_bytes(2, "big") for descriptor in descriptors)
    flags = 0x40 if compressed else 0
    return (
        _u24(7 + len(descriptor_bytes)) + b"\x00" + num_subsets.to_bytes(2, "big") + bytes([flags]) + descriptor_bytes
    )


def bufr_message(
    descriptors: Iterable[int],
    section4_data: bytes,
    *,
    num_subsets: int,
    data_category: int = 0,
    compressed: bool = False,
    edition: int = 4,
) -> bytes:
    """Build a complete BUFR Edition 3 or Edition 4 message."""
    if edition == 3:
        section1 = section1_ed3(data_category=data_category)
    elif edition == 4:
        section1 = section1_ed4(data_category=data_category)
    else:
        raise ValueError(f"Unsupported synthetic BUFR edition: {edition}")

    body = section1 + section3(descriptors, num_subsets, compressed=compressed)
    body += _u24(4 + len(section4_data)) + b"\x00" + section4_data
    total_len = 8 + len(body) + 4
    return b"BUFR" + _u24(total_len) + bytes([edition]) + body + b"7777"


def bufr_ed4(
    descriptors: Iterable[int],
    section4_data: bytes,
    *,
    num_subsets: int,
    data_category: int = 0,
    compressed: bool = False,
) -> bytes:
    """Build a complete BUFR Edition 4 message."""
    return bufr_message(
        descriptors,
        section4_data,
        num_subsets=num_subsets,
        data_category=data_category,
        compressed=compressed,
        edition=4,
    )


def synthetic_prepbufr_dx_case() -> SyntheticPrepbufrCase:
    """Return a one-row PrepBUFR file driven by an embedded DX table."""
    return SyntheticPrepbufrCase(
        name="prepbufr_dx_local_sequence",
        description="One DX-table message defines local SID/TOB descriptors; one ADPUPA-like data message uses them.",
        rows=[{"SID": "KDEN", "TOB": 273.1, "_data_category": 102.0}],
    )


def build_synthetic_prepbufr_dx_messages(case: SyntheticPrepbufrCase) -> bytes:
    """Build a tiny PrepBUFR byte stream containing one DX message and one data message."""
    dx_message = bufr_message([], _build_dx_table_payload(), num_subsets=1, data_category=11)

    writer = BitWriter()
    writer.write_ascii(case.rows[0]["SID"], 4)
    writer.write_bits(round(case.rows[0]["TOB"] * 10 - 2500), 12)
    data_message = bufr_message(
        [desc(3, 63, 1)],
        writer.to_bytes(),
        num_subsets=1,
        data_category=case.data_category,
    )

    return dx_message + data_message


def build_synthetic_prepbufr_ia5_message(raw_sid: bytes) -> bytes:
    """Build a local-table message with an exact four-byte IA5 payload."""
    if len(raw_sid) != 4:
        raise ValueError("Synthetic SID must contain exactly four bytes")
    dx_message = bufr_message([], _build_dx_table_payload(), num_subsets=1, data_category=11)
    writer = BitWriter()
    for byte in raw_sid:
        writer.write_bits(byte, 8)
    writer.write_bits(round(273.1 * 10 - 2500), 12)
    data_message = bufr_message(
        [desc(3, 63, 1)],
        writer.to_bytes(),
        num_subsets=1,
        data_category=102,
    )
    return dx_message + data_message


def build_embedded_code_flag_operator_message() -> bytes:
    """Build a stream whose embedded DX table defines code/flag/numeric descriptors
    used under active 2-01/2-02 numeric operators.

    Exercises the code/flag classification in ``extract_dx_tables``: WMO numeric
    width/scale operators must be exempted for the DX-defined code and flag
    descriptors, so only the numeric field uses the modified representation.
    """
    payload = bytearray()
    payload.append(0)  # Table A entries
    payload.append(3)  # Table B entries
    payload += _dx_table_b_entry(0, 63, 30, "LOC_CODE", "Local code-table value", "CODE TABLE", 0, 0, 4)
    payload += _dx_table_b_entry(0, 63, 31, "LOC_FLAG", "Local flag-table value", "FLAG TABLE", 0, 0, 4)
    payload += _dx_table_b_entry(0, 63, 32, "LOC_NUM", "Local numeric value", "m", 0, 0, 7)
    payload.append(0)  # Table D entries
    dx_message = bufr_message([], bytes(payload), num_subsets=1, data_category=11)

    writer = BitWriter()
    writer.write_bits(5, 4)  # LOC_CODE = 5 (exempt: native 4 bits)
    writer.write_bits(3, 4)  # LOC_FLAG = 3 (exempt: native 4 bits)
    writer.write_bits(123, 7 + 3)  # LOC_NUM: width 7+3, scale 0+1 -> 12.3
    data_message = bufr_message(
        [
            desc(2, 1, 131),
            desc(2, 2, 129),
            desc(0, 63, 30),
            desc(0, 63, 31),
            desc(0, 63, 32),
            desc(2, 2, 0),
            desc(2, 1, 0),
        ],
        writer.to_bytes(),
        num_subsets=1,
        data_category=0,
    )
    return dx_message + data_message


def _build_dx_table_payload(
    second_entry: tuple[str, str, str, int, int, int] = (
        "TOB",
        "Synthetic temperature observation",
        "K",
        1,
        2500,
        12,
    ),
) -> bytes:
    mnemonic, name, units, scale, reference_value, bit_width = second_entry
    payload = bytearray()
    payload.append(0)  # Table A entries
    payload.append(2)  # Table B entries
    payload += _dx_table_b_entry(0, 63, 1, "SID", "Synthetic station identifier", "CCITT IA5", 0, 0, 32)
    payload += _dx_table_b_entry(0, 63, 2, mnemonic, name, units, scale, reference_value, bit_width)
    payload.append(1)  # Table D entries
    payload += _dx_table_d_entry(3, 63, 1, "SYNOBS", "Synthetic PrepBUFR observation", [(0, 63, 1), (0, 63, 2)])
    return bytes(payload)


# The second epoch redefines 0-63-002: different mnemonic, units, scale,
# reference and -- critically -- a different bit width, so a decoder that merges
# both DX blocks file-wide reads the wrong number of bits for one of the two
# epochs and every field after it slides out of alignment.
SECOND_EPOCH_REDEFINITION = ("WSP", "Synthetic wind speed observation", "M/S", 0, 0, 8)


# Big-endian IEEE-754 bytes of 10E10, the pre-NCEPLIBS-10.2 missing placeholder
# that old NCEP files copied straight into short character fields.
LEGACY_MISSING_STRING = bytes([0x42, 0x37, 0x48, 0x76, 0xE0, 0x00, 0x00, 0x00])


def build_ia5_missing_prepbufr(raw: bytes) -> bytes:
    """Build a one-subset file whose second field is an IA5 exactly as wide as `raw`.

    Sequence 3-63-001 is (SID, TXT), so a leading 4-byte SID is written first;
    it doubles as the check that the following field stays aligned.
    """
    entry = ("TXT", "Synthetic text", "CCITT IA5", 0, 0, 8 * len(raw))
    dx = bufr_message([], _build_dx_table_payload(entry), num_subsets=1, data_category=11)
    writer = BitWriter()
    writer.write_ascii("KDEN", 4)
    for byte in raw:
        writer.write_bits(byte, 8)
    data = bufr_message([desc(3, 63, 1)], writer.to_bytes(), num_subsets=1, data_category=102)
    return dx + data


def build_two_epoch_prepbufr_messages() -> bytes:
    """Build a PrepBUFR stream of two DX epochs that redefine one descriptor.

    This is the year-2000 NCEP ``satwnd`` defect in miniature: a physical file
    concatenates two dumps, each carrying its own complete DX block, and the
    second block gives ``0-63-002`` a different meaning and width than the first.
    """
    first_dx = bufr_message([], _build_dx_table_payload(), num_subsets=1, data_category=11)
    first_writer = BitWriter()
    first_writer.write_ascii("KDEN", 4)
    first_writer.write_bits(round(273.1 * 10 - 2500), 12)
    first_data = bufr_message([desc(3, 63, 1)], first_writer.to_bytes(), num_subsets=1, data_category=102)

    second_dx = bufr_message([], _build_dx_table_payload(SECOND_EPOCH_REDEFINITION), num_subsets=1, data_category=11)
    second_writer = BitWriter()
    second_writer.write_ascii("KSEA", 4)
    second_writer.write_bits(42, 8)
    second_data = bufr_message([desc(3, 63, 1)], second_writer.to_bytes(), num_subsets=1, data_category=102)

    return first_dx + first_data + second_dx + second_data


def build_bytcnt_framed_messages(subsets: Sequence[bytes], *, value_bits: int = 8) -> tuple[bytes, bytes]:
    """Build a DX-table message and a data message of (BYTCNT, VAL) subsets.

    Each subset is a 16-bit NCEP BYTCNT, which counts its own two bytes, followed
    by the given bytes. VAL is `value_bits` wide, so the bytes can pad the frame
    or fall short of it.
    """
    payload = bytearray()
    payload.append(0)  # Table A entries
    payload.append(2)  # Table B entries
    payload += _dx_table_b_entry(0, 63, 0, "BYTCNT", "Subset byte count", "BYTES", 0, 0, 16)
    payload += _dx_table_b_entry(0, 63, 2, "VAL", "Synthetic value", "NUMERIC", 0, 0, value_bits)
    payload.append(0)  # Table D entries
    dx_message = bufr_message([], bytes(payload), num_subsets=1, data_category=11)

    section4 = b"".join((2 + len(body)).to_bytes(2, "big") + body for body in subsets)
    data_message = bufr_message(
        [desc(0, 63, 0), desc(0, 63, 2)],
        section4,
        num_subsets=len(subsets),
        data_category=102,
    )
    return dx_message, data_message


def _dx_table_b_entry(
    f: int,
    x: int,
    y: int,
    mnemonic: str,
    name: str,
    units: str,
    scale: int,
    reference_value: int,
    bit_width: int,
) -> bytes:
    sign_scale, scale_mag = _signed_decimal(scale, 3)
    sign_ref, ref_mag = _signed_decimal(reference_value, 10)
    entry = (
        f"{f:1d}{x:02d}{y:03d}"
        f"{mnemonic:<8.8}"
        f"{name:<56.56}"
        f"{units:<24.24}"
        f"{sign_scale}{scale_mag}"
        f"{sign_ref}{ref_mag}"
        f"{bit_width:03d}"
    )
    assert len(entry) == 112
    return entry.encode("ascii")


def _dx_table_d_entry(
    f: int,
    x: int,
    y: int,
    mnemonic: str,
    description: str,
    members: Sequence[tuple[int, int, int]],
) -> bytes:
    header = f"{f:1d}{x:02d}{y:03d}" + f"{mnemonic:<8.8}{description:<56.56}"
    assert len(header) == 70
    payload = bytearray(header.encode("ascii"))
    payload.append(len(members))
    for member in members:
        payload += f"{member[0]:1d}{member[1]:02d}{member[2]:03d}".encode("ascii")
    return bytes(payload)


def _signed_decimal(value: int, width: int) -> tuple[str, str]:
    sign = "-" if value < 0 else "+"
    return sign, f"{abs(value):0{width}d}"


def build_flat_wmo_message(case: SyntheticBufrCase) -> bytes:
    """Build a flat uncompressed or compressed WMO BUFR message."""
    writer = BitWriter()
    if case.compressed:
        _write_compressed_rows(writer, case.fields, case.rows)
    else:
        _write_uncompressed_rows(writer, case.fields, case.rows)
    return bufr_message(
        [field.descriptor for field in case.fields],
        writer.to_bytes(),
        num_subsets=len(case.rows),
        data_category=case.data_category,
        compressed=case.compressed,
        edition=case.edition,
    )


def build_code_flag_operator_message(case: SyntheticBufrCase) -> bytes:
    """Build a message where width/scale operators span categorical data."""
    writer = BitWriter()
    row = case.rows[0]

    # Operators 2-01-131 and 2-02-129 add three bits and one scale digit.
    # WMO exempts code and flag tables, so only ALBEDO uses the modified
    # 10-bit, scale=1 representation.
    writer.write_bits(int(row[FIRST_ORDER_STATISTICS.name]), FIRST_ORDER_STATISTICS.bit_width)
    writer.write_bits(
        (1 << VERTICAL_SOUNDING_SIGNIFICANCE.bit_width) - 1,
        VERTICAL_SOUNDING_SIGNIFICANCE.bit_width,
    )
    writer.write_bits(round(row[ALBEDO.name] * 10), ALBEDO.bit_width + 3)

    descriptors = [
        desc(2, 1, 131),
        desc(2, 2, 129),
        FIRST_ORDER_STATISTICS.descriptor,
        VERTICAL_SOUNDING_SIGNIFICANCE.descriptor,
        ALBEDO.descriptor,
        desc(2, 2, 0),
        desc(2, 1, 0),
    ]
    return bufr_ed4(descriptors, writer.to_bytes(), num_subsets=1, data_category=case.data_category)


def build_code_flag_combined_operator_message(case: SyntheticBufrCase) -> bytes:
    """Build a message where operator 2-07 spans categorical data."""
    writer = BitWriter()
    row = case.rows[0]

    writer.write_bits(int(row[FIRST_ORDER_STATISTICS.name]), FIRST_ORDER_STATISTICS.bit_width)
    writer.write_bits(
        (1 << VERTICAL_SOUNDING_SIGNIFICANCE.bit_width) - 1,
        VERTICAL_SOUNDING_SIGNIFICANCE.bit_width,
    )
    writer.write_bits(round(row[ALBEDO.name] * 10), ALBEDO.bit_width + 4)

    descriptors = [
        desc(2, 7, 1),
        FIRST_ORDER_STATISTICS.descriptor,
        VERTICAL_SOUNDING_SIGNIFICANCE.descriptor,
        ALBEDO.descriptor,
        desc(2, 7, 0),
    ]
    return bufr_ed4(descriptors, writer.to_bytes(), num_subsets=1, data_category=case.data_category)


def build_delayed_replication_message(case: DelayedReplicationCase) -> bytes:
    """Build one subset containing a delayed list of pressure/temperature levels."""
    writer = BitWriter()
    writer.write_bits(YEAR.encode_raw(case.year), YEAR.bit_width)
    writer.write_bits(DELAYED_REPLICATION_FACTOR.encode_raw(len(case.levels)), DELAYED_REPLICATION_FACTOR.bit_width)
    for level in case.levels:
        writer.write_bits(
            NON_COORDINATE_PRESSURE.encode_raw(level[NON_COORDINATE_PRESSURE.name]), NON_COORDINATE_PRESSURE.bit_width
        )
        writer.write_bits(AIR_TEMPERATURE.encode_raw(level[AIR_TEMPERATURE.name]), AIR_TEMPERATURE.bit_width)
    descriptors = [
        YEAR.descriptor,
        desc(1, 2, 0),
        DELAYED_REPLICATION_FACTOR.descriptor,
        NON_COORDINATE_PRESSURE.descriptor,
        AIR_TEMPERATURE.descriptor,
    ]
    return bufr_ed4(descriptors, writer.to_bytes(), num_subsets=1, data_category=case.data_category)


def build_compressed_delayed_replication_message(case: DelayedReplicationCase) -> bytes:
    """Build one compressed subset with delayed pressure/temperature levels."""
    writer = BitWriter()
    writer.write_bits(YEAR.encode_raw(case.year), YEAR.bit_width)
    writer.write_bits(0, 6)
    writer.write_bits(
        DELAYED_REPLICATION_FACTOR.encode_raw(len(case.levels)),
        DELAYED_REPLICATION_FACTOR.bit_width,
    )
    writer.write_bits(0, 6)
    for level in case.levels:
        writer.write_bits(
            NON_COORDINATE_PRESSURE.encode_raw(level[NON_COORDINATE_PRESSURE.name]),
            NON_COORDINATE_PRESSURE.bit_width,
        )
        writer.write_bits(0, 6)
        writer.write_bits(
            AIR_TEMPERATURE.encode_raw(level[AIR_TEMPERATURE.name]),
            AIR_TEMPERATURE.bit_width,
        )
        writer.write_bits(0, 6)
    descriptors = [
        YEAR.descriptor,
        desc(1, 2, 0),
        DELAYED_REPLICATION_FACTOR.descriptor,
        NON_COORDINATE_PRESSURE.descriptor,
        AIR_TEMPERATURE.descriptor,
    ]
    return bufr_message(
        descriptors,
        writer.to_bytes(),
        num_subsets=1,
        data_category=case.data_category,
        compressed=True,
    )


def build_nested_fixed_replication_message(*, compressed: bool) -> bytes:
    """Build two subsets with nested numeric, code, string, and null leaves."""
    rows: list[list[dict[str, Any]]] = [
        [
            {
                "wind": 1.0,
                "region": 2,
                "inner": [(280.0, "FLT00001"), (281.0, "FLT00002"), (None, "FLT00003")],
            },
            {
                "wind": 2.0,
                "region": 3,
                "inner": [(282.0, "FLT00004"), (283.0, "FLT00005"), (284.0, "FLT00006")],
            },
        ],
        [
            {
                "wind": 1.2,
                "region": 4,
                "inner": [(280.5, "FLT00001"), (281.5, "FLT00002"), (None, "FLT00003")],
            },
            {
                "wind": 2.2,
                "region": 5,
                "inner": [(282.5, "FLT00004"), (283.5, "FLT00005"), (284.5, "FLT00006")],
            },
        ],
    ]
    writer = BitWriter()
    if compressed:
        _write_compressed_numeric_values(writer, YEAR, [2024, 2025])
        for outer in range(2):
            _write_compressed_numeric_values(writer, WIND_SPEED, [row[outer]["wind"] for row in rows])
            _write_compressed_numeric_values(writer, WMO_REGION, [row[outer]["region"] for row in rows])
            for inner in range(3):
                _write_compressed_numeric_values(
                    writer,
                    AIR_TEMPERATURE,
                    [row[outer]["inner"][inner][0] for row in rows],
                )
                _write_compressed_strings(
                    writer,
                    [row[outer]["inner"][inner][1] for row in rows],
                    AIRCRAFT_FLIGHT_NUMBER.bit_width // 8,
                )
    else:
        for year, row in zip((2024, 2025), rows, strict=True):
            writer.write_bits(YEAR.encode_raw(year), YEAR.bit_width)
            for outer in row:
                writer.write_bits(WIND_SPEED.encode_raw(outer["wind"]), WIND_SPEED.bit_width)
                writer.write_bits(WMO_REGION.encode_raw(outer["region"]), WMO_REGION.bit_width)
                for temperature, flight in outer["inner"]:
                    writer.write_bits(
                        AIR_TEMPERATURE.encode_raw(temperature),
                        AIR_TEMPERATURE.bit_width,
                    )
                    writer.write_ascii(flight, AIRCRAFT_FLIGHT_NUMBER.bit_width // 8)
    descriptors = [
        YEAR.descriptor,
        desc(1, 5, 2),
        WIND_SPEED.descriptor,
        WMO_REGION.descriptor,
        desc(1, 2, 3),
        AIR_TEMPERATURE.descriptor,
        AIRCRAFT_FLIGHT_NUMBER.descriptor,
    ]
    return bufr_message(
        descriptors,
        writer.to_bytes(),
        num_subsets=len(rows),
        compressed=compressed,
    )


def build_nested_delayed_replication_message(*, compressed: bool) -> bytes:
    """Build two subsets with a fixed wind pair and delayed levels of delayed temperatures."""
    years = [2024, 2026]
    winds = [[1.0, 2.0], [1.5, 2.5]]
    pressures = [[50_000.0, 70_000.0], [50_020.0, 70_020.0]]
    temperatures: list[list[list[float | None]]] = [
        [[280.0], [281.0, 282.0, None]],
        [[280.5], [281.5, 282.5, None]],
    ]
    writer = BitWriter()
    if compressed:
        _write_compressed_numeric_values(writer, YEAR, years)
        for outer in range(2):
            _write_compressed_numeric_values(writer, WIND_SPEED, [row[outer] for row in winds])
        level_count = len(pressures[0])
        _write_compressed_numeric_values(writer, DELAYED_REPLICATION_FACTOR, [level_count] * len(years))
        for level in range(level_count):
            _write_compressed_numeric_values(writer, NON_COORDINATE_PRESSURE, [row[level] for row in pressures])
            temperature_count = len(temperatures[0][level])
            _write_compressed_numeric_values(writer, DELAYED_REPLICATION_FACTOR, [temperature_count] * len(years))
            for inner in range(temperature_count):
                _write_compressed_numeric_values(writer, AIR_TEMPERATURE, [row[level][inner] for row in temperatures])
    else:
        for subset, year in enumerate(years):
            writer.write_bits(YEAR.encode_raw(year), YEAR.bit_width)
            for wind in winds[subset]:
                writer.write_bits(WIND_SPEED.encode_raw(wind), WIND_SPEED.bit_width)
            writer.write_bits(len(pressures[subset]), DELAYED_REPLICATION_FACTOR.bit_width)
            for pressure, level_temperatures in zip(pressures[subset], temperatures[subset], strict=True):
                writer.write_bits(NON_COORDINATE_PRESSURE.encode_raw(pressure), NON_COORDINATE_PRESSURE.bit_width)
                writer.write_bits(len(level_temperatures), DELAYED_REPLICATION_FACTOR.bit_width)
                for temperature in level_temperatures:
                    writer.write_bits(AIR_TEMPERATURE.encode_raw(temperature), AIR_TEMPERATURE.bit_width)
    descriptors = [
        YEAR.descriptor,
        desc(1, 1, 2),
        WIND_SPEED.descriptor,
        desc(1, 4, 0),
        DELAYED_REPLICATION_FACTOR.descriptor,
        NON_COORDINATE_PRESSURE.descriptor,
        desc(1, 1, 0),
        DELAYED_REPLICATION_FACTOR.descriptor,
        AIR_TEMPERATURE.descriptor,
    ]
    return bufr_message(descriptors, writer.to_bytes(), num_subsets=len(years), compressed=compressed)


def build_fixed_replication_operator_message() -> bytes:
    """Build fixed replication containing an active width operator."""
    writer = BitWriter()
    writer.write_bits(45, ALBEDO.bit_width + 1)
    writer.write_bits(67, ALBEDO.bit_width + 1)
    descriptors = [
        desc(1, 3, 2),
        desc(2, 1, 129),
        ALBEDO.descriptor,
        desc(2, 1, 0),
    ]
    return bufr_message(descriptors, writer.to_bytes(), num_subsets=1)


PERCENT_CONFIDENCE = WmoField("percent_confidence", 0, 33, 7, "B033007", "percentConfidence", 0, 0, 7, "%")


def build_quality_bitmap_message() -> bytes:
    """Build quality information on a data present bitmap, the bitmap reused, then a wind speed.

    Operators 2-22-000, 2-36-000, 2-37-000 and 2-37-255 occupy no Section 4 bits.
    """
    writer = BitWriter()
    writer.write_bits(YEAR.encode_raw(2024), YEAR.bit_width)
    writer.write_bits(AIR_TEMPERATURE.encode_raw(281.5), AIR_TEMPERATURE.bit_width)
    writer.write_bits(1, 1)
    writer.write_bits(0, 1)
    writer.write_bits(70, PERCENT_CONFIDENCE.bit_width)
    writer.write_bits(90, PERCENT_CONFIDENCE.bit_width)
    writer.write_bits(WIND_SPEED.encode_raw(3.5), WIND_SPEED.bit_width)
    descriptors = [
        YEAR.descriptor,
        AIR_TEMPERATURE.descriptor,
        desc(2, 22, 0),
        desc(2, 36, 0),
        desc(1, 1, 2),
        desc(0, 31, 31),
        PERCENT_CONFIDENCE.descriptor,
        desc(2, 22, 0),
        desc(2, 37, 0),
        PERCENT_CONFIDENCE.descriptor,
        desc(2, 37, 255),
        WIND_SPEED.descriptor,
    ]
    return bufr_message(descriptors, writer.to_bytes(), num_subsets=1)


def _write_compressed_numeric_values(writer: BitWriter, field: WmoField, values: Sequence[Any]) -> None:
    raw_values = [field.encode_raw(value) for value in values]
    missing = (1 << field.bit_width) - 1
    if all(value == missing for value in raw_values):
        writer.write_bits(missing, field.bit_width)
        writer.write_bits(0, 6)
        return
    if any(value == missing for value in raw_values):
        raise ValueError("Synthetic compressed values must be all missing or all present")
    reference = min(raw_values)
    increments = [value - reference for value in raw_values]
    increment_bits = max(increments).bit_length()
    writer.write_bits(reference, field.bit_width)
    writer.write_bits(increment_bits, 6)
    for increment in increments:
        writer.write_bits(increment, increment_bits)


def build_compressed_station_name_group_message(names: Sequence[str], temperatures: Sequence[float]) -> bytes:
    """Build a compressed message, one subset per name, of a delayed group holding a name and a temperature.

    The names differ between subsets, so their compressed increments are non-zero
    (NBINC counts bytes for strings).
    """
    name_bytes = 20
    writer = BitWriter()
    writer.write_bits(1, 8)  # replication factor reference: one level in every subset
    writer.write_bits(0, 6)
    writer.write_bits(0, 8 * name_bytes)  # string reference, all zero
    writer.write_bits(name_bytes, 6)
    for name in names:
        writer.write_ascii(name.ljust(name_bytes), name_bytes)
    raw = [round(temperature * 100) for temperature in temperatures]
    reference = min(raw)
    increment_bits = max(value - reference for value in raw).bit_length()
    writer.write_bits(reference, 16)
    writer.write_bits(increment_bits, 6)
    for value in raw:
        writer.write_bits(value - reference, increment_bits)
    return bufr_message(
        [desc(1, 2, 0), desc(0, 31, 1), desc(0, 1, 15), desc(0, 12, 101)],
        writer.to_bytes(),
        num_subsets=len(names),
        compressed=True,
    )


def _write_compressed_strings(writer: BitWriter, values: Sequence[str], byte_count: int) -> None:
    if len(set(values)) != 1:
        raise ValueError("Synthetic compressed strings must share one value")
    writer.write_ascii(values[0], byte_count)
    writer.write_bits(0, 6)


def _write_uncompressed_rows(writer: BitWriter, fields: Sequence[WmoField], rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        for field in fields:
            writer.write_bits(field.encode_raw(row[field.name]), field.bit_width)


def _write_compressed_rows(writer: BitWriter, fields: Sequence[WmoField], rows: Sequence[dict[str, Any]]) -> None:
    for field in fields:
        raw_values = [field.encode_raw(row[field.name]) for row in rows]
        if any(raw == (1 << field.bit_width) - 1 for raw in raw_values):
            raise ValueError("Compressed missing-value fixtures are intentionally not supported yet")
        reference = min(raw_values)
        increments = [raw - reference for raw in raw_values]
        nbinc = max(increments).bit_length() if increments else 0
        writer.write_bits(reference, field.bit_width)
        writer.write_bits(nbinc, 6)
        for increment in increments:
            writer.write_bits(increment, nbinc)


def surface_observation_case() -> SyntheticBufrCase:
    """Return a normal WMO scalar case with two subsets."""
    fields = [
        YEAR,
        MONTH,
        DAY,
        HOUR,
        MINUTE,
        LATITUDE,
        LONGITUDE,
        AIR_TEMPERATURE,
        WIND_DIRECTION,
        WIND_SPEED,
        RELATIVE_HUMIDITY,
    ]
    return SyntheticBufrCase(
        name="wmo_surface_scalars",
        description="Two observations with time, geolocation, temperature, wind, and humidity.",
        fields=fields,
        rows=[
            {
                "year": 2024,
                "month": 1,
                "day": 2,
                "hour": 3,
                "minute": 4,
                "latitude": 35.12345,
                "longitude": -97.12345,
                "air_temperature": 280.15,
                "wind_direction": 270,
                "wind_speed": 12.3,
                "relative_humidity": 54,
            },
            {
                "year": 2024,
                "month": 1,
                "day": 2,
                "hour": 6,
                "minute": 30,
                "latitude": -12.5,
                "longitude": 130.25,
                "air_temperature": 295.4,
                "wind_direction": 90,
                "wind_speed": 4.4,
                "relative_humidity": 80,
            },
        ],
    )


def edition3_surface_case() -> SyntheticBufrCase:
    """Return an Edition 3 WMO scalar case with one subset."""
    return SyntheticBufrCase(
        name="wmo_edition3_surface_scalars",
        description="Edition 3 message header with standard time, location, and temperature descriptors.",
        fields=[YEAR, MONTH, DAY, HOUR, MINUTE, LATITUDE, LONGITUDE, AIR_TEMPERATURE],
        rows=[
            {
                "year": 2024,
                "month": 1,
                "day": 2,
                "hour": 3,
                "minute": 4,
                "latitude": 40.0,
                "longitude": -105.25,
                "air_temperature": 281.75,
            }
        ],
        edition=3,
    )


def missing_value_case() -> SyntheticBufrCase:
    """Return a tiny WMO case with an all-bits-set missing value."""
    return SyntheticBufrCase(
        name="wmo_missing_value",
        description="A one-row file where relative humidity is encoded as BUFR missing.",
        fields=[YEAR, RELATIVE_HUMIDITY],
        rows=[{"year": 2024, "relative_humidity": None}],
    )


def compressed_numeric_case() -> SyntheticBufrCase:
    """Return a compressed WMO case with non-zero increments."""
    return SyntheticBufrCase(
        name="wmo_compressed_numeric",
        description="Compressed scalar data with increments for temperature and wind speed.",
        fields=[AIR_TEMPERATURE, WIND_SPEED],
        rows=[
            {"air_temperature": 280.0, "wind_speed": 1.0},
            {"air_temperature": 280.25, "wind_speed": 1.1},
            {"air_temperature": 280.5, "wind_speed": 1.2},
        ],
        compressed=True,
    )


def code_flag_operator_case() -> SyntheticBufrCase:
    """Return a case exercising WMO numeric-operator code/flag exemptions."""
    return SyntheticBufrCase(
        name="wmo_code_flag_operator_exemption",
        description=(
            "Active width and scale operators span code/flag values and numeric albedo; only albedo is modified."
        ),
        fields=[FIRST_ORDER_STATISTICS, VERTICAL_SOUNDING_SIGNIFICANCE, ALBEDO],
        rows=[
            {
                FIRST_ORDER_STATISTICS.name: 2,
                VERTICAL_SOUNDING_SIGNIFICANCE.name: None,
                ALBEDO.name: 45.6,
            }
        ],
    )


def code_flag_combined_operator_case() -> SyntheticBufrCase:
    """Return a case exercising the WMO 2-07 code/flag exemptions."""
    return SyntheticBufrCase(
        name="wmo_code_flag_combined_operator_exemption",
        description="Active 2-07 spans code/flag values and numeric albedo; only albedo is modified.",
        fields=[FIRST_ORDER_STATISTICS, VERTICAL_SOUNDING_SIGNIFICANCE, ALBEDO],
        rows=[
            {
                FIRST_ORDER_STATISTICS.name: 3,
                VERTICAL_SOUNDING_SIGNIFICANCE.name: None,
                ALBEDO.name: 54.3,
            }
        ],
    )


def delayed_replication_case() -> DelayedReplicationCase:
    """Return a nested WMO delayed-replication case."""
    return DelayedReplicationCase(
        name="wmo_delayed_replication_levels",
        description="One observation with two pressure/temperature levels under delayed replication.",
        year=2024,
        levels=[
            {"non_coordinate_pressure": 100_000.0, "air_temperature": 280.15},
            {"non_coordinate_pressure": 85_000.0, "air_temperature": 270.0},
        ],
    )


ASSOCIATED_FIELD_SIGNIFICANCE = WmoField(
    "associated_field_significance", 0, 31, 21, "B031021", "associatedFieldSignificance", 0, 0, 6, "CODE TABLE"
)


@dataclass(frozen=True)
class Fixed:
    """Fixed replication of ``items`` repeated ``count`` times."""

    count: int
    items: tuple[Any, ...]


@dataclass(frozen=True)
class Delayed:
    """Delayed replication of ``items`` with an 8-bit 0-31-001 factor."""

    items: tuple[Any, ...]


def template_descriptors(items: Sequence[Any]) -> list[int]:
    """Section 3 descriptors for a template of fields, ``Fixed`` and ``Delayed`` groups."""
    descriptors: list[int] = []
    for item in items:
        if isinstance(item, Fixed):
            inner = template_descriptors(item.items)
            descriptors += [desc(1, len(inner), item.count), *inner]
        elif isinstance(item, Delayed):
            inner = template_descriptors(item.items)
            descriptors += [desc(1, len(inner), 0), DELAYED_REPLICATION_FACTOR.descriptor, *inner]
        else:
            descriptors.append(item.descriptor)
    return descriptors


def build_template_message(items: Sequence[Any], subsets: Sequence[Sequence[Any]], *, compressed: bool) -> bytes:
    """Encode the same subsets of a template compressed or uncompressed.

    A subset holds one value per template item: a field value (``None`` is
    missing), a list of occurrences for ``Fixed``, or a list of occurrences for
    ``Delayed``; an occurrence holds one value per group item. Compressed
    subsets must share every delayed replication count.
    """
    writer = BitWriter()
    if compressed:
        _write_template_compressed(writer, items, subsets)
    else:
        for subset in subsets:
            _write_template_subset(writer, items, subset)
    return bufr_message(template_descriptors(items), writer.to_bytes(), num_subsets=len(subsets), compressed=compressed)


def _is_string(field: WmoField) -> bool:
    return field.units == "CCITT IA5"


def _write_template_subset(writer: BitWriter, items: Sequence[Any], values: Sequence[Any]) -> None:
    for item, value in zip(items, values, strict=True):
        if isinstance(item, Fixed):
            for occurrence in value:
                _write_template_subset(writer, item.items, occurrence)
        elif isinstance(item, Delayed):
            writer.write_bits(len(value), DELAYED_REPLICATION_FACTOR.bit_width)
            for occurrence in value:
                _write_template_subset(writer, item.items, occurrence)
        elif _is_string(item):
            byte_count = item.bit_width // 8
            if value is None:
                writer.write_bits((1 << item.bit_width) - 1, item.bit_width)
            else:
                writer.write_ascii(value.ljust(byte_count), byte_count)
        else:
            writer.write_bits(item.encode_raw(value), item.bit_width)


def _write_template_compressed(writer: BitWriter, items: Sequence[Any], subsets: Sequence[Sequence[Any]]) -> None:
    for index, item in enumerate(items):
        column = [subset[index] for subset in subsets]
        if isinstance(item, Fixed):
            for occurrence in range(item.count):
                _write_template_compressed(writer, item.items, [value[occurrence] for value in column])
        elif isinstance(item, Delayed):
            counts = {len(value) for value in column}
            if len(counts) != 1:
                raise ValueError("Compressed subsets must share delayed replication counts")
            count = counts.pop()
            _write_compressed_increments(writer, DELAYED_REPLICATION_FACTOR, [count] * len(column))
            for occurrence in range(count):
                _write_template_compressed(writer, item.items, [value[occurrence] for value in column])
        elif _is_string(item):
            _write_compressed_string_column(writer, item, column)
        else:
            _write_compressed_increments(writer, item, column)


def _write_compressed_increments(writer: BitWriter, field: WmoField, values: Sequence[Any]) -> None:
    missing = (1 << field.bit_width) - 1
    present = [field.encode_raw(value) for value in values if value is not None]
    if not present:
        writer.write_bits(missing, field.bit_width)
        writer.write_bits(0, 6)
        return
    reference = min(present)
    increments = [None if value is None else field.encode_raw(value) - reference for value in values]
    if all(increment == 0 for increment in increments):
        writer.write_bits(reference, field.bit_width)
        writer.write_bits(0, 6)
        return
    # An all-ones increment marks a missing subset, so present increments stay below it.
    width = (max(increment for increment in increments if increment is not None) + 1).bit_length()
    writer.write_bits(reference, field.bit_width)
    writer.write_bits(width, 6)
    for increment in increments:
        writer.write_bits((1 << width) - 1 if increment is None else increment, width)


def _write_compressed_string_column(writer: BitWriter, field: WmoField, values: Sequence[str | None]) -> None:
    byte_count = field.bit_width // 8
    missing = (1 << field.bit_width) - 1
    if all(value is None for value in values):
        writer.write_bits(missing, field.bit_width)
        writer.write_bits(0, 6)
        return
    # A column with a missing subset has a missing reference and all-ones increments for its missing subsets.
    writer.write_bits(missing if None in values else 0, field.bit_width)
    writer.write_bits(byte_count, 6)
    for value in values:
        if value is None:
            writer.write_bits(missing, field.bit_width)
        else:
            writer.write_ascii(value.ljust(byte_count), byte_count)
