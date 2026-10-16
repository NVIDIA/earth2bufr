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

"""Reference decoder adapters that normalize BUFR output to DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from .builders import AIR_TEMPERATURE, NON_COORDINATE_PRESSURE, YEAR

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .builders import DelayedReplicationCase, SyntheticBufrCase, WmoField


class ReferenceDecoderUnavailableError(RuntimeError):
    """Raised when a requested reference decoder cannot handle a fixture."""


def expected_frame(case: SyntheticBufrCase) -> pd.DataFrame:
    """Return expected case rows as a normalized DataFrame."""
    return pd.DataFrame([{field.name: row[field.name] for field in case.fields} for row in case.rows])


def decode_earth2bufr_frame(path: Path, fields: Sequence[WmoField]) -> pd.DataFrame:
    """Decode with Earth2Buffer and normalize Table-B columns to case names."""
    import pyarrow as pa

    import earth2bufr

    table = pa.Table.from_batches(earth2bufr.read_bufr(path))
    data = {}
    for field in fields:
        data[field.name] = [_normalize_missing(value) for value in table.column(field.earth2bufr_column).to_pylist()]
    return pd.DataFrame(data)


def decode_eccodes_frame(path: Path, fields: Sequence[WmoField]) -> pd.DataFrame:
    """Decode with ECMWF ecCodes and normalize key arrays to case names."""
    ecc = _import_eccodes()
    rows: list[dict[str, Any]] = []

    with path.open("rb") as handle:
        while True:
            bufr = ecc.codes_bufr_new_from_file(handle)
            if bufr is None:
                break
            try:
                ecc.codes_set(bufr, "skipExtraKeyAttributes", 1)
                ecc.codes_set(bufr, "unpack", 1)
                columns = {field.name: _eccodes_array(ecc, bufr, field.eccodes_key) for field in fields}
                row_count = max(len(values) for values in columns.values()) if columns else 0
                for row_index in range(row_count):
                    rows.append(
                        {name: values[row_index] if len(values) > 1 else values[0] for name, values in columns.items()}
                    )
            finally:
                ecc.codes_release(bufr)

    return pd.DataFrame(rows, columns=[field.name for field in fields])


def decode_pybufrkit_frame(path: Path, fields: Sequence[WmoField]) -> pd.DataFrame:
    """Decode with PyBufrKit and normalize element descriptors to case names."""
    decoder_type, generate_bufr_message = _import_pybufrkit_decoder()

    id_to_name = {field.pybufrkit_id: field.name for field in fields}
    rows: list[dict[str, Any]] = []
    for message in generate_bufr_message(decoder_type(), path.read_bytes()):
        template_data = message.template_data.value
        for descriptors, values in zip(
            template_data.decoded_descriptors_all_subsets,
            template_data.decoded_values_all_subsets,
            strict=False,
        ):
            row: dict[str, Any] = {}
            for descriptor, value in zip(descriptors, values, strict=False):
                if type(descriptor).__name__ != "ElementDescriptor":
                    continue
                name = id_to_name.get(int(str(descriptor)))
                if name is not None and name not in row:
                    row[name] = _normalize_missing(value)
            rows.append({field.name: row.get(field.name) for field in fields})

    return pd.DataFrame(rows, columns=[field.name for field in fields])


def decode_earth2bufr_delayed_replication(path: Path) -> list[dict[str, Any]]:
    """Decode a delayed-replication case with Earth2Buffer nested output."""
    import earth2bufr

    rows = earth2bufr.read_bufr(path)[0].to_pydict()
    result = []
    for year, levels in zip(rows[YEAR.earth2bufr_column], rows["B031001"], strict=True):
        result.append(
            {
                YEAR.name: _normalize_missing(year),
                "levels": [
                    {
                        NON_COORDINATE_PRESSURE.name: _normalize_missing(
                            level[NON_COORDINATE_PRESSURE.earth2bufr_column]
                        ),
                        AIR_TEMPERATURE.name: _normalize_missing(level[AIR_TEMPERATURE.earth2bufr_column]),
                    }
                    for level in levels
                ],
            }
        )
    return result


def decode_eccodes_delayed_replication(path: Path) -> list[dict[str, Any]]:
    """Decode a delayed-replication case with ecCodes."""
    ecc = _import_eccodes()
    result = []
    with path.open("rb") as handle:
        bufr = ecc.codes_bufr_new_from_file(handle)
        if bufr is None:
            return result
        try:
            ecc.codes_set(bufr, "skipExtraKeyAttributes", 1)
            ecc.codes_set(bufr, "unpack", 1)
            years = _eccodes_array(ecc, bufr, YEAR.eccodes_key)
            counts = _eccodes_array(ecc, bufr, "delayedDescriptorReplicationFactor")
            pressures = _eccodes_array(ecc, bufr, NON_COORDINATE_PRESSURE.eccodes_key)
            temperatures = _eccodes_array(ecc, bufr, AIR_TEMPERATURE.eccodes_key)
            offset = 0
            for row_index, count in enumerate(counts):
                level_count = int(count)
                result.append(
                    {
                        YEAR.name: years[row_index] if len(years) > 1 else years[0],
                        "levels": [
                            {
                                NON_COORDINATE_PRESSURE.name: pressures[offset + level_index],
                                AIR_TEMPERATURE.name: temperatures[offset + level_index],
                            }
                            for level_index in range(level_count)
                        ],
                    }
                )
                offset += level_count
        finally:
            ecc.codes_release(bufr)
    return result


def decode_pybufrkit_delayed_replication(path: Path) -> list[dict[str, Any]]:
    """Decode a delayed-replication case with PyBufrKit."""
    decoder_type, generate_bufr_message = _import_pybufrkit_decoder()

    result = []
    for message in generate_bufr_message(decoder_type(), path.read_bytes()):
        template_data = message.template_data.value
        for descriptors, values in zip(
            template_data.decoded_descriptors_all_subsets,
            template_data.decoded_values_all_subsets,
            strict=False,
        ):
            pairs = [
                (int(str(descriptor)), _normalize_missing(value))
                for descriptor, value in zip(descriptors, values, strict=False)
                if type(descriptor).__name__ == "ElementDescriptor"
            ]
            year = pairs[0][1]
            level_count = int(pairs[1][1])
            level_values = pairs[2:]
            levels = []
            for level_index in range(level_count):
                levels.append(
                    {
                        NON_COORDINATE_PRESSURE.name: level_values[level_index * 2][1],
                        AIR_TEMPERATURE.name: level_values[level_index * 2 + 1][1],
                    }
                )
            result.append({YEAR.name: year, "levels": levels})
    return result


def delayed_replication_expected_rows(case: DelayedReplicationCase) -> list[dict[str, Any]]:
    """Return expected nested rows for a delayed-replication case."""
    return [{YEAR.name: case.year, "levels": [dict(level) for level in case.levels]}]


def expected_prepbufr_frame(case) -> pd.DataFrame:
    """Return expected synthetic PrepBUFR rows as a normalized DataFrame."""
    return pd.DataFrame([dict(row) for row in case.rows], columns=case.columns)


def decode_satpy_frame(path: Path) -> pd.DataFrame:
    """Try to decode a BUFR file with SatPy product-specific readers."""
    try:
        from satpy import Scene, available_readers
    except ImportError as exc:
        raise ReferenceDecoderUnavailableError("SatPy is not installed") from exc

    bufr_readers = [reader for reader in available_readers() if "bufr" in reader.lower()]
    for reader in bufr_readers:
        try:
            scene = Scene(reader=reader, filenames=[str(path)])
        except Exception:
            continue
        dataset_names = scene.available_dataset_names()
        if dataset_names:
            scene.load(dataset_names)
            return pd.DataFrame({name: scene[name].values.ravel() for name in dataset_names})

    raise ReferenceDecoderUnavailableError(
        "SatPy did not find a product-specific BUFR reader for this generic synthetic WMO file. "
        f"Available BUFR readers: {bufr_readers}"
    )


def decode_eccodes_keys(path: Path, keys: Sequence[str]) -> dict[str, list[Any]]:
    """Decode the first message with ecCodes and return the arrays of ``keys``."""
    ecc = _import_eccodes()
    with path.open("rb") as handle:
        bufr = ecc.codes_bufr_new_from_file(handle)
        try:
            ecc.codes_set(bufr, "unpack", 1)
            return {key: _eccodes_array(ecc, bufr, key) for key in keys}
        finally:
            ecc.codes_release(bufr)


def _import_eccodes():
    try:
        import eccodes as ecc
    except ImportError as exc:
        raise ReferenceDecoderUnavailableError("ecCodes Python bindings are not installed") from exc
    return ecc


def _import_pybufrkit_decoder():
    try:
        from pybufrkit.decoder import Decoder, generate_bufr_message
    except ImportError as exc:
        raise ReferenceDecoderUnavailableError("PyBufrKit is not installed") from exc
    return Decoder, generate_bufr_message


def _eccodes_array(ecc, bufr, key: str) -> list[Any]:
    return [_normalize_missing(value, ecc=ecc) for value in ecc.codes_get_array(bufr, key).tolist()]


def _normalize_missing(value: Any, *, ecc=None) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").rstrip("\x00").strip()
    if ecc is not None and (value == ecc.CODES_MISSING_LONG or value == ecc.CODES_MISSING_DOUBLE):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value
