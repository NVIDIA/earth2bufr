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
"""Earth2bufr: Rust-based BUFR file parser with Python bindings.

This module provides high-performance BUFR (Binary Universal Form for the
Representation of meteorological data) parsing using a Rust backend with
Python bindings via PyO3.

Features:
- Supports BUFR Edition 3 and Edition 4 formats
- Handles both compressed and uncompressed data
- Outputs data as Apache Arrow RecordBatches or Pandas DataFrames
- Parallel processing of multiple BUFR messages
- Extensible table support (Table B, Table D, local tables)

Example:
    >>> import earth2bufr
    >>> batches = earth2bufr.read_bufr("observations.bufr")
    >>> print(batches[0].schema)

"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import cached_property, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__version__ = "0.2.0a0"

# Path to bundled tables
_TABLES_DIR = Path(__file__).parent / "tables"

# Lazy load the Rust extension
_lib = None


def _get_lib():
    """Lazy load the Rust extension module."""
    global _lib
    if _lib is None:
        try:
            import earth2bufr._lib as lib_module

            _lib = lib_module
        except ImportError as e:
            raise ImportError(
                "Rust extension not built. Run 'maturin develop' or 'pip install -e .' to build the extension."
            ) from e
    return _lib


@lru_cache(maxsize=1)
def _load_default_tables() -> tuple[str, str]:
    """Load the default Table B and Table D JSON files."""
    table_b_path = _TABLES_DIR / "table_b.json"
    table_d_path = _TABLES_DIR / "table_d.json"

    if not table_b_path.exists():
        raise FileNotFoundError(f"Table B not found at {table_b_path}")
    if not table_d_path.exists():
        raise FileNotFoundError(f"Table D not found at {table_d_path}")

    return table_b_path.read_text(), table_d_path.read_text()


def flatten_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Flatten a nested RecordBatch into a fully-scalar table.

    Converts a flat-layout batch from ``read_bufr`` or ``iter_batches`` into a
    RecordBatch of scalar columns. ``read_bufr(flatten=True)`` applies the same
    rules to each of its batches.

    Two phases are applied:

    1. **Row-expanding flatten (CAT lists only)**: A multi-element list<struct>
       column is expanded into one row per element **only** when its inner struct
       has a ``CAT`` field — the NCEP PrepBUFR vertical-level marker. Scalar
       columns are broadcast across the produced rows, and rows whose CAT list
       is empty are dropped. If no such CAT-bearing list exists, this phase is
       skipped.
    2. **Scalarize list<struct> columns**: Every list<struct> column — including
       multi-element non-CAT lists and the CAT list after phase 1 — is **not**
       expanded into rows. It is collapsed by taking **element 0** and promoting
       that struct's fields to top-level columns in its place. Lists nested in
       the promoted fields are collapsed the same way.

    Consequently, repeated values in non-CAT lists (e.g. present-weather / cloud
    groups, LOCAL descriptor arrays) are **not** preserved as separate rows —
    only their first element survives — and subsets whose optional non-CAT list
    is empty or null are retained with null values, not dropped. A list column
    that is empty in every row is promoted the same way, as null columns. A
    promoted field whose name an earlier column already has is renamed with a
    ``_2``, ``_3``, ... suffix. The output column names and types therefore
    depend only on the input schema, not on the values.

    Parameters
    ----------
    batch : pyarrow.RecordBatch
        Flat-layout batch from BUFR decoding. ``FixedSizeList`` columns from
        ``fixed_replication="structural"`` are not flattened.

    Returns
    -------
    pyarrow.RecordBatch
        Batch whose ``List<Struct>`` columns are expanded or scalarized.

    Examples
    --------
    >>> import earth2bufr
    >>> batches = earth2bufr.read_bufr("observations.bufr", flatten=False)
    >>> flat_batches = [earth2bufr.flatten_batch(b) for b in batches]
    """
    import pyarrow as pa

    return pa.record_batch(_get_lib().flatten_batch_rust(batch))


@dataclass(frozen=True)
class ListValueFilter:
    """Retain repeated list entries whose numeric key has an allowed value.

    Parameters
    ----------
    field : str
        Top-level ``List<Struct>`` field, for example ``"DRF16BIT"``.
    key : str
        Numeric child used for matching, for example ``"CHNM"``.
    values : tuple of float
        Exact finite values to retain, compared with the key's physical value
        also when the key is in ``raw_code_paths``. Source order and
        duplicates are kept.

    """

    field: str
    key: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Normalize values so the Rust boundary receives an immutable tuple."""
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


@dataclass(frozen=True)
class MessageInfo:
    """Section 0-3 header of one BUFR message, read without decoding its data.

    ``source_message_index`` is the message's position in the file, the value
    decoded batches carry in their ``_source_message_index`` column. ``offset``
    and ``size_bytes`` locate its bytes. ``table_epoch`` indexes
    ``FileInfo.table_epochs`` for messages that carry data and is None for DX
    table messages and messages without subsets.
    """

    source_message_index: int
    offset: int
    size_bytes: int
    edition: int
    master_table: int
    center: int
    sub_center: int
    update_sequence: int
    optional_section_present: bool
    data_category: int
    international_sub_category: int
    local_sub_category: int
    master_table_version: int
    local_table_version: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    subsets: int
    compressed: bool
    section3_flags: int
    descriptors: tuple[str, ...]
    table_epoch: int | None


@dataclass(frozen=True)
class TableRedefinition:
    """A descriptor an embedded DX block defines differently from an earlier block.

    For Table B, ``before`` and ``after`` hold ``mnemonic``, ``name``,
    ``units``, ``scale``, ``reference_value`` and ``bit_width``. For Table D
    they hold ``mnemonic`` and ``descriptors``.
    """

    descriptor: str
    table: str
    before: dict[str, Any]
    after: dict[str, Any]


@dataclass(frozen=True)
class TableEpoch:
    """The tables that decode a run of data messages.

    A file can concatenate several dumps, each with its own embedded DX block,
    and each block's tables decode the messages that follow it.
    ``embedded_table_sha256`` identifies the block, or is None when messages
    decode with the base tables alone. ``redefines`` lists descriptors the
    block defines differently from earlier blocks.
    """

    embedded_table_sha256: str | None
    redefines: tuple[TableRedefinition, ...]


@dataclass(frozen=True, eq=False, init=False)
class FileInfo:
    """Message headers and table epochs of a BUFR file, read without decoding data.

    The headers are held as an internal Arrow table, one row per message in
    file order, including DX table messages and messages without subsets.
    ``messages`` and ``data_messages`` build ``MessageInfo`` objects from it on
    first access. ``FileInfo(messages, table_epochs)`` builds the table from
    ``MessageInfo`` objects.
    """

    _message_table: pa.RecordBatch
    table_epochs: tuple[TableEpoch, ...]

    def __init__(self, messages: tuple[MessageInfo, ...], table_epochs: tuple[TableEpoch, ...]) -> None:
        """Describe a file by its messages, in file order, and its table epochs."""
        object.__setattr__(self, "_message_table", _message_table(messages))
        object.__setattr__(self, "table_epochs", table_epochs)
        self.__dict__["messages"] = messages

    @classmethod
    def _from_message_table(cls, message_table: pa.RecordBatch, table_epochs: tuple[TableEpoch, ...]) -> FileInfo:
        info = cls.__new__(cls)
        object.__setattr__(info, "_message_table", message_table)
        object.__setattr__(info, "table_epochs", table_epochs)
        return info

    @cached_property
    def messages(self) -> tuple[MessageInfo, ...]:
        """Every message in file order, including DX table messages and messages without subsets."""
        # The table's columns are the MessageInfo fields in declaration order.
        return tuple(MessageInfo(*row) for row in _get_lib().message_rows_rust(self._message_table))

    @cached_property
    def data_messages(self) -> tuple[MessageInfo, ...]:
        """Messages that carry subsets, excluding DX table messages."""
        return tuple(message for message in self.messages if message.table_epoch is not None)

    def __eq__(self, other: object) -> bool:
        """Equal when both describe the same messages and table epochs."""
        if not isinstance(other, FileInfo):
            return NotImplemented
        return self._message_table.equals(other._message_table) and self.table_epochs == other.table_epochs


@dataclass(frozen=True)
class DecodeFailure:
    """A message skipped by ``skip_truncated_messages``.

    ``FileInfo.messages[source_message_index]`` locates its bytes.
    """

    source_message_index: int
    reason: str


def _file_info(native: tuple[Any, str]) -> FileInfo:
    import pyarrow as pa

    message_table, table_epochs_json = native
    table_epochs = tuple(
        TableEpoch(
            embedded_table_sha256=epoch["embedded_table_sha256"],
            redefines=tuple(TableRedefinition(**item) for item in epoch["redefines"]),
        )
        for epoch in json.loads(table_epochs_json)
    )
    return FileInfo._from_message_table(pa.record_batch(message_table), table_epochs)


def _message_table(messages: tuple[MessageInfo, ...]) -> pa.RecordBatch:
    import pyarrow as pa

    # The column types of the table `open_bufr` and `get_metadata` build in Rust.
    u8, u16, u64 = pa.uint8(), pa.uint16(), pa.uint64()
    types = {
        "source_message_index": u64,
        "offset": u64,
        "size_bytes": u64,
        "center": u16,
        "sub_center": u16,
        "year": u16,
        "subsets": u16,
        "optional_section_present": pa.bool_(),
        "compressed": pa.bool_(),
        "descriptors": pa.list_(pa.string()),
        "table_epoch": u64,
    }
    names = [field.name for field in fields(MessageInfo)]
    schema = pa.schema([pa.field(name, types.get(name, u8), nullable=name == "table_epoch") for name in names])
    # Cast, so integer flags such as 0 and 1 are accepted for the boolean columns.
    columns = [pa.array([getattr(message, field.name) for message in messages]).cast(field.type) for field in schema]
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def _native_list_filters(list_filters: list[ListValueFilter] | None) -> list[tuple[str, str, list[float]]] | None:
    if list_filters is None:
        return None
    return [(item.field, item.key, list(item.values)) for item in list_filters]


class BatchIterator:
    """Bounded Arrow batches from one :meth:`BufrFile.iter_batches` call, in file order."""

    def __init__(self, native: Any) -> None:
        self._native = native

    def __iter__(self) -> BatchIterator:
        """Return this iterator."""
        return self

    def __next__(self) -> pa.RecordBatch:
        """Decode and return the next batch."""
        import pyarrow as pa

        return pa.record_batch(next(self._native))

    @property
    def decode_failures(self) -> tuple[DecodeFailure, ...]:
        """Messages skipped by ``skip_truncated_messages`` so far in this iteration."""
        return tuple(DecodeFailure(**failure) for failure in json.loads(self._native.decode_failures_json()))


class BufrFile:
    """A BUFR file indexed by :func:`open_bufr`; message bytes are read as they are decoded.

    Attributes
    ----------
    path : Path
        The file this handle was opened from.

    """

    def __init__(self, path: Path, handle: Any) -> None:
        self.path = path
        self._handle = handle

    def __enter__(self) -> BufrFile:
        """Return this handle for use as a context manager."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release the file, as :meth:`close` does."""
        self.close()

    def close(self) -> None:
        """Release the open file and index; iterators created earlier keep their own reference."""
        self._handle = None

    def _native(self) -> Any:
        if self._handle is None:
            raise ValueError(f"{self.path} is closed")
        return self._handle

    def iter_batches(
        self,
        *,
        messages_per_batch: int | None = None,
        projection: list[str] | None = None,
        raw_code_paths: list[str] | None = None,
        data_category_filter: list[int] | None = None,
        list_filters: list[ListValueFilter] | None = None,
        prefetch_batches: int = 1,
        target_decoded_bytes: int | None = None,
        fixed_replication: str = "flat",
        skip_truncated_messages: bool = False,
        projection_missing: str = "raise",
    ) -> BatchIterator:
        """Decode the file as bounded Arrow batches in message order.

        A batch holds consecutive data messages that share data category,
        descriptor template, compression and table epoch, so a template change
        ends a batch early. Every batch carries ``_data_category``,
        ``_source_message_index`` and ``_source_subset_index`` columns. Options
        are validated when this method is called.

        Parameters
        ----------
        messages_per_batch : int, optional
            Maximum data messages per batch. Defaults to 512, or to no limit
            when ``target_decoded_bytes`` is given.
        projection : list of str, optional
            Fields to keep. Flat mode accepts ``FIELD`` and ``FIELD.CHILD``;
            structural mode accepts paths of any depth.
        raw_code_paths : list of str, optional
            Numeric fields to emit as stored integers, with BUFR missing values
            as nulls. Each must also be kept by ``projection`` when one is given.
        data_category_filter : list of int, optional
            Data categories to decode; other messages are skipped.
        list_filters : list of ListValueFilter, optional
            Keep only the entries of repeated list fields whose key is in
            ``values``, in file order. ``values`` are compared with the key's
            decoded value, also for a key listed in ``raw_code_paths``. Flat
            layout only; each filtered field must be kept by ``projection``
            when one is given.
        prefetch_batches : int, default 1
            Batches a background thread decodes ahead of the consumer, counting
            the one being decoded and not the one last returned. Without
            ``target_decoded_bytes``, free slots are filled by decoding several
            batches in parallel. With a single decode thread, each batch is
            decoded during the ``next()`` call that returns it and none are
            decoded ahead.
        target_decoded_bytes : int, optional
            Soft budget for decoded Arrow bytes per batch, calibrated on the
            first message of each template and refreshed from each decoded
            batch. A batch also stops once its encoded message bytes reach the
            budget. Messages are never split; a batch still stops at a given
            ``messages_per_batch``. Batches are then decoded one at a time, and
            the batches decoded ahead, counting the one being decoded, hold
            about ``prefetch_batches * target_decoded_bytes``.
        fixed_replication : {"flat", "structural"}, default "flat"
            ``"flat"`` emits each fixed-replication occurrence as its own column;
            ``"structural"`` emits each fixed replication as ``FixedSizeList<Struct>``.
        skip_truncated_messages : bool, default False
            Skip messages whose template extends past their Section 4 payload
            and record them in :attr:`BatchIterator.decode_failures`. Other
            decode errors still raise.
        projection_missing : {"raise", "skip"}, default "raise"
            What to do with a ``projection`` path that a decoded template does
            not have. ``"raise"`` fails the decode; ``"skip"`` leaves the field
            out of that template's batches, together with the
            ``raw_code_paths`` and ``list_filters`` under it.

        Returns
        -------
        BatchIterator
            Iterator of ``pyarrow.RecordBatch``.

        Raises
        ------
        ValueError
            If an option is invalid or the file is closed. Iterating raises
            ValueError when a message fails to decode; the error starts with
            ``Source BUFR message N:`` or ``Source BUFR message group A..B:``.
        RuntimeError
            When iterating, if the decode thread pool or the background decode
            thread cannot start.

        """
        return BatchIterator(
            self._native().iter_batches(
                messages_per_batch,
                projection,
                raw_code_paths,
                data_category_filter,
                _native_list_filters(list_filters),
                prefetch_batches,
                target_decoded_bytes,
                fixed_replication,
                skip_truncated_messages,
                projection_missing,
            )
        )

    @cached_property
    def info(self) -> FileInfo:
        """Message headers and table epochs of the file."""
        return _file_info(self._native().file_info())

    def source_structure(self, source_message_index: int) -> dict[str, Any]:
        """Describe the template, tables and effective packing that decode one message.

        Parameters
        ----------
        source_message_index : int
            Position of a data message in the file, as in ``MessageInfo.source_message_index``.

        Returns
        -------
        dict
            ``format_version``, ``embedded_table_sha256`` (the message's table
            epoch), ``observation_descriptors``, ``active_sequences``,
            ``active_elements``, ``effective_occurrences`` (packing after
            operators) and ``decoded_physical_schema`` (flat layout).

        """
        return json.loads(self._native().source_structure_json(source_message_index))


def open_bufr(
    file_path: str | Path,
    *,
    table_b_json: str | None = None,
    table_d_json: str | None = None,
    local_tables_json: str | None = None,
    strict: bool = False,
) -> BufrFile:
    """Index a BUFR file for bounded Arrow decoding.

    Opening reads message headers and DX table messages. Data messages are read
    from the file when they are decoded, so the file must not change while the
    handle or its iterators are in use.

    Parameters
    ----------
    file_path : str or Path
        BUFR file to open.
    table_b_json, table_d_json : str, optional
        WMO Table B/D JSON. The bundled tables are used by default.
    local_tables_json : str, optional
        Table overrides applied after embedded DX tables.
    strict : bool, default False
        Reject unknown descriptors and unsupported operators when opening, and
        while decoding reject non-printable IA5, compressed IA5 increments,
        BYTCNT subset frames that overrun, and reads past Section 4.

    Returns
    -------
    BufrFile

    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"BUFR file not found: {path}")
    if table_b_json is None or table_d_json is None:
        default_b, default_d = _load_default_tables()
        table_b_json = table_b_json or default_b
        table_d_json = table_d_json or default_d
    handle = _get_lib().open_bufr_rust(str(path), table_b_json, table_d_json, local_tables_json, strict)
    return BufrFile(path, handle)


def read_bufr(
    file_path: str | Path,
    *,
    projection: list[str] | None = None,
    list_filters: list[ListValueFilter] | None = None,
    raw_code_paths: list[str] | None = None,
    data_category_filter: list[int] | None = None,
    fixed_replication: str = "flat",
    skip_truncated_messages: bool = False,
    projection_missing: str = "raise",
    strict: bool = False,
    table_b_json: str | None = None,
    table_d_json: str | None = None,
    local_tables_json: str | None = None,
    flatten: bool = False,
    as_pandas: bool = False,
):
    """Read a BUFR file and return decoded data.

    NCEP category-11 DX messages are loaded automatically as embedded local
    descriptor tables and are not returned as data. Explicit
    ``local_tables_json`` definitions are applied after embedded definitions.

    ``projection``, ``list_filters``, ``raw_code_paths``,
    ``data_category_filter``, ``fixed_replication``,
    ``skip_truncated_messages`` and ``projection_missing`` mean what they mean in
    :meth:`BufrFile.iter_batches`, are validated the same way when this
    function is called, and apply while messages are decoded.

    Parameters
    ----------
    file_path : str or Path
        Path to the BUFR file to read.
    projection : list of str, optional
        Fields to keep; unknown fields raise. Flat mode accepts ``FIELD`` and
        ``FIELD.CHILD``; structural mode accepts paths of any depth.
        ``_data_category`` is always kept.
    list_filters : list of ListValueFilter, optional
        Keep only the entries of repeated list fields whose key is in
        ``values``, in file order. ``values`` are compared with the key's
        decoded value, also for a key listed in ``raw_code_paths``. Flat layout
        only; each filtered field must be kept by ``projection`` when one is
        given.
    raw_code_paths : list of str, optional
        Numeric fields to emit as stored integers, with BUFR missing values as
        nulls. Each must also be kept by ``projection`` when one is given.
    data_category_filter : list of int, optional
        Data categories to decode, which also orders the categories within a
        table epoch. If None, all categories are read.
    fixed_replication : {"flat", "structural"}, default "flat"
        ``"flat"`` emits each fixed-replication occurrence as its own column;
        ``"structural"`` emits each fixed replication as ``FixedSizeList<Struct>``.
    skip_truncated_messages : bool, default False
        Omit messages whose template extends past their Section 4 payload. The
        skipped messages are not reported; :attr:`BatchIterator.decode_failures`
        from :meth:`BufrFile.iter_batches` lists them. Other decode errors
        still raise.
    projection_missing : {"raise", "skip"}, default "raise"
        ``"skip"`` leaves projected fields a template does not have out of
        that template's batches instead of raising.
    strict : bool, default False
        As in :func:`open_bufr`.
    table_b_json : str, optional
        JSON string containing Table B definitions. If None, bundled tables are used.
    table_d_json : str, optional
        JSON string containing Table D definitions. If None, bundled tables are used.
    local_tables_json : str, optional
        JSON string containing local table overrides (center-specific definitions).
        Explicit overrides take precedence over embedded DX definitions.
    flatten : bool, default False
        If True, flatten nested list columns into a fully-scalar table. Only a
        multi-element ``List<Struct<...>>`` whose inner struct has a ``CAT`` field
        (the NCEP PrepBUFR vertical-level marker) is row-expanded — each level
        becomes a row, with scalar columns broadcast. All other list columns,
        including multi-element non-CAT lists, are collapsed by taking element 0
        (their repeated values are not preserved as separate rows). See
        :func:`flatten_batch` for the full semantics. Requires
        ``fixed_replication="flat"``.
    as_pandas : bool, default False
        If True, return pandas DataFrames. If False, return PyArrow RecordBatches.

    Returns
    -------
    list of pandas.DataFrame or list of pyarrow.RecordBatch
        One element per group: per table epoch in file order, data categories
        in ``data_category_filter`` order or first appearance, then each run of
        one descriptor template within a category. A data category can
        therefore appear in more than one element, and a group whose messages
        were all skipped is omitted. Each element has a ``_data_category``
        column and no ``_source_message_index`` or ``_source_subset_index``
        columns.

    Raises
    ------
    FileNotFoundError
        If the BUFR file does not exist.
    ValueError
        If an option is invalid, no BUFR messages are found, no data message
        is found or none matches ``data_category_filter``, or decoding fails.
        A decode error starts with ``Source BUFR message N:``, naming the
        failing message, or ``Source BUFR message group A..B:``, naming the
        first and last of the messages decoded with it.
    RuntimeError
        If the decode thread pool cannot start.

    Examples
    --------
    Read all data from a BUFR file:

    >>> batches = read_bufr("synop.bufr")
    >>> print(batches[0].num_rows)

    Keep specific fields:

    >>> batches = read_bufr("synop.bufr", projection=["B005001", "B006001", "B012101"])

    Keep 2 channels of a radiance list:

    >>> batches = read_bufr(
    ...     "gdas.crisf4.bufr_d",
    ...     projection=["CLATH", "CLONH", "DRF16BIT"],
    ...     list_filters=[ListValueFilter("DRF16BIT", "CHNM", (19, 49))],
    ... )

    Filter by data category:

    >>> batches = read_bufr(
    ...     "mixed.bufr",
    ...     data_category_filter=[0, 1]  # Surface data only
    ... )

    One row per PrepBUFR level:

    >>> batches = read_bufr(
    ...     "gdas.prepbufr",
    ...     flatten=True  # expand the CAT level list into rows
    ... )

    """
    import pyarrow as pa

    if flatten and fixed_replication == "structural":
        raise ValueError("flatten=True requires fixed_replication='flat'")
    with open_bufr(
        file_path,
        table_b_json=table_b_json,
        table_d_json=table_d_json,
        local_tables_json=local_tables_json,
        strict=strict,
    ) as source:
        groups = source._native().read_groups(
            data_category_filter,
            projection,
            raw_code_paths,
            _native_list_filters(list_filters),
            fixed_replication,
            skip_truncated_messages,
            projection_missing,
        )

    results = []
    for group in groups:
        batch = pa.record_batch(group)
        if flatten:
            batch = flatten_batch(batch)
        results.append(batch.to_pandas() if as_pandas else batch)
    return results


def get_metadata(file_path: str | Path) -> FileInfo:
    """Read a BUFR file's message headers and table epochs without decoding data.

    Reads Sections 0-3, the Section 4 length and the end marker of each
    message, and DX table messages (data category 11) whole, so the cost
    scales with the number of messages rather than the file size. Raises
    ValueError when the file holds no complete BUFR message or a message
    header cannot be read.

    Parameters
    ----------
    file_path : str or Path
        Path to the BUFR file.

    Returns
    -------
    FileInfo
        Equal to ``open_bufr(file_path).info``.

    Examples
    --------
    >>> info = earth2bufr.get_metadata("observations.bufr")
    >>> for message in info.data_messages:
    ...     print(message.data_category, message.subsets, message.descriptors[:2])

    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"BUFR file not found: {path}")
    return _file_info(_get_lib().file_info_rust(str(path)))


def set_num_threads(threads: int) -> None:
    """Set the number of threads that decode BUFR data.

    Replaces the decode thread pool. A decode already running finishes on the
    pool it started on; the next batch decoded by any iterator, and each later
    :func:`read_bufr` call, uses the new pool. With one thread, an iterator
    started afterwards decodes each batch on the thread that calls ``next()``.

    Parameters
    ----------
    threads : int
        Number of decode threads, at least 1.

    Raises
    ------
    ValueError
        If ``threads`` is less than 1.
    RuntimeError
        If a pool of ``threads`` threads cannot start.

    Examples
    --------
    >>> earth2bufr.set_num_threads(16)
    >>> earth2bufr.get_num_threads()
    16

    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    _get_lib().set_num_threads_rust(threads)


def get_num_threads() -> int:
    """Return the number of threads that decode BUFR data.

    Before any :func:`set_num_threads` call this is ``EARTH2BUFR_NUM_THREADS``,
    else ``RAYON_NUM_THREADS``, else the number of CPUs available to the
    process up to 32, read when the first decode or this call builds the pool.
    Unset, non-integer and zero values are skipped.

    Returns
    -------
    int
        Number of decode threads.

    Raises
    ------
    RuntimeError
        If this call builds the decode thread pool and it cannot start.

    """
    return _get_lib().get_num_threads_rust()


def load_tables(
    table_b_path: str | Path | None = None,
    table_d_path: str | Path | None = None,
) -> tuple[str, str]:
    """Load Table B and Table D from JSON files.

    Parameters
    ----------
    table_b_path : str or Path, optional
        Path to Table B JSON file. If None, uses bundled default.
    table_d_path : str or Path, optional
        Path to Table D JSON file. If None, uses bundled default.

    Returns
    -------
    tuple of (str, str)
        JSON strings for Table B and Table D.

    """
    if table_b_path is None and table_d_path is None:
        return _load_default_tables()

    if table_b_path is not None:
        table_b_json = Path(table_b_path).read_text()
    else:
        table_b_json, _ = _load_default_tables()

    if table_d_path is not None:
        table_d_json = Path(table_d_path).read_text()
    else:
        _, table_d_json = _load_default_tables()

    return table_b_json, table_d_json


def create_local_tables(
    table_b_overrides: dict | None = None,
    table_d_overrides: dict | None = None,
) -> str:
    """Create a local tables JSON string from dictionaries.

    Parameters
    ----------
    table_b_overrides : dict, optional
        Dictionary of Table B entries to add/override.
        Keys should be descriptor strings like "0-01-001".
        Values should be dicts with: mnemonic, name, units, scale,
        reference_value, bit_width.
    table_d_overrides : dict, optional
        Dictionary of Table D entries to add/override.
        Keys should be descriptor strings like "3-01-001".
        Values should be dicts with: mnemonic, descriptors (list of ints).

    Returns
    -------
    str
        JSON string suitable for passing to read_bufr's local_tables_json.

    Examples
    --------
    >>> local = create_local_tables(
    ...     table_b_overrides={
    ...         "0-01-192": {
    ...             "mnemonic": "CUSTOM",
    ...             "name": "Custom local descriptor",
    ...             "units": "NUMERIC",
    ...             "scale": 0,
    ...             "reference_value": 0,
    ...             "bit_width": 16,
    ...         }
    ...     }
    ... )
    >>> df = read_bufr("file.bufr", local_tables_json=local)

    """
    local = {
        "table_b": table_b_overrides or {},
        "table_d": table_d_overrides or {},
    }
    return json.dumps(local)


__all__ = [
    "__version__",
    "BatchIterator",
    "BufrFile",
    "DecodeFailure",
    "FileInfo",
    "MessageInfo",
    "TableEpoch",
    "TableRedefinition",
    "ListValueFilter",
    "open_bufr",
    "read_bufr",
    "get_metadata",
    "set_num_threads",
    "get_num_threads",
    "load_tables",
    "create_local_tables",
    "flatten_batch",
]
