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

"""Focused tests for native bounded Arrow decoding."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from synthetic.builders import (
    AIR_TEMPERATURE,
    AIRCRAFT_FLIGHT_NUMBER,
    ASSOCIATED_FIELD_SIGNIFICANCE,
    NON_COORDINATE_PRESSURE,
    RELATIVE_HUMIDITY,
    WIND_SPEED,
    YEAR,
    BitWriter,
    Delayed,
    Fixed,
    bufr_message,
    build_bytcnt_framed_messages,
    build_compressed_delayed_replication_message,
    build_compressed_station_name_group_message,
    build_delayed_replication_message,
    build_fixed_replication_operator_message,
    build_flat_wmo_message,
    build_nested_delayed_replication_message,
    build_nested_fixed_replication_message,
    build_synthetic_prepbufr_ia5_message,
    build_template_message,
    build_two_epoch_prepbufr_messages,
    delayed_replication_case,
    desc,
    desc_name,
    edition3_surface_case,
    missing_value_case,
    surface_observation_case,
)


def _write_messages(tmp_path: Path, name: str, messages: list[bytes]) -> Path:
    path = tmp_path / name
    path.write_bytes(b"".join(messages))
    return path


def _zero_subset_message(*, compressed: bool = False) -> bytes:
    return bufr_message(
        [],
        b"",
        num_subsets=0,
        data_category=0,
        compressed=compressed,
    )


def _section4_start(message: bytes | bytearray) -> int:
    section3_start = 8 + int.from_bytes(message[8:11], "big")
    return section3_start + int.from_bytes(message[section3_start : section3_start + 3], "big")


def _truncate_section4(message: bytes, dropped_bytes: int = 1) -> bytes:
    data = bytearray(message)
    section4_start = _section4_start(data)
    section4_length = int.from_bytes(data[section4_start : section4_start + 3], "big")
    del data[section4_start + section4_length - dropped_bytes : section4_start + section4_length]
    data[section4_start : section4_start + 3] = (section4_length - dropped_bytes).to_bytes(3, "big")
    data[4:7] = (len(data)).to_bytes(3, "big")
    return bytes(data)


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    return batch.column(batch.schema.get_field_index(name))


PROVENANCE_COLUMNS = ("_source_message_index", "_source_subset_index")


def _assert_documented_batch_bounds(
    batches, *, messages_per_batch: int, message_sizes: tuple[int, ...] = (), target: int | None = None
) -> None:
    """Check the batch bounds iter_batches documents.

    A batch holds at most ``messages_per_batch`` messages, in source order, and
    no message spans two batches. With a byte target, a batch stops once its
    encoded message bytes reach the target.
    """
    seen: list[int] = []
    for batch in batches:
        messages = list(dict.fromkeys(_column(batch, "_source_message_index").to_pylist()))
        assert len(messages) <= messages_per_batch
        assert not set(messages) & set(seen)
        seen.extend(messages)
        if target is not None:
            assert sum(message_sizes[index] for index in messages[:-1]) < target
    assert seen == sorted(seen)


def _message_sizes(source) -> tuple[int, ...]:
    return tuple(message.size_bytes for message in source.info.messages)


def _first_without_provenance(batches) -> pa.RecordBatch:
    batch = next(batches)
    return batch.select([i for i, name in enumerate(batch.schema.names) if name not in PROVENANCE_COLUMNS])


@pytest.mark.rust
def test_bounded_batches_preserve_source_order_and_bounds(tmp_path: Path):
    """Compatible messages are bounded without changing row order."""
    import earth2bufr

    message = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "five_messages.bufr", [message] * 5)

    with earth2bufr.open_bufr(path) as source:
        batches = list(source.iter_batches(messages_per_batch=2))
        whole = list(source.iter_batches())

    _assert_documented_batch_bounds(batches, messages_per_batch=2)
    joined = pa.Table.from_batches(batches)
    assert joined.equals(pa.Table.from_batches(whole))
    assert joined.column("_source_message_index").to_pylist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert joined.column("_source_subset_index").to_pylist() == [0, 1] * 5
    assert batches[0].schema.field("_source_message_index").type == pa.uint64()
    assert batches[0].schema.field("_source_subset_index").type == pa.uint32()


@pytest.mark.rust
@pytest.mark.parametrize("strict", [True, False])
def test_bounded_batches_can_quarantine_one_truncated_message(tmp_path: Path, strict: bool):
    """Truncation quarantine preserves surrounding messages and identity in both modes."""
    import earth2bufr

    valid = build_flat_wmo_message(surface_observation_case())
    truncated = _truncate_section4(valid)
    path = _write_messages(
        tmp_path,
        "one_truncated_message.bufr",
        [valid, truncated, valid],
    )

    with earth2bufr.open_bufr(path, strict=strict) as source:
        if strict:
            with pytest.raises(ValueError):
                list(source.iter_batches(messages_per_batch=3))
        iterator = source.iter_batches(messages_per_batch=3, skip_truncated_messages=True)
        batches = list(iterator)
        failures = iterator.decode_failures

    assert [value for batch in batches for value in _column(batch, "_source_message_index").to_pylist()] == [0, 0, 2, 2]
    assert len(failures) == 1
    assert failures[0].source_message_index == 1
    assert failures[0].reason
    one_shot = earth2bufr.read_bufr(path, strict=strict, skip_truncated_messages=True)
    assert [batch.num_rows for batch in one_shot] == [4]


@pytest.mark.rust
def test_file_changed_after_open_raises(tmp_path: Path):
    """Message bytes are read at decode time and checked against the index built at open."""
    import earth2bufr

    message = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "changed.bufr", [message] * 3)

    with earth2bufr.open_bufr(path) as source:
        path.write_bytes(message + b"XXXX" + message[4:] + message)
        with pytest.raises(ValueError, match=r"\b1\b"):
            list(source.iter_batches())
        path.write_bytes(message)
        with pytest.raises(ValueError):
            list(source.iter_batches())
        assert len(source.info.messages) == 3


@pytest.mark.rust
def test_read_bufr_raises_on_a_truncated_message_in_strict_mode(tmp_path: Path):
    """Without skip_truncated_messages, strict read_bufr raises on a truncated message."""
    import earth2bufr

    valid = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "one_truncated_message.bufr", [valid, _truncate_section4(valid), valid])

    with pytest.raises(ValueError, match=r"^Source BUFR message group 0\.\.2: "):
        earth2bufr.read_bufr(path, strict=True)


@pytest.mark.rust
@pytest.mark.parametrize("strict", [True, False])
# Each subset frame is 3 bytes: dropping 1 byte cuts the last frame, 2 leave one
# byte of its BYTCNT, 3 remove it at the frame boundary.
@pytest.mark.parametrize("dropped_bytes", [1, 2, 3])
def test_bounded_batches_quarantine_bytcnt_frame_past_section4(tmp_path: Path, strict: bool, dropped_bytes: int):
    """A BYTCNT subset frame that overruns Section 4 is quarantined as truncation."""
    import earth2bufr

    dx, valid = build_bytcnt_framed_messages([b"\x07", b"\x09"])
    truncated = _truncate_section4(valid, dropped_bytes)
    path = _write_messages(tmp_path, "bytcnt_truncated.bufr", [dx, valid, truncated, valid])

    with earth2bufr.open_bufr(path, strict=strict) as source:
        iterator = source.iter_batches(messages_per_batch=4, skip_truncated_messages=True)
        batches = list(iterator)
        failures = iterator.decode_failures

    assert [value for batch in batches for value in _column(batch, "_source_message_index").to_pylist()] == [1, 1, 3, 3]
    assert [failure.source_message_index for failure in failures] == [2]
    assert failures[0].reason
    one_shot = earth2bufr.read_bufr(path, strict=strict, skip_truncated_messages=True)
    assert sum(batch.num_rows for batch in one_shot) == 4


@pytest.mark.rust
def test_many_consecutive_skipped_messages_yield_the_next_batch(tmp_path: Path):
    """Thousands of fully skipped one-message batches in a row are passed over in one next() call."""
    import earth2bufr

    valid = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "mostly_truncated.bufr", [_truncate_section4(valid)] * 5000 + [valid])

    with earth2bufr.open_bufr(path) as source:
        iterator = source.iter_batches(messages_per_batch=1, skip_truncated_messages=True)
        batches = list(iterator)

    assert len(batches) == 1
    assert set(_column(batches[0], "_source_message_index").to_pylist()) == {5000}
    assert len(iterator.decode_failures) == 5000


@pytest.mark.rust
def test_read_bufr_omits_a_group_whose_messages_were_all_skipped(tmp_path: Path):
    """A template run made only of truncated messages leaves no batch in read_bufr."""
    import earth2bufr

    valid = build_flat_wmo_message(surface_observation_case())
    other_template = _truncate_section4(build_flat_wmo_message(missing_value_case()))
    path = _write_messages(tmp_path, "skipped_group.bufr", [valid, other_template, valid])

    batches = earth2bufr.read_bufr(path, skip_truncated_messages=True)

    assert [batch.num_rows for batch in batches] == [2, 2]
    assert all(batch.schema.equals(batches[0].schema) for batch in batches)


@pytest.mark.rust
def test_data_category_filter_selects_and_orders_categories(tmp_path: Path):
    """iter_batches keeps source order; read_bufr orders the categories of an epoch as the filter does."""
    import earth2bufr

    case = surface_observation_case()
    messages = [build_flat_wmo_message(replace(case, data_category=category)) for category in (0, 1, 0, 2)]
    path = _write_messages(tmp_path, "categories.bufr", messages)

    with earth2bufr.open_bufr(path) as source:
        streamed = pa.Table.from_batches(source.iter_batches(data_category_filter=[2, 0]))
    assert streamed.column("_source_message_index").to_pylist() == [0, 0, 2, 2, 3, 3]
    assert streamed.column("_data_category").to_pylist() == [0.0, 0.0, 0.0, 0.0, 2.0, 2.0]

    batches = earth2bufr.read_bufr(path, data_category_filter=[2, 0])
    assert [batch.column("_data_category").to_pylist() for batch in batches] == [[2.0, 2.0], [0.0] * 4]


@pytest.mark.rust
def test_unparseable_dx_message_fails_open(tmp_path: Path):
    """A DX table message whose Section 4 overruns the message fails open and names the message."""
    import earth2bufr

    dx, valid = build_bytcnt_framed_messages([b"\x07"])
    broken = bytearray(dx)
    section4_start = _section4_start(broken)
    broken[section4_start : section4_start + 3] = len(broken).to_bytes(3, "big")
    path = _write_messages(tmp_path, "broken_dx.bufr", [bytes(broken), valid])

    with pytest.raises(ValueError, match=r"\b0\b"):
        earth2bufr.open_bufr(path)


@pytest.mark.rust
def test_bytcnt_frame_skips_subset_padding(tmp_path: Path):
    """Each subset resumes at its declared BYTCNT end, past any padding bytes."""
    import earth2bufr

    dx, data = build_bytcnt_framed_messages([b"\x05\x00", b"\x07\x00"])
    path = _write_messages(tmp_path, "bytcnt_padded.bufr", [dx, data])

    with earth2bufr.open_bufr(path) as source:
        batches = list(source.iter_batches())

    assert [value for batch in batches for value in _column(batch, "VAL").to_pylist()] == [5, 7]


@pytest.mark.rust
@pytest.mark.parametrize("strict", [True, False])
def test_bytcnt_frame_bounds_a_field_that_overruns_its_subset(tmp_path: Path, strict: bool):
    """A field longer than its BYTCNT frame never reads into the next subset."""
    import earth2bufr

    dx, data = build_bytcnt_framed_messages([b"\x05", b"\x07"], value_bits=16)
    path = _write_messages(tmp_path, "bytcnt_short_subset.bufr", [dx, data])

    with earth2bufr.open_bufr(path, strict=strict) as source:
        if strict:
            with pytest.raises(ValueError):
                list(source.iter_batches())
            return
        iterator = source.iter_batches(skip_truncated_messages=True)
        batches = list(iterator)
        failures = iterator.decode_failures

    assert [value for batch in batches for value in _column(batch, "VAL").to_pylist()] == [None, None]
    assert failures == ()


@pytest.mark.rust
def test_decoded_byte_target_calibrates_then_respects_message_cap(tmp_path: Path):
    """Adaptive groups preserve messages and retain an independent hard cap."""
    import earth2bufr

    message = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "adaptive_messages.bufr", [message] * 5)

    with earth2bufr.open_bufr(path) as source:
        sizes = _message_sizes(source)
        whole = pa.Table.from_batches(source.iter_batches())
        tiny = list(
            source.iter_batches(
                target_decoded_bytes=1,
                messages_per_batch=8,
            )
        )
        capped = list(
            source.iter_batches(
                target_decoded_bytes=1 << 30,
                messages_per_batch=2,
            )
        )
        capped_ahead = list(source.iter_batches(target_decoded_bytes=1 << 30, messages_per_batch=2, prefetch_batches=4))

    _assert_documented_batch_bounds(tiny, messages_per_batch=8, message_sizes=sizes, target=1)
    _assert_documented_batch_bounds(capped, messages_per_batch=2, message_sizes=sizes, target=1 << 30)
    _assert_documented_batch_bounds(capped_ahead, messages_per_batch=2, message_sizes=sizes, target=1 << 30)
    for batches in (tiny, capped, capped_ahead):
        assert pa.Table.from_batches(batches).equals(whole)


@pytest.mark.rust
def test_decoded_byte_target_also_bounds_encoded_message_bytes(tmp_path: Path):
    """A narrow projection decodes few bytes per message; its batches still stop at the target in message bytes."""
    import earth2bufr

    subsets = [[2024, [[50_000.0 + level, 250.0] for level in range(40)]] for _ in range(3)]
    message = build_template_message(LEVELS, subsets, compressed=True)
    path = _write_messages(tmp_path, "wide_messages.bufr", [message] * 8)

    with earth2bufr.open_bufr(path) as source:
        sizes = _message_sizes(source)
        whole = list(source.iter_batches(projection=["B004001"]))
        bounded = list(source.iter_batches(projection=["B004001"], target_decoded_bytes=2 * len(message)))

    _assert_documented_batch_bounds(bounded, messages_per_batch=512, message_sizes=sizes, target=2 * len(message))
    assert pa.Table.from_batches(bounded).equals(pa.Table.from_batches(whole))


@pytest.mark.rust
def test_prefetched_batches_preserve_order_and_values(tmp_path: Path):
    """Concurrent compatibility groups are yielded in original source order."""
    import earth2bufr

    uncompressed = surface_observation_case()
    compressed = replace(uncompressed, name="compressed_surface", compressed=True)
    path = _write_messages(
        tmp_path,
        "prefetched_groups.bufr",
        [
            build_flat_wmo_message(compressed),
            build_flat_wmo_message(uncompressed),
            build_flat_wmo_message(compressed),
            build_flat_wmo_message(uncompressed),
        ],
    )

    with earth2bufr.open_bufr(path) as source:
        serial = list(source.iter_batches(messages_per_batch=8, prefetch_batches=1))
        prefetched = list(source.iter_batches(messages_per_batch=8, prefetch_batches=4))

    assert len(serial) == len(prefetched) == 4
    for expected, actual in zip(serial, prefetched, strict=True):
        assert actual.equals(expected)


def _thread_count() -> int:
    return sum(1 for _ in Path("/proc/self/task").iterdir())


@pytest.mark.rust
@pytest.mark.skipif(not Path("/proc/self/task").is_dir(), reason="counts threads through Linux /proc")
def test_early_break_stops_the_prefetch_thread(tmp_path: Path):
    """Dropping a partly consumed iterator stops its background thread, as docs/usage.md states."""
    import earth2bufr

    message = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "many_messages.bufr", [message] * 64)

    with earth2bufr.open_bufr(path) as source:
        list(source.iter_batches(messages_per_batch=1, prefetch_batches=4))
        threads_before = _thread_count()
        for _ in source.iter_batches(messages_per_batch=1, prefetch_batches=4):
            break
    assert _thread_count() == threads_before


@pytest.mark.rust
def test_decode_error_follows_the_batches_before_it(tmp_path: Path):
    """A failing message raises at its own position in the stream."""
    import earth2bufr

    valid = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "error_mid_stream.bufr", [valid, valid, _truncate_section4(valid), valid])

    with earth2bufr.open_bufr(path, strict=True) as source:
        iterator = source.iter_batches(messages_per_batch=1, prefetch_batches=4)
        assert [_column(next(iterator), "_source_message_index")[0].as_py() for _ in range(2)] == [0, 1]
        with pytest.raises(ValueError, match=r"\b2\b"):
            next(iterator)


@pytest.mark.rust
def test_batches_can_be_taken_on_a_worker_thread(tmp_path: Path):
    """A consumer that calls next() one batch ahead on another thread sees the source order."""
    import earth2bufr

    message = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "worker_thread.bufr", [message] * 16)

    with earth2bufr.open_bufr(path) as source, ThreadPoolExecutor(max_workers=1) as pool:
        iterator = source.iter_batches(messages_per_batch=3, target_decoded_bytes=1, prefetch_batches=2)
        pending = pool.submit(next, iterator, None)
        indices = []
        while (batch := pending.result()) is not None:
            pending = pool.submit(next, iterator, None)
            indices.extend(_column(batch, "_source_message_index").to_pylist())
    assert indices == [index for index in range(16) for _ in range(2)]


_ONE_THREAD_CONSUMER = """
import json, sys
from pathlib import Path
import earth2bufr
if sys.argv[3] == "set":
    earth2bufr.set_num_threads(1)
indices, error = [], None
with earth2bufr.open_bufr(sys.argv[1], strict=True) as source:
    try:
        for batch in source.iter_batches(messages_per_batch=1, prefetch_batches=4):
            indices.append(batch.column("_source_message_index")[0].as_py())
    except ValueError as exc:
        error = str(exc)
changed = Path(sys.argv[2])
with earth2bufr.open_bufr(changed) as source:
    batches = source.iter_batches(messages_per_batch=1, prefetch_batches=4)
    next(batches)
    with changed.open("r+b") as file:
        file.seek(source.info.messages[1].offset)
        file.write(b"XXXX")
    try:
        next(batches)
        changed_raised = False
    except ValueError:
        changed_raised = True
num_threads = earth2bufr.get_num_threads()
print(json.dumps(dict(indices=indices, error=error, changed_raised=changed_raised, num_threads=num_threads)))
"""


@pytest.mark.rust
@pytest.mark.parametrize("setting", ["RAYON_NUM_THREADS", "EARTH2BUFR_NUM_THREADS", "set"])
def test_one_decode_thread_decodes_each_batch_in_its_own_next_call(tmp_path: Path, setting: str):
    """With one decode thread, batches and a decode error keep source order and nothing is decoded ahead.

    Message 1 is overwritten after the first next() returns; the second next() reads it from disk.
    """
    import json
    import os
    import re
    import subprocess
    import sys

    valid = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "one_thread.bufr", [valid] * 5 + [_truncate_section4(valid), valid])
    changed = _write_messages(tmp_path, "changed.bufr", [valid] * 4)
    env = {key: value for key, value in os.environ.items() if not key.endswith("_NUM_THREADS")}
    if setting != "set":
        env[setting] = "1"

    result = subprocess.run(
        [sys.executable, "-c", _ONE_THREAD_CONSUMER, str(path), str(changed), setting],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["indices"] == [0, 1, 2, 3, 4]
    assert re.search(r"\b5\b", report["error"])
    assert report["changed_raised"]
    assert report["num_threads"] == 1


def _num_threads_in_subprocess(env_overrides: dict[str, str]) -> int:
    import os
    import subprocess
    import sys

    env = {key: value for key, value in os.environ.items() if not key.endswith("_NUM_THREADS")}
    result = subprocess.run(
        [sys.executable, "-c", "import earth2bufr; print(earth2bufr.get_num_threads())"],
        env={**env, **env_overrides},
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout)


@pytest.mark.rust
def test_default_num_threads_follow_the_environment():
    """EARTH2BUFR_NUM_THREADS wins over RAYON_NUM_THREADS; invalid values are skipped; else all available CPUs."""
    import os

    default = _num_threads_in_subprocess({})
    # The available parallelism also honours cgroup CPU quotas.
    assert 1 <= default <= len(os.sched_getaffinity(0))
    assert _num_threads_in_subprocess({"EARTH2BUFR_NUM_THREADS": "3"}) == 3
    assert _num_threads_in_subprocess({"RAYON_NUM_THREADS": "5"}) == 5
    assert _num_threads_in_subprocess({"EARTH2BUFR_NUM_THREADS": "3", "RAYON_NUM_THREADS": "5"}) == 3
    assert _num_threads_in_subprocess({"EARTH2BUFR_NUM_THREADS": "0", "RAYON_NUM_THREADS": "5"}) == 5
    assert _num_threads_in_subprocess({"EARTH2BUFR_NUM_THREADS": "many"}) == default


@pytest.mark.rust
@pytest.mark.parametrize("threads", [0, -2])
def test_set_num_threads_rejects_fewer_than_one(threads: int):
    """set_num_threads raises ValueError below one thread and keeps the pool."""
    import earth2bufr

    before = earth2bufr.get_num_threads()
    with pytest.raises(ValueError, match="at least 1"):
        earth2bufr.set_num_threads(threads)
    assert earth2bufr.get_num_threads() == before


_THREAD_CHANGES = """
import json, sys
import pyarrow as pa
import earth2bufr
options = dict(messages_per_batch=3, prefetch_batches=2)
with earth2bufr.open_bufr(sys.argv[1]) as source:
    reference = pa.Table.from_batches(list(source.iter_batches(**options)))
    read_reference = pa.Table.from_batches(earth2bufr.read_bufr(sys.argv[1]))
    counts, equal = [], []
    for threads in (3, 1, 2):
        earth2bufr.set_num_threads(threads)
        counts.append(earth2bufr.get_num_threads())
        equal.append(pa.Table.from_batches(list(source.iter_batches(**options))).equals(reference))
        equal.append(pa.Table.from_batches(earth2bufr.read_bufr(sys.argv[1])).equals(read_reference))
    earth2bufr.set_num_threads(1)
    batches, changes = [], iter((4, 1, 2))
    for batch in source.iter_batches(**options):
        batches.append(batch)
        earth2bufr.set_num_threads(next(changes, 3))
    equal.append(pa.Table.from_batches(batches).equals(reference))
print(json.dumps(dict(counts=counts, equal=equal)))
"""


@pytest.mark.rust
def test_changing_num_threads_keeps_the_output(tmp_path: Path):
    """Thread changes between and during iterations give the same batches; get_num_threads returns the set value."""
    import json
    import subprocess
    import sys

    message = build_flat_wmo_message(surface_observation_case())
    path = _write_messages(tmp_path, "thread_changes.bufr", [message] * 20)

    result = subprocess.run(
        [sys.executable, "-c", _THREAD_CHANGES, str(path)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["counts"] == [3, 1, 2]
    assert all(report["equal"])


def _compressed_year_humidity_message(years: list[int], humidity: float | None) -> bytes:
    writer = BitWriter()
    for field, values in ((YEAR, years), (RELATIVE_HUMIDITY, [humidity] * len(years))):
        raw_values = [field.encode_raw(value) for value in values]
        reference = min(raw_values)
        increments = [raw - reference for raw in raw_values]
        increment_bits = max(increments).bit_length()
        writer.write_bits(reference, field.bit_width)
        writer.write_bits(increment_bits, 6)
        if increment_bits:
            for increment in increments:
                writer.write_bits(increment, increment_bits)
    descriptors = [YEAR.descriptor, RELATIVE_HUMIDITY.descriptor]
    return bufr_message(descriptors, writer.to_bytes(), num_subsets=len(years), compressed=True)


@pytest.mark.rust
def test_joined_batch_keeps_nulls_across_odd_message_boundaries(tmp_path: Path):
    """Validity bits of messages with odd subset counts land on the right rows of the joined batch."""
    import earth2bufr

    subset_counts = (3, 5, 1, 7, 9, 2, 11)
    messages = [
        _compressed_year_humidity_message([2000 + subset for subset in range(subsets)], None if position % 2 else 50.0)
        for position, subsets in enumerate(subset_counts)
    ]
    path = _write_messages(tmp_path, "odd_boundaries.bufr", messages)

    with earth2bufr.open_bufr(path) as source:
        (joined,) = source.iter_batches(messages_per_batch=len(messages))
        singles = list(source.iter_batches(messages_per_batch=1))

    expected = pa.Table.from_batches(singles).combine_chunks().to_batches()[0]
    humidity = _column(joined, "B013003")
    assert joined.num_rows == sum(subset_counts)
    assert humidity.null_count == 5 + 7 + 2
    assert humidity.equals(_column(expected, "B013003"))
    assert joined.equals(expected)


@pytest.mark.rust
def test_zero_subset_message_does_not_split_a_batch(tmp_path: Path):
    """An empty uncompressed message between compressed messages adds no rows and does not end the batch."""
    import earth2bufr

    uncompressed = surface_observation_case()
    compressed = replace(uncompressed, name="compressed_surface", compressed=True)
    path = _write_messages(
        tmp_path,
        "compressed_with_empty.bufr",
        [build_flat_wmo_message(compressed), _zero_subset_message(), build_flat_wmo_message(compressed)],
    )

    with earth2bufr.open_bufr(path) as source:
        batches = list(source.iter_batches(messages_per_batch=8))
        data_messages = source.info.data_messages

    assert len(batches) == 1
    assert batches[0].num_rows == 4
    assert _column(batches[0], "_source_message_index").to_pylist() == [0, 0, 2, 2]
    assert [(message.source_message_index, message.subsets) for message in data_messages] == [
        (0, 2),
        (2, 2),
    ]


@pytest.mark.rust
def test_true_mixed_compression_splits_at_compatibility_boundary(tmp_path: Path):
    """Nonzero-subset compressed and uncompressed messages retain source order."""
    import earth2bufr

    uncompressed = surface_observation_case()
    compressed = replace(uncompressed, name="compressed_surface", compressed=True)
    path = _write_messages(
        tmp_path,
        "mixed.bufr",
        [
            build_flat_wmo_message(compressed),
            _zero_subset_message(),
            build_flat_wmo_message(uncompressed),
        ],
    )

    with earth2bufr.open_bufr(path) as source:
        batches = list(source.iter_batches(messages_per_batch=8))

    assert len(batches) == 2
    assert _column(batches[0], "_source_message_index").to_pylist() == [0, 0]
    assert _column(batches[1], "_source_message_index").to_pylist() == [2, 2]
    assert pa.Table.from_batches(batches).column("B012101").to_pylist() == [
        280.15,
        295.4,
        280.15,
        295.4,
    ]

    assert earth2bufr.read_bufr(path)[0].num_rows == 4


@pytest.mark.rust
def test_nested_child_projection_preserves_list_alignment(tmp_path: Path):
    """Nested projection retains offsets while dropping unrequested children."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "nested.bufr",
        [build_delayed_replication_message(delayed_replication_case())],
    )
    with earth2bufr.open_bufr(path) as source:
        full = _first_without_provenance(source.iter_batches())
        projected = _first_without_provenance(
            source.iter_batches(
                projection=["B004001", "B031001.B010004"],
            )
        )

    assert projected.schema.names == ["B004001", "B031001", "_data_category"]
    full_list = _column(full, "B031001")
    projected_list = _column(projected, "B031001")
    assert full_list.offsets.to_pylist() == projected_list.offsets.to_pylist()
    assert projected_list.type.value_type.num_fields == 1
    assert projected_list.type.value_type.field(0).name == "B010004"
    assert projected_list.values.field("B010004").to_pylist() == [100_000.0, 85_000.0]


@pytest.mark.rust
@pytest.mark.parametrize(
    "build",
    [build_delayed_replication_message, build_compressed_delayed_replication_message],
    ids=["uncompressed", "compressed"],
)
def test_list_value_filter_preserves_rows_offsets_and_projection(tmp_path: Path, build):
    """A repeated-value filter keeps list identity and can hide its key."""
    import earth2bufr

    path = _write_messages(tmp_path, "nested.bufr", [build(delayed_replication_case())])
    with earth2bufr.open_bufr(path) as source:
        filtered = _first_without_provenance(
            source.iter_batches(
                projection=["B004001", "B031001.B012101"],
                list_filters=[
                    earth2bufr.ListValueFilter(
                        field="B031001",
                        key="B010004",
                        values=(85_000.0,),
                    )
                ],
            )
        )
    levels = _column(filtered, "B031001")
    assert filtered.num_rows == 1
    assert levels.offsets.to_pylist() == [0, 1]
    assert levels.type.value_type.num_fields == 1
    assert levels.values.field("B012101").to_pylist() == [270.0]


@pytest.mark.rust
@pytest.mark.parametrize("compressed", [False, True])
def test_list_value_filter_keeps_duplicate_keys_in_source_order(tmp_path: Path, compressed: bool):
    """Every entry whose key matches is kept, repeated keys included, in source order."""
    import earth2bufr

    subsets = [[2000, [[50_000.0, 250.0], [51_000.0, 251.0], [50_000.0, 252.0]]]]
    path = _write_messages(
        tmp_path, "repeated_keys.bufr", [build_template_message(LEVELS, subsets, compressed=compressed)]
    )
    keep = [earth2bufr.ListValueFilter("B031001", "B010004", (50_000.0,))]
    with earth2bufr.open_bufr(path) as source:
        batch = next(source.iter_batches(list_filters=keep))
    assert _column(batch, "B031001").to_pylist() == [
        [{"B010004": 50_000.0, "B012101": 250.0}, {"B010004": 50_000.0, "B012101": 252.0}]
    ]


@pytest.mark.rust
def test_compressed_raw_code_path_preserves_codes_and_field_metadata(tmp_path: Path):
    """A selected repeated numeric leaf bypasses physical-value scaling."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "compressed_raw_codes.bufr",
        [build_compressed_delayed_replication_message(delayed_replication_case())],
    )
    with earth2bufr.open_bufr(path) as source:
        physical = _first_without_provenance(
            source.iter_batches(
                projection=["B031001"],
            )
        )
        raw = _first_without_provenance(
            source.iter_batches(
                projection=["B031001"],
                raw_code_paths=["B031001.B012101"],
            )
        )

    physical_levels = _column(physical, "B031001")
    raw_levels = _column(raw, "B031001")
    assert physical_levels.values.field("B012101").to_pylist() == [280.15, 270.0]
    assert raw_levels.values.field("B012101").type == pa.int32()
    assert raw_levels.values.field("B012101").to_pylist() == [28_015, 27_000]
    assert raw_levels.values.field("B010004").to_pylist() == [100_000.0, 85_000.0]
    metadata = raw_levels.type.value_type.field("B012101").metadata
    assert metadata == {
        b"representation": b"bufr_raw_code",
        b"descriptor": b"0-12-101",
        b"scale": b"2",
        b"reference": b"0",
        b"bit_width": b"16",
        b"missing_code": b"65535",
        b"source_units": b"K",
        b"physical_formula": b"(code) * 1e-2",
    }


@pytest.mark.rust
def test_uncompressed_raw_code_path_preserves_missing_as_arrow_null(tmp_path: Path):
    """Raw output keeps source missing codes in the Arrow validity bitmap."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "raw_missing.bufr",
        [build_flat_wmo_message(missing_value_case())],
    )
    with earth2bufr.open_bufr(path) as source:
        batch = _first_without_provenance(
            source.iter_batches(
                projection=["B013003"],
                raw_code_paths=["B013003"],
            )
        )

    values = _column(batch, "B013003")
    assert values.type == pa.int32()
    assert values.null_count == 1
    assert values.to_pylist() == [None]


@pytest.mark.rust
def test_invalid_raw_code_requests_raise(tmp_path: Path):
    """Raw-code paths must be kept by projection and cannot be combined with list filters."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)
    with earth2bufr.open_bufr(path) as source, pytest.raises(ValueError):
        list(
            source.iter_batches(
                projection=["B004001"],
                raw_code_paths=["B012101"],
            )
        )

    nested = delayed_replication_case().write(tmp_path)
    raw_code_paths = ["B031001.B012101"]
    list_filters = [earth2bufr.ListValueFilter("B031001", "B010004", (85_000.0,))]
    with earth2bufr.open_bufr(nested) as source, pytest.raises(ValueError):
        source.iter_batches(raw_code_paths=raw_code_paths, list_filters=list_filters)
    with pytest.raises(ValueError):
        earth2bufr.read_bufr(nested, raw_code_paths=raw_code_paths, list_filters=list_filters)


@pytest.mark.rust
@pytest.mark.parametrize("compressed", [False, True])
def test_structural_fixed_replication_matches_flat_values(tmp_path: Path, compressed: bool):
    """Nested fixed groups preserve every flat occurrence in source order."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        f"nested_fixed_{compressed}.bufr",
        [build_nested_fixed_replication_message(compressed=compressed)],
    )
    with earth2bufr.open_bufr(path, strict=True) as source:
        flat = _first_without_provenance(source.iter_batches())
        structural = _first_without_provenance(
            source.iter_batches(
                fixed_replication="structural",
            )
        )

    outer_name = "fixed_1_05_002__occ_0"
    inner_name = "fixed_1_02_003__occ_0"
    assert structural.schema.names == [
        "B004001",
        outer_name,
        "_data_category",
    ]
    outer = _column(structural, outer_name)
    assert pa.types.is_fixed_size_list(outer.type)
    assert outer.type.list_size == 2
    assert outer.values.type.names == ["B011002", "B001003", inner_name]
    assert outer.values.field("B011002").to_pylist() == [1.0, 2.0, 1.2, 2.2]
    assert outer.values.field("B001003").to_pylist() == [2.0, 3.0, 4.0, 5.0]
    inner = outer.values.field(inner_name)
    assert pa.types.is_fixed_size_list(inner.type)
    assert inner.type.list_size == 3
    assert inner.values.field("B012101").to_pylist() == [
        280.0,
        281.0,
        None,
        282.0,
        283.0,
        284.0,
        280.5,
        281.5,
        None,
        282.5,
        283.5,
        284.5,
    ]
    assert inner.values.field("B001006").to_pylist() == [
        "FLT00001",
        "FLT00002",
        "FLT00003",
        "FLT00004",
        "FLT00005",
        "FLT00006",
        "FLT00001",
        "FLT00002",
        "FLT00003",
        "FLT00004",
        "FLT00005",
        "FLT00006",
    ]

    flat_winds = _columns_by_name(flat, "B011002")
    flat_regions = _columns_by_name(flat, "B001003")
    flat_temperatures = _columns_by_name(flat, "B012101")
    flat_flights = _columns_by_name(flat, "B001006")
    for outer_index in range(2):
        outer_indices = pa.array([outer_index, outer_index + 2], type=pa.int64())
        assert pc.take(outer.values.field("B011002"), outer_indices).equals(flat_winds[outer_index])
        assert pc.take(outer.values.field("B001003"), outer_indices).equals(flat_regions[outer_index])
        for inner_index in range(3):
            structural_indices = pa.array(
                [
                    outer_index * 3 + inner_index,
                    6 + outer_index * 3 + inner_index,
                ],
                type=pa.int64(),
            )
            flat_index = outer_index * 3 + inner_index
            assert pc.take(inner.values.field("B012101"), structural_indices).equals(flat_temperatures[flat_index])
            assert pc.take(inner.values.field("B001006"), structural_indices).equals(flat_flights[flat_index])


@pytest.mark.rust
@pytest.mark.parametrize("compressed", [False, True])
def test_structural_nested_delayed_replication_matches_flat(tmp_path: Path, compressed: bool):
    """Delayed groups nested in delayed groups decode identically in both layouts."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        f"nested_delayed_{compressed}.bufr",
        [build_nested_delayed_replication_message(compressed=compressed)],
    )
    with earth2bufr.open_bufr(path, strict=True) as source:
        flat = _first_without_provenance(source.iter_batches())
        structural = _first_without_provenance(source.iter_batches(fixed_replication="structural"))

    assert structural.schema.names == ["B004001", "fixed_1_01_002__occ_0", "B031001", "_data_category"]
    winds = _column(structural, "fixed_1_01_002__occ_0").values.field("B011002")
    assert winds.to_pylist() == [1.0, 2.0, 1.5, 2.5]
    levels = _column(structural, "B031001")
    assert levels.equals(_column(flat, "B031001"))
    assert levels.to_pylist() == [
        [
            {"B010004": 50_000.0, "B031001": [{"B012101": 280.0}]},
            {"B010004": 70_000.0, "B031001": [{"B012101": 281.0}, {"B012101": 282.0}, {"B012101": None}]},
        ],
        [
            {"B010004": 50_020.0, "B031001": [{"B012101": 280.5}]},
            {"B010004": 70_020.0, "B031001": [{"B012101": 281.5}, {"B012101": 282.5}, {"B012101": None}]},
        ],
    ]


@pytest.mark.rust
def test_compressed_nested_delayed_replication_decodes_raw_codes(tmp_path: Path):
    """Compressed delayed groups nested in delayed groups decode directly to Arrow, raw codes included."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "nested_delayed_compressed_raw.bufr",
        [build_nested_delayed_replication_message(compressed=True)],
    )
    with earth2bufr.open_bufr(path, strict=True) as source:
        batch = _first_without_provenance(source.iter_batches(raw_code_paths=["B031001.B010004"]))

    levels = _column(batch, "B031001")
    assert levels.values.field("B010004").type == pa.int32()
    assert levels.to_pylist() == [
        [
            {"B010004": 5_000, "B031001": [{"B012101": 280.0}]},
            {"B010004": 7_000, "B031001": [{"B012101": 281.0}, {"B012101": 282.0}, {"B012101": None}]},
        ],
        [
            {"B010004": 5_002, "B031001": [{"B012101": 280.5}]},
            {"B010004": 7_002, "B031001": [{"B012101": 281.5}, {"B012101": 282.5}, {"B012101": None}]},
        ],
    ]


def _columns_by_name(batch: pa.RecordBatch, name: str) -> list[pa.Array]:
    return [batch.column(index) for index, field in enumerate(batch.schema) if field.name == name]


@pytest.mark.rust
def test_structural_projection_raw_codes_and_metadata(tmp_path: Path):
    """Deep projection and raw selection preserve nested fixed alignment."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "nested_fixed_projection.bufr",
        [build_nested_fixed_replication_message(compressed=True)],
    )
    outer_name = "fixed_1_05_002__occ_0"
    inner_name = "fixed_1_02_003__occ_0"
    temperature_path = f"{outer_name}.{inner_name}.B012101"
    with earth2bufr.open_bufr(path, strict=True) as source:
        batch = _first_without_provenance(
            source.iter_batches(
                projection=[temperature_path],
                raw_code_paths=[temperature_path],
                fixed_replication="structural",
            )
        )

    outer_field = batch.schema.field(outer_name)
    assert outer_field.type.list_size == 2
    assert outer_field.metadata[b"bufr.descriptor"] == b"1-05-002"
    outer = _column(batch, outer_name)
    assert outer.values.type.names == [inner_name]
    inner = outer.values.field(inner_name)
    assert inner.values.type.names == ["B012101"]
    temperatures = inner.values.field("B012101")
    assert temperatures.type == pa.int32()
    assert temperatures.to_pylist() == [
        28_000,
        28_100,
        None,
        28_200,
        28_300,
        28_400,
        28_050,
        28_150,
        None,
        28_250,
        28_350,
        28_450,
    ]
    metadata = inner.type.value_type.field("B012101").metadata
    assert metadata[b"bufr.structural_path"] == temperature_path.encode()
    assert metadata[b"bufr.units"] == b"K"
    assert metadata[b"representation"] == b"bufr_raw_code"


@pytest.mark.rust
def test_flatten_batch_leaves_structural_fixed_replication_alone(tmp_path: Path):
    """flatten_batch does not flatten a FixedSizeList column from the structural layout."""
    import earth2bufr

    subsets = [[2003, [[1.0], [2.0]]], [2004, [[3.0], [None]]]]
    path = _write_messages(tmp_path, "wind_pair.bufr", [build_template_message(WIND_PAIR, subsets, compressed=False)])
    with earth2bufr.open_bufr(path) as source:
        batch = next(source.iter_batches(fixed_replication="structural"))

    flat = earth2bufr.flatten_batch(batch)
    assert _column(flat, "fixed_1_01_002__occ_0").equals(_column(batch, "fixed_1_01_002__occ_0"))


_LEVELS = pa.list_(pa.struct([("CAT", pa.float64()), ("POB", pa.float64())]))
_WEATHER = pa.list_(pa.struct([("CODE", pa.float64()), ("CLOUD", pa.list_(pa.struct([("POB", pa.float64())])))]))


@pytest.mark.rust
@pytest.mark.parametrize(
    ("levels", "weather"),
    [
        (
            [[{"CAT": 1.0, "POB": 850.0}, {"CAT": 2.0, "POB": 700.0}], []],
            [[{"CODE": 1.0, "CLOUD": [{"POB": 5.0}]}], []],
        ),
        ([[{"CAT": 1.0, "POB": 850.0}], None], [[], None]),
        ([[], []], [[{"CODE": 1.0, "CLOUD": []}], []]),
        ([], []),
    ],
    ids=["two_levels", "one_level", "no_levels", "no_rows"],
)
def test_flatten_batch_schema_depends_only_on_the_input_schema(levels: list, weather: list):
    """Flattened column names and types are the same whatever the lists hold."""
    import earth2bufr

    batch = pa.record_batch(
        {
            "SID": pa.array([f"S{row}" for row in range(len(levels))], pa.string()),
            "LEVELS": pa.array(levels, _LEVELS),
            "WEATHER": pa.array(weather, _WEATHER),
            "_data_category": pa.array([0.0] * len(levels), pa.float64()),
        }
    )

    assert earth2bufr.flatten_batch(batch).schema == pa.schema(
        [
            ("SID", pa.string()),
            ("CAT", pa.float64()),
            ("POB", pa.float64()),
            ("CODE", pa.float64()),
            ("POB_2", pa.float64()),
            ("_data_category", pa.float64()),
        ]
    )


@pytest.mark.rust
def test_structural_fixed_replication_applies_operators(tmp_path: Path):
    """Operator-adjusted packing remains active inside a fixed group."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "fixed_operator.bufr",
        [build_fixed_replication_operator_message()],
    )
    with earth2bufr.open_bufr(path, strict=True) as source:
        flat = _first_without_provenance(source.iter_batches())
        structural = _first_without_provenance(
            source.iter_batches(
                fixed_replication="structural",
            )
        )

    assert [column.to_pylist() for column in _columns_by_name(flat, "B014027")] == [
        [45.0],
        [67.0],
    ]
    fixed = _column(structural, "fixed_1_03_002__occ_0")
    assert fixed.values.field("B014027").to_pylist() == [45.0, 67.0]
    assert fixed.type.value_type.field("B014027").metadata[b"bufr.bit_width"] == b"8"


@pytest.mark.rust
def test_structural_layout_validation_is_strict(tmp_path: Path):
    """Unknown layouts, structural paths and structural list filters fail before yielding data."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "nested_fixed_validation.bufr",
        [build_nested_fixed_replication_message(compressed=False)],
    )
    with earth2bufr.open_bufr(path) as source:
        with pytest.raises(ValueError, match="fixed_replication"):
            list(source.iter_batches(fixed_replication="nested"))
        with pytest.raises(ValueError):
            list(
                source.iter_batches(
                    fixed_replication="structural",
                    projection=["fixed_1_05_002__occ_0.missing"],
                )
            )
        with pytest.raises(ValueError):
            source.iter_batches(
                fixed_replication="structural",
                list_filters=[earth2bufr.ListValueFilter("LIST", "KEY", (1.0,))],
            )


@pytest.mark.rust
def test_open_handle_exposes_cached_headers_and_actual_structure(tmp_path: Path):
    """Planning metadata comes from the same actual-file index as decoding."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)
    with earth2bufr.open_bufr(path) as source:
        info = source.info
        structure = source.source_structure(info.data_messages[0].source_message_index)

    assert len(info.data_messages) == 1
    message = info.data_messages[0]
    assert (message.source_message_index, message.offset, message.size_bytes) == (0, 0, path.stat().st_size)
    assert message.subsets > 0
    assert message.table_epoch == 0
    assert info.table_epochs[0].embedded_table_sha256 is None
    assert structure["observation_descriptors"] == list(message.descriptors)
    assert structure["active_elements"]
    assert structure["effective_occurrences"]
    assert structure["decoded_physical_schema"]
    assert set(structure) == {
        "format_version",
        # The hash of this message's table epoch, matching info.table_epochs.
        "embedded_table_sha256",
        "observation_descriptors",
        "active_sequences",
        "active_elements",
        "effective_occurrences",
        "decoded_physical_schema",
    }


def _with_section2(message: bytes) -> bytes:
    section1_end = 8 + int.from_bytes(message[8:11], "big")
    section2 = (6).to_bytes(3, "big") + b"\x00ab"
    total = len(message) + len(section2)
    flagged = message[:17] + bytes([message[17] | 0x80]) + message[18:section1_end]
    return b"BUFR" + total.to_bytes(3, "big") + flagged[7:] + section2 + message[section1_end:]


def _padded_messages() -> bytes:
    wmo = build_flat_wmo_message(surface_observation_case())
    # Zero padding, a zero-length marker right after an end marker, a marker
    # without a matching end, and a truncated trailing message.
    return (
        b"\x00" * 5
        + wmo
        + b"BUFR\x00\x00\x00"
        + b"\x00" * 7
        + b"BUFR\x00\x00\x09"
        + _zero_subset_message()
        + wmo
        + wmo[:40]
    )


@pytest.mark.rust
@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(build_two_epoch_prepbufr_messages(), id="two_epoch_prepbufr"),
        pytest.param(build_flat_wmo_message(surface_observation_case()), id="wmo"),
        pytest.param(_padded_messages(), id="padding_between_messages"),
        pytest.param(build_flat_wmo_message(edition3_surface_case()), id="edition3"),
        pytest.param(_with_section2(build_flat_wmo_message(surface_observation_case())), id="section2"),
    ],
)
def test_get_metadata_equals_open_file_info(tmp_path: Path, raw: bytes):
    """get_metadata equals the info of the same file opened with open_bufr."""
    import earth2bufr

    path = tmp_path / "messages.bufr"
    path.write_bytes(raw)
    info = earth2bufr.get_metadata(path)
    with earth2bufr.open_bufr(path) as source:
        assert info == source.info
    assert info.data_messages


@pytest.mark.rust
def test_source_structure_selects_one_message_from_mixed_templates(tmp_path: Path):
    """Indexed lookup resolves one template while unindexed lookup stays strict."""
    import earth2bufr

    first = build_flat_wmo_message(surface_observation_case())
    second = build_flat_wmo_message(missing_value_case())
    path = _write_messages(tmp_path, "mixed_templates.bufr", [first, second])

    with earth2bufr.open_bufr(path) as source:
        messages = source.info.data_messages
        first_structure = source.source_structure(0)
        second_structure = source.source_structure(1)
        with pytest.raises(ValueError):
            source.source_structure(2)

    assert first_structure["observation_descriptors"] == list(messages[0].descriptors)
    assert second_structure["observation_descriptors"] == list(messages[1].descriptors)
    assert first_structure["observation_descriptors"] != second_structure["observation_descriptors"]


@pytest.mark.rust
def test_empty_file_iterator_and_invalid_options(tmp_path: Path):
    """Empty data and invalid bounds have explicit iterator behavior."""
    import earth2bufr

    path = _write_messages(tmp_path, "empty.bufr", [_zero_subset_message()])
    with earth2bufr.open_bufr(path) as source:
        assert list(source.iter_batches()) == []
        assert source.info.data_messages == ()
        with pytest.raises(ValueError, match="messages_per_batch"):
            source.iter_batches(messages_per_batch=0)
        with pytest.raises(ValueError, match="prefetch_batches"):
            source.iter_batches(prefetch_batches=0)
        with pytest.raises(ValueError, match="target_decoded_bytes"):
            source.iter_batches(target_decoded_bytes=0)
        with pytest.raises(ValueError, match="projection"):
            list(source.iter_batches(projection=[]))


@pytest.mark.rust
def test_unknown_nested_projection_is_rejected(tmp_path: Path):
    """Projection errors identify a missing nested child."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "nested.bufr",
        [build_delayed_replication_message(delayed_replication_case())],
    )
    with earth2bufr.open_bufr(path) as source, pytest.raises(ValueError):
        list(source.iter_batches(projection=["B031001.NOT_A_FIELD"]))


@pytest.mark.rust
def test_strict_open_rejects_unknown_table_b_descriptor(tmp_path: Path):
    """Strict indexing fails before a zero-width placeholder can decode."""
    import earth2bufr

    message = bufr_message(
        [desc(0, 63, 255)],
        b"",
        num_subsets=1,
    )
    path = _write_messages(tmp_path, "unknown_descriptor.bufr", [message])

    with earth2bufr.open_bufr(path) as source:
        assert next(source.iter_batches()).num_rows == 1
    with pytest.raises(ValueError, match="Unknown Table B descriptor 0-63-255"):
        earth2bufr.open_bufr(path, strict=True)


def _operator_message(operator: tuple[int, int, int]) -> bytes:
    writer = BitWriter()
    writer.write_bits(YEAR.encode_raw(2024), YEAR.bit_width)
    writer.write_bits(AIR_TEMPERATURE.encode_raw(281.5), AIR_TEMPERATURE.bit_width)
    return bufr_message(
        [YEAR.descriptor, desc(*operator), AIR_TEMPERATURE.descriptor], writer.to_bytes(), num_subsets=1
    )


# Operators that read bits from Section 4 or change the widths of later elements, and unassigned 2-10.
BIT_CHANGING_OPERATORS = [
    (2, 3, 8),
    (2, 3, 255),
    (2, 8, 10),
    (2, 9, 32),
    (2, 10, 0),
    (2, 21, 1),
    (2, 23, 255),
    (2, 24, 255),
    (2, 25, 255),
    (2, 32, 255),
    (2, 43, 0),
]
NO_DATA_OPERATORS = [
    (2, 3, 0),
    (2, 5, 0),
    (2, 22, 0),
    (2, 23, 0),
    (2, 24, 0),
    (2, 25, 0),
    (2, 32, 0),
    (2, 35, 0),
    (2, 36, 0),
    (2, 37, 0),
    (2, 37, 255),
    (2, 41, 0),
    (2, 41, 255),
    (2, 42, 0),
    (2, 42, 255),
]
STRICT_NO_DATA_OPERATORS = [(2, 3, 0), (2, 5, 0)]


@pytest.mark.rust
@pytest.mark.parametrize("operator", BIT_CHANGING_OPERATORS, ids=lambda operator: desc_name(*operator))
def test_permissive_decode_rejects_operators_that_change_the_bit_layout(tmp_path: Path, operator):
    """An operator that would misalign the following values fails in both modes."""
    import earth2bufr

    path = _write_messages(tmp_path, "operator.bufr", [_operator_message(operator)])
    match = f"Unsupported operator {desc_name(*operator)}"
    with earth2bufr.open_bufr(path) as source, pytest.raises(ValueError, match=match):
        list(source.iter_batches())
    with pytest.raises(ValueError, match=match):
        earth2bufr.open_bufr(path, strict=True)


@pytest.mark.rust
@pytest.mark.parametrize("operator", NO_DATA_OPERATORS, ids=lambda operator: desc_name(*operator))
def test_operators_without_data_bits_decode_permissively_and_fail_strict(tmp_path: Path, operator):
    """Values after an operator that occupies no bits stay aligned; strict mode rejects it unless it is
    2-03-000 or 2-05-000."""
    import earth2bufr

    path = _write_messages(tmp_path, "operator.bufr", [_operator_message(operator)])
    with earth2bufr.open_bufr(path) as source:
        batch = next(source.iter_batches())
    assert _column(batch, "B004001").to_pylist() == [2024.0]
    assert _column(batch, "B012101").to_pylist() == [281.5]
    if operator in STRICT_NO_DATA_OPERATORS:
        with earth2bufr.open_bufr(path, strict=True) as source:
            assert next(source.iter_batches()).equals(batch)
    else:
        with pytest.raises(ValueError, match=f"Unsupported operator {desc_name(*operator)}"):
            earth2bufr.open_bufr(path, strict=True)


@pytest.mark.rust
def test_ia5_missing_is_null_and_blank_stays_blank(tmp_path: Path):
    """All-ones IA5 decodes as null in strict and permissive mode; a genuine blank stays blank."""
    import earth2bufr

    missing_path = _write_messages(
        tmp_path,
        "missing_ia5.prepbufr",
        [build_synthetic_prepbufr_ia5_message(b"\xff" * 4)],
    )
    blank_path = _write_messages(
        tmp_path,
        "blank_ia5.prepbufr",
        [build_synthetic_prepbufr_ia5_message(b" " * 4)],
    )

    with earth2bufr.open_bufr(missing_path) as source:
        assert next(source.iter_batches()).column("SID").to_pylist() == [None]
    with earth2bufr.open_bufr(missing_path, strict=True) as source:
        assert next(source.iter_batches()).column("SID").to_pylist() == [None]
    with earth2bufr.open_bufr(blank_path, strict=True) as source:
        assert next(source.iter_batches()).column("SID").to_pylist() == [""]


@pytest.mark.rust
def test_strict_ia5_rejects_non_printable_bytes(tmp_path: Path):
    """Strict mode cannot silently delete invalid IA5 bytes."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "invalid_ia5.prepbufr",
        [build_synthetic_prepbufr_ia5_message(b"K\x00EN")],
    )
    with earth2bufr.open_bufr(path, strict=True) as source, pytest.raises(ValueError):
        next(source.iter_batches())


@pytest.mark.rust
def test_strict_decode_rejects_short_section4_payload(tmp_path: Path):
    """Strict mode cannot manufacture missing values beyond Section 4."""
    import earth2bufr

    path = _write_messages(
        tmp_path,
        "short_section4.bufr",
        [bufr_message([YEAR.descriptor], b"", num_subsets=1)],
    )
    with earth2bufr.open_bufr(path) as source:
        assert next(source.iter_batches()).column("B004001").to_pylist() == [None]
    with (
        earth2bufr.open_bufr(path, strict=True) as source,
        pytest.raises(ValueError),
    ):
        next(source.iter_batches())


@pytest.mark.rust
def test_projection_skips_compressed_strings_with_increments(tmp_path: Path):
    """Omitting a compressed string whose subsets differ keeps the following field aligned."""
    import earth2bufr

    temperatures = [271.5, 280.25, 290.0]
    message = build_compressed_station_name_group_message(["ALPHA", "BRAVO", "CHARLIE"], temperatures)
    path = _write_messages(tmp_path, "station_names.bufr", [message])

    with earth2bufr.open_bufr(path) as source:
        full = next(source.iter_batches())
        projected = next(source.iter_batches(projection=["B031001.B012101"]))

    full_rows = _column(full, "B031001").to_pylist()
    assert [row[0]["B001015"].strip() for row in full_rows] == ["ALPHA", "BRAVO", "CHARLIE"]
    assert [row[0]["B012101"] for row in full_rows] == temperatures
    assert [row[0]["B012101"] for row in _column(projected, "B031001").to_pylist()] == temperatures


@pytest.mark.rust
def test_closed_file_rejects_new_reads(tmp_path: Path):
    """Closing releases the handle; an iterator created before closing still finishes."""
    import earth2bufr

    path = surface_observation_case().write(tmp_path)
    with earth2bufr.open_bufr(path) as source:
        batches = source.iter_batches()
    assert sum(batch.num_rows for batch in batches) == 2
    with pytest.raises(ValueError):
        source.iter_batches()


# One template per shape, two subsets each; compressed subsets share delayed counts.
# 63 is the all-ones value of the 6-bit class-31 field; None is missing.
TEMPLATE_SHAPES = {
    "scalars": (
        (YEAR, ASSOCIATED_FIELD_SIGNIFICANCE, WIND_SPEED, AIRCRAFT_FLIGHT_NUMBER),
        [[2024, 63, 3.5, None], [2025, 63, None, None]],
    ),
    "delayed": (
        (YEAR, Delayed((AIR_TEMPERATURE, AIRCRAFT_FLIGHT_NUMBER, ASSOCIATED_FIELD_SIGNIFICANCE))),
        [
            [2024, [[280.0, "AB12", 63], [281.0, None, 1]]],
            [2025, [[282.0, "CD34", 63], [None, None, 2]]],
        ],
    ),
    "nested_delayed": (
        (YEAR, Delayed((NON_COORDINATE_PRESSURE, Delayed((AIR_TEMPERATURE,))))),
        [
            [2024, [[50_000.0, [[280.0]]], [70_000.0, [[281.0], [None]]]]],
            [2025, [[50_020.0, [[280.5]]], [70_020.0, [[281.5], [282.5]]]]],
        ],
    ),
    "fixed": (
        (YEAR, Fixed(3, (AIR_TEMPERATURE, ASSOCIATED_FIELD_SIGNIFICANCE))),
        [
            [2024, [[280.0, 63], [281.0, 1], [None, 63]]],
            [2025, [[282.0, 63], [283.0, 2], [284.0, 63]]],
        ],
    ),
    "delayed_in_fixed": (
        (YEAR, Fixed(2, (Delayed((AIR_TEMPERATURE,)),))),
        [
            [2024, [[[[280.0], [281.0]]], [[[282.0]]]]],
            [2025, [[[[283.0], [284.0]]], [[[285.0]]]]],
        ],
    ),
    "fixed_in_delayed": (
        (YEAR, Delayed((Fixed(2, (WIND_SPEED,)),))),
        [
            [2024, [[[[1.0], [2.0]]], [[[3.0], [4.0]]]]],
            [2025, [[[[1.5], [2.5]]], [[[3.5], [None]]]]],
        ],
    ),
    "missing_strings": (
        (AIRCRAFT_FLIGHT_NUMBER, Delayed((AIRCRAFT_FLIGHT_NUMBER, AIR_TEMPERATURE))),
        [
            ["AB12", [["CD34", 280.0], [None, 281.0]]],
            [None, [[None, 282.0], ["EF56", 283.0]]],
        ],
    ),
    "interleaved": (
        (
            Delayed((AIR_TEMPERATURE,)),
            YEAR,
            Fixed(2, (WIND_SPEED,)),
            AIRCRAFT_FLIGHT_NUMBER,
            Delayed((NON_COORDINATE_PRESSURE, Delayed((AIR_TEMPERATURE,)))),
            WIND_SPEED,
        ),
        [
            [
                [[280.0], [281.0]],
                2024,
                [[1.0], [2.0]],
                "AB12",
                [[50_000.0, [[280.0]]], [70_000.0, [[281.0], [None]]]],
                3.5,
            ],
            [
                [[282.0], [None]],
                2025,
                [[1.5], [None]],
                None,
                [[50_020.0, [[280.5]]], [70_020.0, [[281.5], [282.5]]]],
                None,
            ],
        ],
    ),
    "delayed_after_delayed_in_fixed": (
        (
            Delayed((AIR_TEMPERATURE,)),
            YEAR,
            Fixed(2, (Delayed((WIND_SPEED,)),)),
            Delayed((NON_COORDINATE_PRESSURE, Delayed((AIR_TEMPERATURE,)))),
        ),
        [
            [[[280.0]], 2024, [[[[1.0], [2.0]]], [[[3.0]]]], [[50_000.0, [[280.0], [None]]]]],
            [[[None]], 2025, [[[[1.5], [None]]], [[[3.5]]]], [[50_020.0, [[281.0], [282.0]]]]],
        ],
    ),
}


@pytest.mark.rust
@pytest.mark.parametrize("shape", list(TEMPLATE_SHAPES))
@pytest.mark.parametrize("fixed_replication", ["flat", "structural"])
def test_compressed_and_uncompressed_encodings_decode_alike(tmp_path: Path, shape: str, fixed_replication: str):
    """The same subsets decode to the same columns whichever Section 4 encoding carries them."""
    import earth2bufr

    items, subsets = TEMPLATE_SHAPES[shape]
    decoded = []
    for compressed in (False, True):
        message = build_template_message(items, subsets, compressed=compressed)
        path = _write_messages(tmp_path, f"{shape}_{compressed}.bufr", [message])
        with earth2bufr.open_bufr(path) as source:
            decoded.append(_first_without_provenance(source.iter_batches(fixed_replication=fixed_replication)))
    uncompressed, compressed = decoded
    assert compressed.schema.equals(uncompressed.schema, check_metadata=True)
    differing = [
        name
        for name, left, right in zip(compressed.schema.names, compressed.columns, uncompressed.columns, strict=True)
        if not left.equals(right)
    ]
    assert not differing, differing


@pytest.mark.rust
def test_structural_raw_codes_after_delayed_groups_in_fixed_occurrences(tmp_path: Path):
    """Raw-code paths under delayed groups resolve to the same leaves compressed and uncompressed."""
    import earth2bufr

    items, subsets = TEMPLATE_SHAPES["delayed_after_delayed_in_fixed"]
    raw_paths = ["fixed_1_03_002__occ_0.B031001.B011002", "B031001_2.B010004"]
    decoded = []
    for compressed in (False, True):
        message = build_template_message(items, subsets, compressed=compressed)
        path = _write_messages(tmp_path, f"raw_{compressed}.bufr", [message])
        with earth2bufr.open_bufr(path) as source:
            batches = source.iter_batches(fixed_replication="structural", raw_code_paths=raw_paths)
            decoded.append(_first_without_provenance(batches))
    uncompressed, compressed = decoded
    assert compressed.equals(uncompressed)
    pressures = _column(compressed, "B031001_2").values.field("B010004")
    assert pressures.type == pa.int32()
    assert pressures.to_pylist() == [5_000, 5_002]


@pytest.mark.rust
@pytest.mark.parametrize("compressed", [False, True])
def test_class31_all_ones_is_a_value_and_missing_strings_are_null(tmp_path: Path, compressed: bool):
    """Class-31 fields are exempt from the all-ones missing rule; a missing string decodes as null."""
    import earth2bufr

    items, subsets = TEMPLATE_SHAPES["scalars"]
    path = _write_messages(tmp_path, "scalars.bufr", [build_template_message(items, subsets, compressed=compressed)])
    with earth2bufr.open_bufr(path) as source:
        batch = _first_without_provenance(source.iter_batches())
    assert _column(batch, "B031021").to_pylist() == [63, 63]
    assert _column(batch, "B001006").to_pylist() == [None, None]
    assert _column(batch, "B011002").to_pylist() == [3.5, None]


LEVELS = (YEAR, Delayed((NON_COORDINATE_PRESSURE, AIR_TEMPERATURE)))
WIND_PAIR = (YEAR, Fixed(2, (WIND_SPEED,)))


def _levels_subsets(message: int, subsets: int) -> list:
    levels = 1 + message % 3
    return [
        [
            2000 + message,
            [
                [50_000.0 + 1_000 * level + 10 * subset, None if (subset + level) % 4 == 3 else 250.0 + message + level]
                for level in range(levels)
            ],
        ]
        for subset in range(subsets)
    ]


def _mixed_file(tmp_path: Path) -> Path:
    """Two templates, both encodings, odd subset counts and a zero-subset message, in one file."""
    messages = [
        build_template_message(LEVELS, _levels_subsets(0, 3), compressed=True),
        build_template_message(LEVELS, _levels_subsets(1, 1), compressed=False),
        build_template_message(LEVELS, _levels_subsets(2, 5), compressed=True),
        _zero_subset_message(),
        build_template_message(WIND_PAIR, [[2003, [[1.0], [2.0]]], [2003, [[1.5], [None]]]], compressed=False),
        build_template_message(WIND_PAIR, [[2004, [[3.0], [4.0]]]], compressed=True),
        build_template_message(LEVELS, _levels_subsets(5, 7), compressed=True),
        build_template_message(LEVELS, _levels_subsets(6, 2), compressed=False),
    ]
    return _write_messages(tmp_path, "mixed.bufr", messages)


def _runs(batches) -> list[pa.Table]:
    """Consecutive batches with one schema, joined."""
    runs: list[list[pa.RecordBatch]] = []
    for batch in batches:
        if runs and runs[-1][0].schema.equals(batch.schema, check_metadata=True):
            runs[-1].append(batch)
        else:
            runs.append([batch])
    return [pa.Table.from_batches(run) for run in runs]


@pytest.mark.rust
@pytest.mark.parametrize("fixed_replication", ["flat", "structural"])
@pytest.mark.parametrize(
    "options",
    [
        {"messages_per_batch": 1},
        {"messages_per_batch": 2, "prefetch_batches": 4},
        {"target_decoded_bytes": 1},
        {"target_decoded_bytes": 10**9},
        {"target_decoded_bytes": 1, "prefetch_batches": 4},
        {"target_decoded_bytes": 10**9, "prefetch_batches": 4},
    ],
    ids=["one_message", "prefetch", "tiny_budget", "large_budget", "tiny_budget_prefetch", "large_budget_prefetch"],
)
def test_batching_options_do_not_change_decoded_rows(tmp_path: Path, fixed_replication: str, options: dict):
    """Batch size, decode-ahead and byte budget change only where batches split."""
    import earth2bufr

    path = _mixed_file(tmp_path)
    with earth2bufr.open_bufr(path) as source:
        expected = _runs(source.iter_batches(fixed_replication=fixed_replication))
        actual = _runs(source.iter_batches(fixed_replication=fixed_replication, **options))
    assert [table.num_rows for table in actual] == [table.num_rows for table in expected]
    assert all(left.equals(right) for left, right in zip(actual, expected, strict=True))


@pytest.mark.rust
def test_read_bufr_equals_the_streamed_batches(tmp_path: Path):
    """read_bufr returns each template run of the stream, without provenance columns."""
    import earth2bufr

    path = _mixed_file(tmp_path)
    with earth2bufr.open_bufr(path) as source:
        streamed = [table.drop_columns(list(PROVENANCE_COLUMNS)) for table in _runs(source.iter_batches())]
    one_shot = [pa.Table.from_batches([batch]) for batch in earth2bufr.read_bufr(path)]
    assert len(one_shot) == len(streamed)
    assert all(left.equals(right) for left, right in zip(one_shot, streamed, strict=True))


@pytest.mark.rust
@pytest.mark.parametrize(
    "options",
    [
        {"projection": ["B004001", "B031001.B012101"]},
        {
            "projection": ["B004001", "B031001"],
            "list_filters": [("B031001", "B010004", (51_000.0, 50_010.0))],
        },
        {"raw_code_paths": ["B004001"]},
        {"fixed_replication": "structural"},
    ],
    ids=["projection", "list_filter", "raw_codes", "structural"],
)
def test_read_bufr_options_equal_the_streamed_batches(tmp_path: Path, options: dict):
    """read_bufr applies decode options as iter_batches does."""
    import earth2bufr

    filters = options.get("list_filters")
    if filters is not None:
        options = {**options, "list_filters": [earth2bufr.ListValueFilter(*item) for item in filters]}

    messages = [
        build_template_message(LEVELS, _levels_subsets(0, 3), compressed=True),
        build_template_message(LEVELS, _levels_subsets(1, 2), compressed=False),
        build_template_message(LEVELS, _levels_subsets(2, 4), compressed=True),
    ]
    path = _write_messages(tmp_path, "levels.bufr", messages)
    with earth2bufr.open_bufr(path) as source:
        streamed = [table.drop_columns(list(PROVENANCE_COLUMNS)) for table in _runs(source.iter_batches(**options))]
    one_shot = [pa.Table.from_batches([batch]) for batch in earth2bufr.read_bufr(path, **options)]
    assert len(one_shot) == len(streamed) == 1
    assert one_shot[0].equals(streamed[0])


@pytest.mark.rust
def test_projection_equals_selecting_from_a_full_decode(tmp_path: Path):
    """Projected decoding returns the values a full decode holds, in both encodings."""
    import earth2bufr

    messages = [
        build_template_message(LEVELS, _levels_subsets(0, 3), compressed=True),
        build_template_message(LEVELS, _levels_subsets(1, 2), compressed=False),
    ]
    path = _write_messages(tmp_path, "levels.bufr", messages)
    with earth2bufr.open_bufr(path) as source:
        full = pa.Table.from_batches(source.iter_batches())
        projected = pa.Table.from_batches(source.iter_batches(projection=["B004001", "B031001.B012101"]))
    assert projected.column("B004001").equals(full.column("B004001"))
    expected = [[{"B012101": level["B012101"]} for level in row] for row in full.column("B031001").to_pylist()]
    assert projected.column("B031001").to_pylist() == expected


@pytest.mark.rust
def test_projection_quarantines_a_message_truncated_after_the_projected_fields(tmp_path: Path):
    """Quarantine reads the items after the last projected field; permissive decoding keeps the message."""
    import earth2bufr

    valid = build_template_message(LEVELS, _levels_subsets(0, 3), compressed=True)
    path = _write_messages(tmp_path, "truncated_tail.bufr", [valid, _truncate_section4(valid, dropped_bytes=4), valid])
    with earth2bufr.open_bufr(path) as source:
        iterator = source.iter_batches(projection=["B004001"], skip_truncated_messages=True)
        quarantined = list(iterator)
        permissive = list(source.iter_batches(projection=["B004001"]))

    assert [value for batch in quarantined for value in _column(batch, "_source_message_index").to_pylist()] == [
        0,
        0,
        0,
        2,
        2,
        2,
    ]
    assert [failure.source_message_index for failure in iterator.decode_failures] == [1]
    assert [value for batch in permissive for value in _column(batch, "B004001").to_pylist()] == [2000.0] * 9


@pytest.mark.rust
@pytest.mark.parametrize("shape", list(TEMPLATE_SHAPES))
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("fixed_replication", ["flat", "structural"])
def test_projected_fields_equal_the_full_decode(tmp_path: Path, shape: str, compressed: bool, fixed_replication: str):
    """Each field projected alone, and alternate fields together, decode as in the full decode."""
    import earth2bufr

    items, subsets = TEMPLATE_SHAPES[shape]
    path = _write_messages(tmp_path, "shape.bufr", [build_template_message(items, subsets, compressed=compressed)])
    with earth2bufr.open_bufr(path) as source:
        full = _first_without_provenance(source.iter_batches(fixed_replication=fixed_replication))
        names = [name for name in full.schema.names if full.schema.names.count(name) == 1 and name != "_data_category"]
        for projection in [[name] for name in names] + [names[::2], names[1::2]] * (len(names) > 1):
            batches = source.iter_batches(projection=projection, fixed_replication=fixed_replication)
            projected = _first_without_provenance(batches)
            assert projected.select(projection).equals(full.select(projection)), projection


@pytest.mark.rust
def test_list_filter_during_decode_equals_filter_after_decode(tmp_path: Path):
    """A list filter keeps the same levels in compressed and uncompressed messages."""
    import earth2bufr

    subsets = _levels_subsets(4, 3)
    keep = [earth2bufr.ListValueFilter("B031001", "B010004", (50_000.0, 51_020.0))]
    decoded = []
    for compressed in (True, False):
        path = _write_messages(
            tmp_path, f"levels_{compressed}.bufr", [build_template_message(LEVELS, subsets, compressed=compressed)]
        )
        with earth2bufr.open_bufr(path) as source:
            decoded.append(_first_without_provenance(source.iter_batches(list_filters=keep)))
    compressed, uncompressed = decoded
    assert _column(compressed, "B031001").equals(_column(uncompressed, "B031001"))
    assert _column(compressed, "B031001").to_pylist() == [
        [{"B010004": 50_000.0, "B012101": 254.0}],
        [],
        [{"B010004": 51_020.0, "B012101": None}],
    ]
