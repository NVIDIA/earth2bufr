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

"""Opening, inspecting and decoding files from many Python threads at once."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from synthetic.builders import build_flat_wmo_message, build_two_epoch_prepbufr_messages, surface_observation_case

if TYPE_CHECKING:
    from pathlib import Path


def _read(path: Path):
    import earth2bufr

    try:
        with earth2bufr.open_bufr(path) as source:
            info = source.info
            batches = list(source.iter_batches())
        return info, earth2bufr.get_metadata(path), batches
    except ValueError as error:
        return str(error)


def _same(left, right) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and len(left[2]) == len(right[2])
        and all(a.equals(b) for a, b in zip(left[2], right[2], strict=True))
    )


@pytest.mark.rust
@pytest.mark.timeout(300)
def test_concurrent_opens_equal_serial_opens(tmp_path: Path):
    """16 threads opening the same and different files, one of them truncated, while the thread count changes."""
    import earth2bufr

    wmo = build_flat_wmo_message(surface_observation_case())
    files = {
        "prepbufr": build_two_epoch_prepbufr_messages(),
        # Junk between messages spans several open-time reads.
        "spread": wmo + b"\x00" * (9 << 20) + wmo,
        "truncated": wmo[: len(wmo) // 2],
    }
    paths = []
    for name, raw in files.items():
        path = tmp_path / f"{name}.bufr"
        path.write_bytes(raw)
        paths.append(path)
    expected = {path: _read(path) for path in paths}
    assert isinstance(expected[paths[-1]], str)

    threads_before = earth2bufr.get_num_threads()
    stop = threading.Event()

    def change_thread_count() -> None:
        counts = (1, 2, 4)
        index = 0
        while not stop.is_set():
            earth2bufr.set_num_threads(counts[index % len(counts)])
            index += 1

    changer = threading.Thread(target=change_thread_count)
    changer.start()
    try:
        with ThreadPoolExecutor(16) as pool:
            for _ in range(10):
                jobs = [paths[index % len(paths)] for index in range(48)]
                for path, result in zip(jobs, pool.map(_read, jobs), strict=True):
                    assert _same(result, expected[path]), path.name
    finally:
        stop.set()
        changer.join()
        earth2bufr.set_num_threads(threads_before)
