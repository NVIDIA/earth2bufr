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

"""Assertion helpers for human-readable synthetic decoder comparisons."""

from __future__ import annotations

from typing import Any

import pandas as pd


def assert_frame_matches(case, library: str, actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Assert two normalized frames match, including case details on failure."""
    actual = _normalize_frame(actual, expected.columns.tolist())
    expected = _normalize_frame(expected, expected.columns.tolist())
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-6, atol=1e-8)
    except AssertionError as exc:
        raise AssertionError(
            "\n".join(
                [
                    f"{library} mismatch for {case.name}",
                    case.describe(),
                    "Expected frame:",
                    expected.to_string(index=False),
                    "Actual frame:",
                    actual.to_string(index=False),
                ]
            )
        ) from exc


def assert_reference_frames_match(case, frames: dict[str, pd.DataFrame], expected: pd.DataFrame) -> None:
    """Assert every library frame matches the expected case rows."""
    for library, frame in frames.items():
        assert_frame_matches(case, library, frame, expected)


def assert_nested_rows_match(case, library: str, actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    """Assert nested delayed-replication rows match with useful failure text."""
    try:
        _assert_nested_equal(actual, expected)
    except AssertionError as exc:
        raise AssertionError(
            "\n".join(
                [
                    f"{library} nested-row mismatch for {case.name}",
                    case.describe(),
                    f"Expected: {expected!r}",
                    f"Actual:   {actual!r}",
                ]
            )
        ) from exc


def _normalize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.loc[:, columns].reset_index(drop=True).copy()
    for column in columns:
        normalized[column] = normalized[column].map(lambda value: None if pd.isna(value) else value)
    return normalized


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key, expected_value in expected.items():
            _assert_nested_equal(actual[key], expected_value)
        return
