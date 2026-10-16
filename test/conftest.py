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

"""Pytest policy hooks for CI validation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter

TRUTHY = {"1", "true", "yes", "on"}


def _fail_on_skip_enabled(config: Config) -> bool:
    return os.environ.get("EARTH2BUFR_FAIL_ON_SKIP", "").lower() in TRUTHY and not config.getoption("collectonly")


def pytest_configure(config: Config) -> None:
    """Initialize storage for skips that should be fatal in CI."""
    config._earth2bufr_skip_reports = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect skipped test reports for the end-of-session CI policy check."""
    outcome = yield
    report = outcome.get_result()
    if _fail_on_skip_enabled(item.config) and report.skipped and not hasattr(report, "wasxfail"):
        item.config._earth2bufr_skip_reports.append(report)


def pytest_sessionfinish(session, exitstatus: int) -> None:
    """Fail CI when any selected test skipped instead of running."""
    if _fail_on_skip_enabled(session.config) and session.config._earth2bufr_skip_reports:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus: int, config: Config) -> None:
    """Print concrete skip locations when CI forbids skips."""
    reports: list[TestReport] = getattr(config, "_earth2bufr_skip_reports", [])
    if not _fail_on_skip_enabled(config) or not reports:
        return

    terminalreporter.write_sep("=", "earth2bufr unexpected skips")
    for report in reports:
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        terminalreporter.write_line(f"{report.nodeid}: {reason}")
