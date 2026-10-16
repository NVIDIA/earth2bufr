# Earth2BUFR Contribution Guide

## Introduction

Welcome to Earth2BUFR! We welcome contributions to the Rust-based BUFR decoder,
its Python bindings, documentation, examples, and validation tools. Earth2BUFR
decodes meteorological BUFR and PrepBUFR messages into in-memory Apache Arrow
tables for ETL and analysis.

Please read this guide before submitting a contribution.

### Communicate with Us

Use the repository's issue tracker to report bugs, discuss feature requests,
or propose larger changes before
starting work. Check existing issues first to avoid duplicating ongoing work.

For decoding bugs, include the Earth2BUFR version or commit, Python and Rust
versions, a minimal reproducer, and the expected and actual results. When
possible, include a small BUFR fixture you have permission to share, along with
its descriptor sequence and any required local tables.

For security vulnerabilities, follow [SECURITY.md](SECURITY.md).

## Codebase Structure

- [src/rust](src/rust): Rust decoder and native Python extension.
- [src/earth2bufr](src/earth2bufr): Python API and bundled BUFR tables.
- [test](test): Python tests, synthetic fixtures, cross-decoder validation,
  and license checks.
- [docs](docs): Sphinx documentation, including usage and validation guidance.
- [examples](examples): Example decoding workflows.

## Development Setup

Install Python 3.11 or newer, a current stable Rust toolchain with `rustfmt` and
`clippy` (see [rustup](https://rustup.rs/)), and [uv](https://docs.astral.sh/uv/).
Clone the repository or your fork and create a feature branch.

Install the dependency groups, build the extension, and enable pre-commit hooks:

```bash
uv sync --group dev --group test
uv run maturin develop --release
uv run pre-commit install
```

Rebuild with `uv run maturin develop --release` after changing Rust code.

## Coding Style and Checks

Follow the surrounding code's conventions. Python code uses Ruff for formatting
and linting, NumPy-style docstrings, and `ty` for type checking. Rust code uses
`rustfmt` and Clippy. Markdown is checked with `markdownlint`.

Run the configured hooks before submitting changes:

```bash
uv run pre-commit run --all-files
```

The local Cargo dependency-check hook also requires `cargo-deny`, which can be
installed with `cargo install cargo-deny`.

For changes to Rust code, run the same formatting and lint commands as CI:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

### Testing

Add targeted regression tests for bug fixes and meaningful coverage for new
functionality. Decoder tests should use real BUFR bytes, including small
synthetic messages, and compare decoded tabular output. Prefer ecCodes for
standard WMO BUFR reference checks and PyBufrKit for independent structural
validation where supported. See [docs/validation.md](docs/validation.md) for
the test design rules and PrepBUFR-specific checks.

Run the fast test suite from the repository root:

```bash
uv run pytest --timeout=300 -rs -m "not slow"
```

Local tests can skip when optional dependencies are missing; check
the skip reasons to ensure the tests relevant to your change actually ran.
CI sets `EARTH2BUFR_FAIL_ON_SKIP=1` to reject unexpected skips.

For performance-sensitive changes, include reproducible timing results and
describe the input data, hardware, and measured operation.

### Documentation

Update documentation and examples when changing the public API or decoding
behavior. To build the documentation locally:

```bash
uv sync --group dev --group docs
uv run maturin develop --release
uv run sphinx-build -b html docs docs/_build/html
```

## Submitting Changes

1. Create a branch in your fork, or in the project if you have write access.
2. Keep changes focused and add appropriate tests and documentation.
3. Run the checks above and sign off your commits as described below.
4. Push your branch and open a pull request against `main`.
5. Describe the motivation, summarize the changes, link related issues, and
   list the checks you ran, including any skipped tests or unavailable fixtures.
6. Address review feedback and ensure the applicable CI jobs pass before merge.

### Continuous Integration

GitHub Actions runs lint checks, self-contained synthetic decoder tests, and
release-mode wheel builds for Linux x86-64/ARM64, Windows x86-64, and macOS ARM64,
plus a source distribution.
The documentation workflow builds the Sphinx docs on every pull request and
publishes them to GitHub Pages from `main`, without running data-dependent benchmarks.

### GitHub Repository Automation

Every day at 01:30 UTC, GitHub Actions closes issues more than 21 days old unless
they have the `confirmed` label, and closes all PRs more than 14 days old (including
drafts). These deadlines use creation time; comments and other activity do not
reset them. The closing workflow can also be run manually from the Actions tab.

## Licensing Information

Contributions are licensed under the [Apache License 2.0](LICENSE). Preserve
existing copyright notices and add the project's license header to new source
files, using the appropriate comment syntax (`#` for Python, `//` for Rust)
and copyright year. For example:

```python
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
```

The Python source header check can be run with:

```bash
uv run python test/_license/header_check.py
```

## Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

