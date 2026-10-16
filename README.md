# earth2bufr

Rust-based BUFR (Binary Universal Form for the Representation of meteorological data)
file parser with Python bindings focused entirely on decoding BUFR messages into
in-memory PyArrow tables for ETL and analysis.

Documentation: <https://nvidia.github.io/earth2bufr/>

## Installation

earth2bufr needs Python 3.11+ and a Rust toolchain to build from source.

```bash
git clone https://github.com/NVIDIA/earth2bufr.git
cd earth2bufr

# Install with uv (builds the Rust extension)
uv pip install -e .

# Or build the extension in a development environment
uv sync --group dev
uv run maturin develop --release
```

Prebuilt wheels are built by the `build-wheels` workflow and ship with
Sigstore signatures. See the
[installation guide](https://nvidia.github.io/earth2bufr/installation.html)
for how to verify them.

## Quick start

```python
import earth2bufr

# Read a BUFR file into a list of PyArrow RecordBatches
batches = earth2bufr.read_bufr("observations.bufr")

# Keep only some fields
batches = earth2bufr.read_bufr("observations.bufr", projection=["B005001", "B006001"])

# Stream bounded batches
with earth2bufr.open_bufr("observations.bufr") as source:
    for batch in source.iter_batches():
        ...

# Read message headers and table epochs without decoding
info = earth2bufr.get_metadata("observations.bufr")
```

## Development

```bash
# Install hooks and run all checks
pre-commit install
pre-commit run --all-files

# Run tests
uv sync --group dev --group test
uv run pytest

# Build the documentation
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache-2.0
