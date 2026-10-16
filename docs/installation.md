# Installation

## From PyPI (once published)

```bash
pip install earth2bufr
```

## From source

earth2bufr requires a Rust toolchain (1.70+) for building the native extension.

### Prerequisites

- Python 3.11+
- Rust toolchain (install via [rustup](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin) build tool

### Development install

```bash
# Clone the repository
git clone https://github.com/NVIDIA/earth2bufr.git
cd earth2bufr

# Install with uv (recommended)
uv sync --dev
uv run maturin develop --release

# Or with pip
pip install maturin
maturin develop --release
```

### Build a wheel

```bash
maturin build --release
```

### Verify a signed release artifact

Release wheels and source distributions built by the `build-wheels`
workflow ship with a Sigstore bundle (`<file>.sigstore.json`). Check one with
the [`sigstore`](https://pypi.org/project/sigstore/) CLI, using the exact
workflow identity and the branch it was built from:

```bash
pip install sigstore
sigstore verify identity \
  --bundle earth2bufr-1.0.0-cp311-abi3-manylinux_2_34_x86_64.whl.sigstore.json \
  --cert-identity "https://github.com/NVIDIA/earth2bufr/.github/workflows/build-wheels.yml@refs/heads/1.0.0-rc" \
  --cert-oidc-issuer https://token.actions.githubusercontent.com \
  earth2bufr-1.0.0-cp311-abi3-manylinux_2_34_x86_64.whl
```

Wheels built from an `X.Y.Z-rc` branch are published as version `X.Y.Z`.

## Dependencies

### Runtime

- `pyarrow >= 14.0` — Arrow columnar format support

### Optional

- `pandas >= 2.0` — for DataFrame output (opt-in via `as_pandas=True`)
