# earth2bufr

A high-performance BUFR (Binary Universal Form for the Representation
of meteorological data) parser built with Rust and Python bindings via PyO3.

## Features

- Supports BUFR Edition 3 and Edition 4 formats
- Handles both compressed and uncompressed data
- Outputs data as Apache Arrow RecordBatches or Pandas DataFrames
- Parallel processing of multiple BUFR messages
- Extensible table support (Table B, Table D, local tables)
- PrepBUFR support (NOAA/NCEP DX-table format)

## Quick Start

```python
import earth2bufr

# Read a standard BUFR file (returns list of PyArrow RecordBatches)
batches = earth2bufr.read_bufr("observations.bufr")

# Read a PrepBUFR file
batches = earth2bufr.read_bufr("gdas.prepbufr")

# Keep only some fields
batches = earth2bufr.read_bufr("observations.bufr", projection=["B005001", "B006001"])

# Stream bounded batches, keeping only some fields
with earth2bufr.open_bufr("observations.bufr") as source:
    for batch in source.iter_batches(projection=["B005001", "B006001"]):
        ...

# Read message headers and table epochs without decoding
info = earth2bufr.get_metadata("observations.bufr")
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
validation
references
api
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
