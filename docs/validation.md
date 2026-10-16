# Validation Suite

Earth2bufr validation is intentionally small, real, and readable. The tests
reward consistency with independent decoders, clear fixture structure, and high
impact coverage. They avoid mocks for BUFR decoding: each synthetic case writes
real BUFR bytes, decodes those bytes, and compares normalized tabular output.

## Design Rules

- Prefer one targeted fixture over many broad fixtures.
- Keep each synthetic BUFR message small enough to inspect by eye.
- Compare decoded DataFrames or nested rows, because Earth2bufr's public contract
  is tabular output.
- Use ecCodes as the gold reference for standard non-PrepBUFR WMO BUFR.
- Use PyBufrKit as an independent structural reference when it can decode the
  fixture.
- Use SatPy only for product-specific BUFR reader validation; generic WMO BUFR
  files are documented as outside SatPy's reader model.

## Synthetic Cross-Decoder Tests

`test/synthetic/test_wmo_cross_validation.py` builds standard WMO BUFR messages
from small descriptor lists and Section 4 bit streams. It compares Earth2bufr,
ecCodes, and PyBufrKit after normalizing each result to the same DataFrame
columns. Assertion failures include the fixture descriptor list and expected rows.

| Case | Structure | References | Purpose |
| --- | --- | --- | --- |
| `wmo_surface_scalars` | Edition 4, uncompressed, two subsets, time/location/weather scalars | ecCodes, PyBufrKit | Baseline WMO scalar decoding and row ordering |
| `wmo_edition3_surface_scalars` | Edition 3, uncompressed, one subset | ecCodes, PyBufrKit | Header and data decoding for the older BUFR edition |
| `wmo_missing_value` | All-bits-set missing numeric value | ecCodes, PyBufrKit | Consistent null handling |
| `wmo_compressed_numeric` | Compressed numeric fields with non-zero increments | ecCodes, PyBufrKit | Compressed reference/increment decoding |
| `wmo_code_flag_operator_exemption` | Active `2-01`/`2-02` spanning code, flag, and numeric fields | ecCodes | Code/flag packing remains unchanged while numeric albedo uses the modified packing |
| `wmo_code_flag_combined_operator_exemption` | Active `2-07` spanning code, flag, and numeric fields | ecCodes | Combined numeric packing changes do not affect code/flag descriptors |
| `wmo_delayed_replication_levels` | Delayed replication of pressure/temperature levels | ecCodes, PyBufrKit | Nested list structure consistency |
| `wmo_quality_bitmap_operators` | `2-22-000` and `2-36-000` bitmap with confidence values, `2-37-000`/`2-37-255` bitmap reuse, then a wind speed | ecCodes | Operators without data bits leave later values aligned |

The code/flag operator fixtures use ecCodes only; PyBufrKit 0.2.25 compares code/flag units case-sensitively against its bundled tables.

The delayed-replication flattening test is marked `xfail` because it documents
the desired generic DataFrame shape: one flattened row per replicated level. It
should become a normal passing test when generic non-PrepBUFR flattening expands
all replicated levels.

## Synthetic PrepBUFR DX Tests

`test/synthetic/test_prepbufr_dx.py` writes PrepBUFR files with embedded DX
tables (`dataCategory=11`) and checks that `get_metadata` and `read_bufr` see
the local descriptors and sequences, that each DX block's table epoch governs
only the data messages after it, and that legacy IA5 missing placeholders decode
as null. Generic ecCodes/PyBufrKit decoding does not consume NCEP embedded DX
tables, so these compare against values built into the test.

## Earth2bufr-Specific Synthetic Tests

`test/test_synthetic_bufr.py` covers structures that are hard to compare with
generic references without custom external table setup:

| Structure | What is validated |
| --- | --- |
| Local Table B and Table D overrides | User-supplied descriptors and sequences decode to mnemonic columns |
| Strings and code-table-like numerics | CCITT IA5 and small categorical numeric fields decode correctly |
| Missing values | All-bits-set non-class-31 values become null |
| Nested delayed replication | List-of-struct output preserves the readable BUFR tree |
| Compressed constant fields | Compressed references with zero increments repeat across subsets |
| Operator `2-05-YYY` | Inline character data from an operator descriptor becomes a text column |

`test/test_bounded_batches.py` covers `open_bufr` and `iter_batches`:

| Property | What is validated |
| --- | --- |
| Encoding equivalence | Nine template shapes decode alike compressed and uncompressed, flat and structural |
| Table C operators | Operators that change the bit layout fail in both modes; operators without data bits leave later values aligned |
| Batching invariance | Batch size, decode-ahead and byte budget change only where batches split |
| One-shot vs streamed | `read_bufr` equals the streamed batches without provenance columns |
| Projection and list filters | Projected and filtered decodes equal selecting from a full decode |
| Framing and failures | NCEP BYTCNT subsets, truncated-message skipping and strict-mode rejections |

These tests use generated BUFR bytes and local descriptor tables; they do not
mock parser output.

## Running Validation

Use `uv` from the repository root after building the Rust extension:

```bash
uv run maturin develop --release
uv run pytest -q test/synthetic test/test_synthetic_bufr.py
uv run pytest -q -m 'not slow'
```

The test dependency group installs ecCodes, PyBufrKit, and SatPy. No sibling
repository checkout is required.
