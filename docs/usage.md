# Usage Guide

## Reading BUFR Files

The primary interface for reading standard WMO BUFR files:

```python
import earth2bufr

# Basic usage - returns a list of PyArrow RecordBatches: per table epoch, data
# category, then each run of one descriptor template
batches = earth2bufr.read_bufr("observations.bufr")

# Return as pandas DataFrames instead
dfs = earth2bufr.read_bufr("observations.bufr", as_pandas=True)

# Keep some fields; unknown fields raise
batches = earth2bufr.read_bufr(
    "observations.bufr",
    projection=["B012101", "B010004", "B011002"]  # air temperature, pressure, wind speed
)

# Filter by WMO data category
batches = earth2bufr.read_bufr(
    "observations.bufr",
    data_category_filter=[0, 1]  # surface land, surface sea
)
```

`read_bufr` takes the decode options of `iter_batches` -- `projection`,
`list_filters`, `raw_code_paths`, `data_category_filter`, `fixed_replication`,
`skip_truncated_messages` and `projection_missing` -- with the same meaning and
validation, and the
`strict` option of `open_bufr`. Its batches carry `_data_category` and no
provenance columns. The `mnemonics` argument is removed; `projection` replaces
it and raises on fields a decoded template does not have.

## Reading PrepBUFR Files

NOAA/NCEP PrepBUFR files embed their own descriptor tables (DX-tables).
`read_bufr` reads them from the file:

```python
batches = earth2bufr.read_bufr("gdas.t00z.prepbufr")
```

## Bounded Arrow Batches

`read_bufr` holds every decoded row of a file in memory at once.
`open_bufr(...).iter_batches(...)` holds a bounded number of batches, so its
memory does not grow with the file:

```python
import earth2bufr

projection = [
    "SAID",
    "FOVN",
    "YEAR",
    "MNTH",
    "DAYS",
    "HOUR",
    "MINU",
    "SECO",
    "DRF16BIT.CHNM",
    "DRF16BIT.TMANT",
    "DRF16BIT.TMBR",
]

with earth2bufr.open_bufr("atms.bufr") as source:
    for batch in source.iter_batches(
        messages_per_batch=512,
        projection=projection,
    ):
        consume_arrow_batch(batch)
```

`list_filters` keeps only the entries of a repeated group whose key matches,
for example a channel subset:

```python
from earth2bufr import ListValueFilter

channels = (24, 26, 28, 32, 37)
with earth2bufr.open_bufr("crisf4.bufr") as source:
    for batch in source.iter_batches(
        projection=["SAID", "FORN", "FOVN", "DRF16BIT.CHNM", "DRF16BIT.SRAD"],
        list_filters=[
            ListValueFilter(
                field="DRF16BIT",
                key="CHNM",
                values=channels,
            )
        ],
    ):
        consume_compact_spectra(batch)
```

Matching entries keep source order, duplicates and list boundaries.

A batch holds at most `messages_per_batch` data messages, in source order, and
ends early where the data category, descriptor template, compression or table
epoch changes. Decoding runs on a background thread, ahead of the consumer
and in source order; `prefetch_batches` is how many batches it decodes ahead
of the one last returned. A decode error is raised by the `next()` call that
would have returned that batch. Its message starts with `Source BUFR message N:`,
the source message index of the failing message, or `Source BUFR message group
A..B:`, the first and last index of the messages decoded with it. `read_bufr`
decode errors start the same way.
Dropping the iterator waits for its current decode and stops the background
thread. An iterator may be advanced from any thread, one call at a time.

Decoding runs on a thread pool (see [Threads](#threads)). With a single thread
there is no background thread: each batch is decoded during the `next()` call
that returns it.

`_source_message_index` (`uint64`) is the message's position in the file,
counting DX table and empty messages, and `_source_subset_index` (`uint32`) is
the subset's position within the message.

Projection in flat mode accepts `FIELD` and `FIELD.CHILD`; structural mode
accepts deeper paths.

A projected field that a decoded template does not have raises. In files that
mix templates, such as NCEP satellite-wind dumps, `projection_missing="skip"`
leaves it out of that template's batches instead, along with the
`raw_code_paths` and `list_filters` under it; each template's batches keep the
projected fields it has. A projected list whose template lacks a list filter's
key counts as missing and is left out the same way:

```python
with earth2bufr.open_bufr("gdas.t00z.satwnd.tm00.bufr_d") as source:
    for batch in source.iter_batches(
        projection=["SAID", "CLAT", "CLATH", "CLON", "CLONH", "PRLC", "WDIR", "WSPD"],
        projection_missing="skip",
    ):
        ...
```

A template with none of the projected fields gives batches with only
`_data_category`, `_source_message_index` and `_source_subset_index`.

## Threads

```python
earth2bufr.set_num_threads(16)
earth2bufr.get_num_threads()   # 16
```

Decoding runs on its own thread pool. Without a `set_num_threads` call, its size
is taken from `EARTH2BUFR_NUM_THREADS`, then `RAYON_NUM_THREADS`, then the number
of CPUs available to the process. A new size applies from the next batch; with
one thread, decoding runs on the calling thread.

On multi-socket machines, a thread count of about one socket's cores is usually
fastest: on a 2 x 56-core node, 56 threads decoded about twice as fast as all
224 logical CPUs and used less memory.

## Channel Structure: `flat` vs `structural`

Sounder channel radiances are BUFR *fixed replication*: the same group of
descriptors repeats once per channel. `iter_batches` decodes that group two
ways, selected by `fixed_replication`.

`"flat"` (the default) expands each occurrence
into its own top-level column. Because every occurrence carries the same
mnemonics, the result has repeated column names:

```python
with earth2bufr.open_bufr("gdas.t00z.1bamua.tm00.bufr_d") as source:
    batch = next(iter(source.iter_batches()))

batch.schema.names.count("TMBR")   # 15 -- one per AMSU-A channel, all named "TMBR"
```

Looking a repeated name up raises in pyarrow (`batch.column("TMBR")`) and returns
every match in pandas (`df["TMBR"]`), so channels are read by position.

`"structural"` resolves the replication instead, producing one
`fixed_size_list<struct>` per footprint:

```python
with earth2bufr.open_bufr("gdas.t00z.1bamua.tm00.bufr_d") as source:
    batch = next(iter(source.iter_batches(fixed_replication="structural")))

column = batch.column(batch.schema.get_field_index("fixed_1_01_015__occ_0"))
footprint = column[0].as_py()          # 15 entries
[entry["CHNM"] for entry in footprint] # [1.0, 2.0, ..., 15.0]
[entry["TMBR"] for entry in footprint] # the matching brightness temperatures
```

The values are identical between the two modes; only the shape differs.

Which to use:

| Use | When |
|---|---|
| `"flat"` | products with no fixed replication, or when indexing by position |
| `"structural"` | sounders and anything where a channel/level axis must stay attached to its row |

Delayed replication (`DRF16BIT` and similar) already yields a list in both
modes and is unaffected by this setting. ATMS, CrIS FSR and IASI carry their
channel axis that way, so for those products `"structural"` reshapes only the
fixed replication elsewhere in the template.

## Selecting Data Categories

`data_category_filter` takes BUFR Section 1 data category numbers. The
categories a file contains, with their subset counts:

```python
from collections import Counter

info = earth2bufr.get_metadata(path)
subsets = Counter()
for message in info.data_messages:
    subsets[message.data_category] += message.subsets
```

Categories seen in NCEP dumps include `21` (satellite radiances -- AMSU-A, MHS,
ATMS, CrIS, IASI), `5` (satellite winds), `11` (embedded DX tables, not data),
and the conventional categories `0`-`4`. To decode only radiances from a mixed
dump:

```python
with earth2bufr.open_bufr(path) as source:
    for batch in source.iter_batches(data_category_filter=[21]):
        ...
```

## Bounding by Decoded Bytes

`messages_per_batch` bounds a batch by message count. `target_decoded_bytes`
also bounds it by decoded Arrow bytes: the first message of each template
calibrates bytes per subset, each later batch of that template is sized from
the batch decoded before it, a batch also stops once its encoded message bytes
reach the target, and `messages_per_batch` stays the hard cap. Batches
are then decoded one at a time, and the batches decoded ahead, counting the one
being decoded, hold about `prefetch_batches * target_decoded_bytes`:

```python
with earth2bufr.open_bufr("gdas.t00z.crisf4.tm00.bufr_d") as source:
    for batch in source.iter_batches(
        target_decoded_bytes=256 << 20,   # aim for ~256 MiB per batch
        messages_per_batch=2048,          # never exceed this many messages
        fixed_replication="structural",
    ):
        ...
```

## Strict Decoding and Truncated Messages

`strict=True` rejects a file at open time -- before any batch is decoded --
when it contains an unknown Table B descriptor or an unsupported active
operator, and fails iteration on non-printable IA5 or a short Section 4
payload:

```python
with earth2bufr.open_bufr(path, strict=True) as source:
    ...
```

Operators the decoder does not apply fail in both modes when they read Section 4
bits or change the widths of later elements: `2-03-YYY` reference value
definitions (YYY > 0) and `2-03-255`, `2-08`, `2-09`, `2-21`, and any other
operator outside the lists below. `2-03-000` (end of reference value changes)
and `2-05-000` (zero characters) occupy no bits and decode in both modes.

A Table D sequence that contains itself, directly or inside a replicated group,
is a malformed table: it raises `ValueError` in both modes, naming the sequence
or replication descriptor (at open time under `strict=True`, otherwise at the
first message that uses it). Expansion stops after 100,000 inlined sequences or
64 nested replications.

The quality operators decode in both modes: `2-22-000`, `2-23-000`, `2-24-000`,
`2-25-000` and `2-32-000` ("values follow"), `2-35-000`, `2-36-000`, `2-37-000`
and `2-37-255` occupy no bits, and the data-present bitmaps (`0-31-031`) and
class 33 values they announce decode as ordinary columns. A bitmap follows a
"values follow" operator directly, or follows `2-36-000`, which defines it for
reuse; `2-37-000` reuses it and `2-37-255` ends the reuse. Strict decoding
rejects a `2-37-000` with no `2-36-000` bitmap before it. Permissive decoding
skips the event operators `2-41-000`, `2-41-255`, `2-42-000` and `2-42-255`,
which occupy no bits; strict decoding rejects them.

A bitmap refers to the elements before the first "values follow" or
`2-36-000` operator, counted back over the expanded template; later bitmaps
refer to the same elements until `2-35-000` cancels the backward reference
(WMO-No. 306 Table C, as ecCodes and PyBufrKit apply it). The `2-23-255`,
`2-24-255`, `2-25-255` and `2-32-255` markers each carry one value of the next
element their bitmap marks present, packed like that element (`2-25-255`
carries a difference one bit wider, from a reference value of -2^width). They
decode as columns named after the element and the marker, such as
`B012063_224255`; inside a fixed replication each repetition gets its own
columns in both layouts. A marker needs one bitmap for the whole batch, read
from the first subset: decoding fails if later subsets or messages mark other
elements present, if no bitmap is defined before the marker, if the bitmap lies
inside a replication or refers back over a delayed replication, if the marker
lies inside a delayed replication, or if an associated field (`2-04-YYY`) is
defined before it.

When Section 3 starts with a sequence that the embedded DX tables name as a
Table A message type, that sequence alone describes the data, as NCEPLIBS-bufr
reads it; the descriptors after it restate the sequence and are not decoded
again. NCEP GOES radiance dumps (`gsrcsr`, `gsrasr`) carry such a Section 3.

`skip_truncated_messages` skips messages whose template extends past their
Section 4 payload and decodes the rest; the iterator's `decode_failures` lists
what was skipped. Without it, a truncated message raises under `strict=True` and
decodes its missing tail as nulls otherwise. `read_bufr(..., skip_truncated_messages=True)`
omits the skipped messages without reporting them.

```python
with earth2bufr.open_bufr(path) as source:
    batches = source.iter_batches(skip_truncated_messages=True)
    for batch in batches:
        ...
    for failure in batches.decode_failures:
        message = source.info.messages[failure.source_message_index]
        log.warning("skipped message %s at byte %s: %s", failure.source_message_index, message.offset, failure.reason)
```

## Huge Pages for Large Decodes

Large decodes spend much of their time in page faults on new output buffers. With
glibc 2.35 or newer and transparent huge pages set to `madvise` or `always`, the
glibc tunable below backs large allocations with huge pages:

```bash
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 python decode.py
```

On a 2-socket Xeon node this made multi-threaded CrIS and IASI decodes 1.4-1.8x
faster at the same peak memory. Single-threaded decodes change little.

## Tracing Rows to Source Messages

Every `iter_batches` batch has `_source_message_index` and `_source_subset_index`
columns, so every row traces back to one subset of one message. `info.messages`
gives that message's byte offset and size, and `source_structure()` describes
the template, tables and effective packing that decoded it:

```python
with earth2bufr.open_bufr(path) as source:
    messages = source.info.messages
    for batch in source.iter_batches():
        index = batch.column("_source_message_index")[0].as_py()
        message = messages[index]
        print(message.offset, message.size_bytes, message.descriptors[:2])
        structure = source.source_structure(index)
```

## Raw Codes

Numeric fields decode as `float64` after applying their scale and reference.
`raw_code_paths` keeps the stored integers instead, with BUFR missing values as
Arrow nulls. Pair them with the field's effective packing from
`source_structure()` to store values compactly and losslessly:

```python
with earth2bufr.open_bufr(path) as source:
    for batch in source.iter_batches(raw_code_paths=["DRF16BIT.SRAD"]):
        ...
```

## Output Schema

Columns follow the descriptor template in order:

- An element is a column named by its Table B mnemonic. The bundled WMO tables
  name descriptor F-XX-YYY `BFXXYYY` (`B012101` for 0-12-101); embedded DX
  tables and `local_tables_json` give their own mnemonics. Numbers are
  `float64`, raw codes `int32` and character data `string`.
- A delayed replication is a `List<Struct>` column named by the mnemonic of its
  replication factor (`B031001`, `DRF16BIT`). A later delayed replication with
  the same factor at the same nesting level is named with a `_2`, `_3`, ...
  suffix.
- In `"flat"` mode each fixed-replication occurrence repeats its columns under
  the same names. In `"structural"` mode a fixed replication is a
  `FixedSizeList<Struct>` column named `fixed_F_XX_YYY__occ_N`, where F-XX-YYY
  is the replication descriptor and N counts, from 0, the earlier fixed
  replications with the same descriptor at the same nesting level.
- `_data_category` (`float64`) is the message's data category; `iter_batches`
  batches also carry `_source_message_index` and `_source_subset_index`.

A raw-code field carries this field metadata:

| Key | Value |
|---|---|
| `representation` | `bufr_raw_code` |
| `descriptor` | the element descriptor, `F-XX-YYY` |
| `scale`, `reference`, `bit_width` | the packing that decoded the element, after operators |
| `missing_code` | the all-ones code, decoded as null |
| `source_units` | the Table B units |
| `physical_formula` | the physical value as a function of `code`, such as `(code) * 1e-2` |

In `"structural"` mode every element field also carries `bufr.descriptor`,
`bufr.mnemonic`, `bufr.scale`, `bufr.reference`, `bufr.bit_width`,
`bufr.units`, `bufr.value_kind` (`numeric`, `code_or_flag` or `string`) and
`bufr.structural_path`, the dotted path that `projection` and `raw_code_paths`
accept. A replication field carries `bufr.structural_path` and
`bufr.descriptor`: the fixed replication descriptor, or the factor descriptor
of a delayed replication.

## Embedded Table Epochs

A file concatenated from several dumps can carry more than one embedded DX
table block, and a later block may redefine a descriptor the earlier one
defined -- including at a different bit width. `info.table_epochs` reports the
table set each run of messages decodes with, without decoding subset data, and
each data message's `table_epoch` indexes it:

```python
info = earth2bufr.get_metadata(path)
for message in info.data_messages:
    epoch = info.table_epochs[message.table_epoch]
    print(message.source_message_index, epoch.embedded_table_sha256)
for epoch in info.table_epochs:
    for redefinition in epoch.redefines:
        print(redefinition.descriptor, redefinition.before, redefinition.after)
```

Each epoch's tables apply to the data that follows it; they are not merged
file-wide.

## Inspecting Metadata

`get_metadata` reads Sections 0-3 of every message, and DX table messages
whole, without decoding data sections, and returns a `FileInfo` equal to
`open_bufr(path).info`. It raises `ValueError` when the file holds no complete
BUFR message or a header cannot be read.

```python
info = earth2bufr.get_metadata("observations.bufr")

info.messages           # MessageInfo for every message, including DX table and empty messages
info.data_messages      # messages that carry subsets
info.table_epochs       # TableEpoch per table set
for message in info.data_messages:
    print(message.source_message_index, message.offset, message.size_bytes)
    print(message.data_category, message.local_sub_category, message.center)
    print(message.subsets, message.compressed, message.descriptors[:3])
    print(message.year, message.month, message.day, message.hour, message.minute)
```

## Flattening Nested Output

`read_bufr` and `iter_batches` return replication as nested columns
(`List<Struct>` for delayed replication). `flatten=True`, or `flatten_batch` on a
flat-layout batch, turns them into scalar columns:

- The NCEP PrepBUFR level list (the group carrying `CAT`) becomes one row per
  level, with the report's scalars repeated; reports without levels are dropped.
- Every other list keeps only its first entry, promoted to top-level columns in
  its place; rows where the list is empty or null get nulls. A list empty in
  every row is promoted the same way, as null columns.
- A promoted field whose name an earlier column already has gets a `_2`, `_3`,
  ... suffix.

The flattened column names and types depend only on the input schema, so batches
of one descriptor template flatten to the same schema whatever their values.

```python
import earth2bufr

levels = earth2bufr.read_bufr("gdas.prepbufr", data_category_filter=[102], flatten=True)

batches = earth2bufr.read_bufr("gdas.prepbufr")
flat = [earth2bufr.flatten_batch(batch) for batch in batches]
```

To keep every entry of other lists, such as satellite channels, expand them in
Arrow instead:

```python
import pyarrow as pa
import pyarrow.compute as pc

lists = batch.column("DRF16BIT")
entries = lists.flatten()
long = pa.Table.from_arrays(
    [pc.list_parent_indices(lists), *entries.flatten()],
    names=["footprint", *[field.name for field in entries.type]],
)
```

## Custom Tables

Use custom BUFR descriptor tables for non-standard or local descriptors:

```python
# Load custom tables from JSON files
table_b_json, table_d_json = earth2bufr.load_tables(
    table_b_path="my_table_b.json",
    table_d_path="my_table_d.json"
)

# Read with custom tables
batches = earth2bufr.read_bufr(
    "observations.bufr",
    table_b_json=table_b_json,
    table_d_json=table_d_json
)

# Add local table overrides for a specific centre
local_json = earth2bufr.create_local_tables(
    table_b_overrides={
        "0-48-001": {
            "mnemonic": "myLocalVar",
            "name": "My local variable",
            "units": "K",
            "scale": 2,
            "reference_value": 0,
            "bit_width": 16,
        }
    }
)

batches = earth2bufr.read_bufr(
    "observations.bufr",
    local_tables_json=local_json
)
```
