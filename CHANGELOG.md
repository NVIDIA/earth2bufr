# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - Unreleased

### Added

- List filters may use a raw-code field as their key, so a channel subset can keep its radiances and channel numbers
  as raw codes. Filter values are compared with the key's physical value. (#2)
- `projection_missing="skip"` leaves projected fields that a decoded template does not have out of that template's
  batches (with their raw-code paths and list filters), so files that mix templates can decode only the fields they
  need. The default `"raise"` keeps the current behavior. (#2)
- Decoding of a DX Table A message type that leads Section 3 on its own, as in NCEP GOES radiance dumps (`gsrcsr`,
  `gsrasr`). The sequence is expanded once, as NCEPLIBS-bufr does. (#4)
- Decoding of the 2-23/24/25/32-255 bitmap marker operators. The bitmap is read from the first subset and checked
  against every subset and message of the batch. (#4)
- Strict-mode decoding of the WMO quality operators 2-22/23/24/25/32-000, 2-35-000, 2-36-000, 2-37-000 and
  2-37-255. Bitmaps may follow a "values follow" operator directly or after 2-36-000, and may be a delayed
  replication of 0-31-031. (#4)

### Changed

- Files are opened faster on network file systems. The open-time scan now reads 8 MiB windows as 1 MiB concurrent
  pieces on the decode pool instead of one synchronous read per message. (#2)
- `FileInfo.messages` and `FileInfo.data_messages` are built from an internal Arrow table of message headers, which
  is built without the GIL. `FileInfo(messages=..., table_epochs=...)` keeps working and accepts integer boolean
  flags. (#2, #3)
- `open_bufr` and `get_metadata` scan and index files without holding the GIL, so other Python threads run while a
  file is opened. (#2)
- Dropping a `BufrFile` handle frees its message index and per-epoch tables without the GIL. (#3)
- A marker operator inside a fixed replication expands per repetition. A bitmap refers to the elements before the
  first "values follow" operator until 2-35-000, as ecCodes and PyBufrKit do. (#4)

### Fixed

- A Table D sequence that contains itself, directly or under replication, raises an error naming the descriptor
  instead of overflowing a decode thread's stack and killing the process. (#2)
- A child process forked after a multi-thread decode rebuilds the decode pool instead of hanging on its first
  decode. An iterator whose prefetch thread lives in the parent is refused. (#2)
- NCEP GOES radiance dumps no longer read their Table A sequence twice. (#4)

## [0.1.0] - 2026-10-16

Initial release.

[0.1.1]: https://github.com/NVIDIA/earth2bufr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NVIDIA/earth2bufr/releases/tag/v0.1.0
