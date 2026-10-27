# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

### Changed

- Use at most 32 decode threads by default; `EARTH2BUFR_NUM_THREADS`,
  `RAYON_NUM_THREADS` and `set_num_threads` still set any count
- Split uncompressed batches into parallel chunks of at least 128 KiB of
  messages instead of 1 MiB, so PrepBUFR and GPS-RO batches use more threads
- `iter_batches` with `target_decoded_bytes` and no `messages_per_batch` no
  longer caps batches at 512 messages
- Fill the `_source_message_index`, `_source_subset_index` and `_data_category`
  columns in parallel

### Deprecated

### Removed

### Fixed

### Security

## [0.1.1] - 2026-10-26

### Added

- Raw-code list filters, `projection_missing="skip"`, Arrow message headers, and faster file open
- Decode NNJA GOES radiance dumps and WMO quality operators (MTG FCI)

### Changed

- Free a closed file's index and build its header table without the GIL

## [0.1.0] - 2026-10-16

Initial release.

[0.2.0]: https://github.com/NVIDIA/earth2bufr/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/NVIDIA/earth2bufr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NVIDIA/earth2bufr/releases/tag/v0.1.0
