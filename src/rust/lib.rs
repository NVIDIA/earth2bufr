// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Earth2bufr: High-performance BUFR file parser.
//!
//! This module provides Rust-based BUFR (Binary Universal Form for the
//! Representation of meteorological data) parsing with Python bindings
//! via PyO3. It supports both BUFR Edition 3 and Edition 4 formats,
//! compressed and uncompressed data, and outputs data as Apache Arrow
//! RecordBatches for efficient interoperability with Python data tools.

use arrow::array::{
    downcast_primitive_array, make_array, new_null_array, Array, ArrayBuilder, ArrayData, ArrayRef,
    AsArray, BooleanArray, FixedSizeListArray, Float64Array, Float64Builder, GenericByteArray,
    Int32Array, Int32Builder, ListArray, ListBuilder, RecordBatch, RecordBatchOptions, StringArray,
    StringBuilder, StructArray, StructBuilder, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::{BooleanBuffer, Buffer, MutableBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{
    BinaryType, ByteArrayType, DataType, Field, FieldRef, Fields, Float64Type, Int32Type, Schema,
    UInt16Type, UInt64Type, UInt8Type, Utf8Type,
};
use arrow::util::bit_mask::set_bits;
use arrow_select::concat::concat;
use arrow_select::filter::filter_record_batch;
use arrow_select::take::take;
use memchr::memmem;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;
use pyo3_arrow::PyRecordBatch;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::JoinHandle;

// =============================================================================
// Table Structures
// =============================================================================

/// Table B entry defining a BUFR element descriptor.
#[derive(Debug, Clone, Deserialize)]
struct TableBEntry {
    /// Mnemonic name for the descriptor
    mnemonic: Arc<str>,
    /// Human-readable name
    name: String,
    /// Units of measurement
    units: Arc<str>,
    /// Scale factor (10^(-scale) applied to decoded value)
    scale: i32,
    /// Reference value (added to raw bits before scaling)
    reference_value: i64,
    /// Number of bits used to encode the value
    bit_width: u32,
    /// Precomputed: true if units indicate a string field (CCITT IA5 / CHARACTER)
    #[serde(skip, default)]
    is_string: bool,
    /// Precomputed: true if units indicate a WMO code or flag table.
    #[serde(skip, default)]
    is_code_or_flag: bool,
}

fn is_string_units(units: &str) -> bool {
    units.contains("CCITT") || units.contains("CHARACTER")
}

fn is_code_or_flag_units(units: &str) -> bool {
    let units = units.to_ascii_uppercase();
    units.contains("CODE TABLE") || units.contains("FLAG TABLE")
}

fn value_kind(entry: &TableBEntry) -> &'static str {
    if entry.is_string {
        "string"
    } else if entry.is_code_or_flag {
        "code_or_flag"
    } else {
        "numeric"
    }
}

/// Table D entry defining a sequence descriptor (macro).
#[derive(Debug, Clone, Deserialize)]
struct TableDEntry {
    /// Mnemonic name for the sequence
    mnemonic: Arc<str>,
    /// List of descriptors this sequence expands to (as F-X-Y integers)
    descriptors: SmallVec<[u32; 16]>,
}

/// Combined table set for BUFR decoding.
#[derive(Debug, Clone)]
struct TableSet {
    /// Table B entries keyed by F-X-Y descriptor integer
    table_b: FxHashMap<u32, Arc<TableBEntry>>,
    /// Table D entries keyed by F-X-Y descriptor integer
    table_d: FxHashMap<u32, Arc<TableDEntry>>,
}

impl TableSet {
    /// Create an empty table set.
    fn new() -> Self {
        Self {
            table_b: FxHashMap::default(),
            table_d: FxHashMap::default(),
        }
    }

    /// Load tables from JSON strings.
    fn from_json(table_b_json: &str, table_d_json: &str) -> Result<Self, String> {
        let table_b_raw: HashMap<String, TableBEntry> =
            serde_json::from_str(table_b_json).map_err(|e| format!("Table B parse error: {e}"))?;

        let table_d_raw: HashMap<String, TableDEntry> =
            serde_json::from_str(table_d_json).map_err(|e| format!("Table D parse error: {e}"))?;

        let table_b = table_b_raw
            .into_iter()
            .filter_map(|(k, mut v)| {
                v.is_string = is_string_units(&v.units);
                v.is_code_or_flag = is_code_or_flag_units(&v.units);
                parse_descriptor_key(&k).map(|d| (d, Arc::new(v)))
            })
            .collect();

        let table_d = table_d_raw
            .into_iter()
            .filter_map(|(k, v)| parse_descriptor_key(&k).map(|d| (d, Arc::new(v))))
            .collect();

        Ok(Self { table_b, table_d })
    }

    /// Merge local tables (center-specific overrides).
    fn merge_local(&mut self, local_json: &str) -> Result<(), String> {
        #[derive(Deserialize)]
        struct LocalTables {
            #[serde(default)]
            table_b: HashMap<String, TableBEntry>,
            #[serde(default)]
            table_d: HashMap<String, TableDEntry>,
        }

        let local: LocalTables = serde_json::from_str(local_json)
            .map_err(|e| format!("Local tables parse error: {e}"))?;

        for (k, mut v) in local.table_b {
            if let Some(d) = parse_descriptor_key(&k) {
                v.is_string = is_string_units(&v.units);
                v.is_code_or_flag = is_code_or_flag_units(&v.units);
                self.table_b.insert(d, Arc::new(v));
            }
        }

        for (k, v) in local.table_d {
            if let Some(d) = parse_descriptor_key(&k) {
                self.table_d.insert(d, Arc::new(v));
            }
        }

        Ok(())
    }

    /// Merge another TableSet into this one (for DX-table extraction).
    fn merge(&mut self, other: TableSet) {
        for (k, v) in other.table_b {
            self.table_b.insert(k, v);
        }
        for (k, v) in other.table_d {
            self.table_d.insert(k, v);
        }
    }
}

/// Parse a descriptor key like "001001" or "0-01-001" into F-X-Y integer.
fn parse_descriptor_key(key: &str) -> Option<u32> {
    if key.contains('-') {
        // Format: F-XX-YYY
        let parts: Vec<&str> = key.split('-').collect();
        if parts.len() == 3 {
            let f: u32 = parts[0].parse().ok()?;
            let x: u32 = parts[1].parse().ok()?;
            let y: u32 = parts[2].parse().ok()?;
            return Some((f << 14) | (x << 8) | y);
        }
    } else if key.len() == 6 {
        // Format: FXXYYY
        let f: u32 = key[0..1].parse().ok()?;
        let x: u32 = key[1..3].parse().ok()?;
        let y: u32 = key[3..6].parse().ok()?;
        return Some((f << 14) | (x << 8) | y);
    }
    None
}

/// Convert F-X-Y integer back to string format.
fn descriptor_to_string(desc: u32) -> String {
    let f = (desc >> 14) & 0x03;
    let x = (desc >> 8) & 0x3F;
    let y = desc & 0xFF;
    format!("{f}-{x:02}-{y:03}")
}

// =============================================================================
// BUFR Message Structures
// =============================================================================

/// A single BUFR message with zero-copy buffer reference.
#[derive(Debug, Clone)]
struct BufrMessage {
    /// Shared reference to the underlying buffer
    buffer: Arc<[u8]>,
    /// Start offset in buffer
    start: usize,
    /// End offset in buffer (exclusive)
    end: usize,
}

impl BufrMessage {
    /// Get the message bytes.
    fn bytes(&self) -> &[u8] {
        &self.buffer[self.start..self.end]
    }
}

/// Parsed BUFR message metadata.
#[derive(Debug)]
struct ParsedMessage<'a> {
    /// BUFR edition number (3 or 4)
    edition: u8,
    /// Master table number
    master_table: u8,
    /// Originating center
    center: u16,
    /// Originating sub-center
    sub_center: u16,
    /// Update sequence number
    update_sequence: u8,
    /// Whether BUFR Section 2 is present.
    optional_section_present: bool,
    /// Data category
    data_category: u8,
    /// International data sub-category
    intl_sub_category: u8,
    /// Local data sub-category
    local_sub_category: u8,
    /// Master table version
    master_table_version: u8,
    /// Local table version
    local_table_version: u8,
    /// Year (4-digit for Ed4, 2-digit for Ed3)
    year: u16,
    /// Month
    month: u8,
    /// Day
    day: u8,
    /// Hour
    hour: u8,
    /// Minute
    minute: u8,
    /// Second (Ed4 only)
    second: u8,
    /// Number of data subsets
    num_subsets: u16,
    /// Whether data is compressed
    is_compressed: bool,
    /// Raw Section 3 flags byte.
    section3_flags: u8,
    /// Raw descriptors from Section 3
    descriptors: Vec<u32>,
    /// Section 4 data bytes — borrowed from the BufrMessage buffer
    data: &'a [u8],
}

/// Expanded item from descriptor expansion.
#[derive(Debug, Clone)]
enum ExpandedItem {
    /// A Table B descriptor
    Descriptor(u32, Arc<TableBEntry>),
    /// A fixed replication group retained for structural Arrow output. Only
    /// the structural expansion emits it; the flat expansion repeats the group.
    FixedReplication {
        descriptor: u32,
        count: usize,
        group: Vec<ExpandedItem>,
    },
    /// Delayed replication marker with factor descriptor and group
    DelayedReplication {
        factor_desc: u32,
        factor_entry: Arc<TableBEntry>,
        group: Vec<ExpandedItem>,
    },
}

/// Operator state during descriptor expansion.
#[derive(Debug, Clone, Default)]
struct OperatorState {
    /// Data width change from operator 201
    data_width_change: i32,
    /// Scale change from operator 202
    scale_change: i32,
    /// Associated field significance bits stack from operator 204.
    /// Each 2-04-YYY pushes Y; 2-04-000 pops last entry.
    /// Before each element descriptor (except class 31), sum(stack) bits
    /// are read as an associated field.
    associated_field_stack: Vec<u32>,
    /// Increase scale/width from operator 207 (Y value, 0 means inactive)
    increase_scale_width_y: u32,
}

// =============================================================================
// BUFR Reading Functions
// =============================================================================

/// Random-access bytes that BUFR messages are located in.
trait MessageSource {
    type Error;
    fn len(&self) -> usize;
    fn read_at(&mut self, offset: usize, buf: &mut [u8]) -> Result<(), Self::Error>;
    /// Position of the first "BUFR" marker at or after `from`.
    fn find_marker(&mut self, from: usize) -> Result<Option<usize>, Self::Error>;
}

/// Minimum bytes read from a file at a time while locating messages, in
/// `SCAN_CHUNK_SIZE` pieces read concurrently on the decode pool. Network file
/// systems serve one small read per message slowly and concurrent large reads
/// quickly.
const SCAN_READ_SIZE: usize = 8 << 20;
const SCAN_CHUNK_SIZE: usize = 1 << 20;

/// Bytes searched for a marker first; each later step doubles, up to
/// `MARKER_SEARCH_MAX`. Gaps between messages are usually a few padding bytes.
const MARKER_SEARCH_START: usize = 256;
const MARKER_SEARCH_MAX: usize = 1 << 20;

/// A file read on demand in windows of at least `SCAN_READ_SIZE` bytes.
///
/// Keeps the two most recent reads: the scanner reads one message's end
/// marker while its header is parsed from the read that found its start.
struct FileSource {
    file: fs::File,
    len: usize,
    /// (file offset, bytes) of the two most recent reads.
    windows: [(usize, Vec<u8>); 2],
    newest: usize,
}

impl FileSource {
    fn open(path: &str) -> std::io::Result<Self> {
        let file = fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        Ok(Self {
            file,
            len,
            windows: [(0, Vec::new()), (0, Vec::new())],
            newest: 0,
        })
    }
}

impl MessageSource for FileSource {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        self.len
    }

    fn read_at(&mut self, offset: usize, buf: &mut [u8]) -> std::io::Result<()> {
        let covers = |(start, bytes): &(usize, Vec<u8>)| {
            offset >= *start && offset + buf.len() <= start + bytes.len()
        };
        let index = match self.windows.iter().position(covers) {
            Some(index) => index,
            None => {
                let index = 1 - self.newest;
                let read_len = buf.len().max(SCAN_READ_SIZE).min(self.len - offset);
                let (start, bytes) = &mut self.windows[index];
                bytes.resize(read_len, 0);
                read_chunks_at(&self.file, bytes, offset)?;
                *start = offset;
                index
            }
        };
        self.newest = index;
        let (start, bytes) = &self.windows[index];
        let begin = offset - start;
        buf.copy_from_slice(&bytes[begin..begin + buf.len()]);
        Ok(())
    }

    fn find_marker(&mut self, from: usize) -> std::io::Result<Option<usize>> {
        let mut chunk = Vec::new();
        let mut chunk_len = MARKER_SEARCH_START;
        let mut start = from;
        while start + 4 <= self.len {
            let end = (start + chunk_len).min(self.len);
            chunk.resize(end - start, 0);
            self.read_at(start, &mut chunk)?;
            if let Some(offset) = find_pattern(&chunk, b"BUFR") {
                return Ok(Some(start + offset));
            }
            if end == self.len {
                break;
            }
            // A marker may straddle the chunk boundary.
            start = end - 3;
            chunk_len = (chunk_len * 2).min(MARKER_SEARCH_MAX);
        }
        Ok(None)
    }
}

/// Finds the byte ranges of BUFR messages one at a time, in file order.
///
/// A message is a "BUFR" marker whose Section 0 length, at least 12 bytes,
/// ends on "7777". A marker without that end is skipped and the search
/// resumes one byte later.
#[derive(Default)]
struct MessageScanner {
    pos: usize,
}

impl MessageScanner {
    fn next_span<S: MessageSource>(
        &mut self,
        source: &mut S,
    ) -> Result<Option<std::ops::Range<usize>>, S::Error> {
        let total = source.len();
        while self.pos + 8 <= total {
            let Some(start) = source.find_marker(self.pos)? else {
                break;
            };
            if start + 7 > total {
                break;
            }
            let mut length = [0u8; 3];
            source.read_at(start + 4, &mut length)?;
            let end = start + u24(&length);

            let mut end_marker = [0u8; 4];
            let complete = end >= start + 12 && end <= total && {
                source.read_at(end - 4, &mut end_marker)?;
                &end_marker == b"7777"
            };
            if complete {
                self.pos = end;
                return Ok(Some(start..end));
            }
            self.pos = start + 1;
        }
        self.pos = total;
        Ok(None)
    }
}

fn u24(bytes: &[u8]) -> usize {
    ((bytes[0] as usize) << 16) | ((bytes[1] as usize) << 8) | (bytes[2] as usize)
}

/// Find a byte pattern in a slice using SIMD-accelerated search.
#[inline]
fn find_pattern(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memmem::find(haystack, needle)
}

/// Why a message header could not be parsed from the bytes at hand.
#[derive(Debug)]
enum HeaderError {
    /// The header continues past the bytes read so far; read this many from
    /// the message start and parse again.
    NeedBytes(usize),
    Invalid(String),
}

impl From<String> for HeaderError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

impl From<&str> for HeaderError {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}

/// The leading bytes of a message of known total length.
#[derive(Clone, Copy)]
struct MessagePrefix<'a> {
    data: &'a [u8],
    message_len: usize,
}

impl MessagePrefix<'_> {
    /// Ask for bytes up to `end` in one read when the prefix is shorter.
    fn require(&self, end: usize) -> Result<(), HeaderError> {
        if end > self.data.len() && self.data.len() < self.message_len {
            return Err(HeaderError::NeedBytes(end.min(self.message_len)));
        }
        Ok(())
    }

    fn byte(&self, index: usize) -> Result<u8, HeaderError> {
        if let Some(value) = self.data.get(index) {
            Ok(*value)
        } else if index < self.message_len {
            Err(HeaderError::NeedBytes(index + 1))
        } else {
            Err(HeaderError::Invalid(format!(
                "Byte {index} lies beyond the {}-byte message",
                self.message_len
            )))
        }
    }

    fn u16(&self, index: usize) -> Result<u16, HeaderError> {
        Ok(((self.byte(index)? as u16) << 8) | (self.byte(index + 1)? as u16))
    }

    fn u24(&self, index: usize) -> Result<usize, HeaderError> {
        Ok(((self.byte(index)? as usize) << 16)
            | ((self.byte(index + 1)? as usize) << 8)
            | (self.byte(index + 2)? as usize))
    }
}

/// Parse a BUFR message into structured metadata.
fn parse_message<'a>(msg: &'a BufrMessage) -> Result<ParsedMessage<'a>, String> {
    let data = msg.bytes();
    match parse_header(data, data.len()) {
        Ok(parsed) => Ok(parsed),
        Err(HeaderError::Invalid(error)) => Err(error),
        Err(HeaderError::NeedBytes(_)) => unreachable!("a whole message needs no further bytes"),
    }
}

/// Parse Sections 0-3 and locate Section 4 from the leading `data` bytes of a
/// `message_len`-byte message.
///
/// Every byte is read through `MessagePrefix`, so a prefix yields the same
/// result as the whole message or asks for more bytes. The result's `data`
/// holds the part of Section 4 present in `data`.
fn parse_header(data: &[u8], message_len: usize) -> Result<ParsedMessage<'_>, HeaderError> {
    if message_len < 8 {
        return Err("Message too short".into());
    }
    let bytes = MessagePrefix { data, message_len };
    bytes.require(11)?;

    // Section 0: Indicator
    if [
        bytes.byte(0)?,
        bytes.byte(1)?,
        bytes.byte(2)?,
        bytes.byte(3)?,
    ] != *b"BUFR"
    {
        return Err("Invalid BUFR marker".into());
    }

    let edition = bytes.byte(7)?;
    let mut parsed = match edition {
        3 => parse_section1_ed3(bytes)?,
        4 => parse_section1_ed4(bytes)?,
        _ => return Err(format!("Unsupported BUFR edition: {edition}").into()),
    };
    let section1_len = bytes.u24(8)?;
    let section4 = parse_sections_2_3(bytes, 8 + section1_len, &mut parsed)?;

    // Section 4: Data
    if section4 + 4 > message_len {
        return Err("Section 4 header beyond message".into());
    }
    let s4_len = bytes.u24(section4)?;
    if s4_len < 4 || section4 + s4_len > message_len {
        return Err("Section 4 extends beyond message".into());
    }
    let present = |offset: usize| offset.min(data.len());
    parsed.data = &data[present(section4 + 4)..present(section4 + s4_len)];
    Ok(parsed)
}

/// Read Edition 3 Section 1, which starts at byte 8.
fn parse_section1_ed3(bytes: MessagePrefix<'_>) -> Result<ParsedMessage<'static>, HeaderError> {
    let s1_len = bytes.u24(8)?;
    bytes.require(8 + s1_len.max(17))?;
    if 8 + s1_len > bytes.message_len {
        return Err("Section 1 extends beyond message".into());
    }

    // Year is 2-digit in Ed3
    let year = bytes.byte(20)? as u16;
    let year = if year > 50 { 1900 + year } else { 2000 + year };
    Ok(ParsedMessage {
        edition: 3,
        master_table: bytes.byte(11)?,
        center: bytes.byte(13)? as u16,
        sub_center: bytes.byte(12)? as u16,
        update_sequence: bytes.byte(14)?,
        optional_section_present: (bytes.byte(15)? & 0x80) != 0,
        data_category: bytes.byte(16)?,
        intl_sub_category: 0,
        local_sub_category: bytes.byte(17)?,
        master_table_version: bytes.byte(18)?,
        local_table_version: bytes.byte(19)?,
        year,
        month: bytes.byte(21)?,
        day: bytes.byte(22)?,
        hour: bytes.byte(23)?,
        minute: bytes.byte(24)?,
        second: 0,
        num_subsets: 0,
        is_compressed: false,
        section3_flags: 0,
        descriptors: Vec::new(),
        data: &[],
    })
}

/// Read Edition 4 Section 1, which starts at byte 8.
fn parse_section1_ed4(bytes: MessagePrefix<'_>) -> Result<ParsedMessage<'static>, HeaderError> {
    if bytes.message_len < 12 {
        return Err("Message too short for Section 1".into());
    }
    let s1_len = bytes.u24(8)?;
    bytes.require(8 + s1_len.max(22))?;
    if 8 + s1_len > bytes.message_len {
        return Err("Section 1 extends beyond message".into());
    }

    Ok(ParsedMessage {
        edition: 4,
        master_table: bytes.byte(11)?,
        center: bytes.u16(12)?,
        sub_center: bytes.u16(14)?,
        update_sequence: bytes.byte(16)?,
        optional_section_present: (bytes.byte(17)? & 0x80) != 0,
        data_category: bytes.byte(18)?,
        intl_sub_category: bytes.byte(19)?,
        local_sub_category: bytes.byte(20)?,
        master_table_version: bytes.byte(21)?,
        local_table_version: bytes.byte(22)?,
        year: bytes.u16(23)?,
        month: bytes.byte(25)?,
        day: bytes.byte(26)?,
        hour: bytes.byte(27)?,
        minute: bytes.byte(28)?,
        second: bytes.byte(29)?,
        num_subsets: 0,
        is_compressed: false,
        section3_flags: 0,
        descriptors: Vec::new(),
        data: &[],
    })
}

/// Skip the optional Section 2 at `pos`, read Section 3, and return where
/// Section 4 starts.
fn parse_sections_2_3(
    bytes: MessagePrefix<'_>,
    mut pos: usize,
    parsed: &mut ParsedMessage<'_>,
) -> Result<usize, HeaderError> {
    // Section 2 (optional)
    if parsed.optional_section_present {
        bytes.require(pos + 3)?;
        if pos + 3 > bytes.message_len {
            return Err("Section 2 header beyond message".into());
        }
        pos += bytes.u24(pos)?;
    }

    // Section 3: Data Description
    bytes.require(pos + 7)?;
    if pos + 7 > bytes.message_len {
        return Err("Section 3 header beyond message".into());
    }
    let s3_len = bytes.u24(pos)?;
    // Section 3 and the Section 4 header.
    bytes.require(pos + s3_len + 4)?;
    parsed.num_subsets = bytes.u16(pos + 4)?;
    parsed.section3_flags = bytes.byte(pos + 6)?;
    parsed.is_compressed = (parsed.section3_flags & 0x40) != 0;

    let desc_end = pos + s3_len;
    let mut i = pos + 7;
    while i + 1 < desc_end {
        parsed.descriptors.push(u32::from(bytes.u16(i)?));
        i += 2;
    }
    Ok(desc_end)
}

// =============================================================================
// PrepBUFR DX-Table Handling
// =============================================================================

/// Data category for DX-table messages in PrepBUFR files.
const DX_TABLE_DATA_CATEGORY: u8 = 11;

/// Extract data_category from raw BUFR message bytes without full parsing.
/// Returns None if message is too short or has unsupported edition.
#[inline]
fn quick_data_category(msg: &BufrMessage) -> Option<u8> {
    let data = msg.bytes();
    if data.len() < 19 || &data[0..4] != b"BUFR" {
        return None;
    }
    match data[7] {
        3 => Some(data[16]),
        4 => Some(data[18]),
        _ => None,
    }
}

/// Extract num_subsets from raw BUFR message bytes without full parsing.
#[inline]
fn quick_num_subsets(msg: &BufrMessage) -> Option<u16> {
    let data = msg.bytes();
    if data.len() < 19 || &data[0..4] != b"BUFR" {
        return None;
    }
    let edition = data[7];
    let s1_len = ((data[8] as usize) << 16) | ((data[9] as usize) << 8) | (data[10] as usize);
    let mut pos = 8 + s1_len;

    // Skip optional Section 2
    let optional_flag_byte = if edition == 4 { 17 } else { 15 };
    if optional_flag_byte < data.len() && (data[optional_flag_byte] & 0x80) != 0 {
        if pos + 3 > data.len() {
            return None;
        }
        let s2_len = ((data[pos] as usize) << 16)
            | ((data[pos + 1] as usize) << 8)
            | (data[pos + 2] as usize);
        pos += s2_len;
    }

    // Section 3: num_subsets at offset 4-5
    if pos + 6 > data.len() {
        return None;
    }
    Some(((data[pos + 4] as u16) << 8) | (data[pos + 5] as u16))
}

/// One embedded-table epoch: a table set plus the data messages it governs.
///
/// A file can concatenate several NCEP dumps, each with its own DX block, and a
/// later block can redefine a descriptor an earlier block declared, including
/// its bit width. Each epoch's tables are the base tables plus that epoch's own
/// block, and they decode only the data messages that follow the block.
#[derive(Debug)]
struct TableEpoch {
    /// Tables in force for this epoch: base plus this epoch's DX block.
    tables: TableSet,
    /// Indices into the caller's message slice, in source order.
    data_message_indices: Vec<usize>,
    /// Indices of the DX messages whose block opened this epoch. Lets callers
    /// recover what the block itself declared, as opposed to what the epoch
    /// inherited from the base tables.
    dx_message_indices: Vec<usize>,
    /// SHA-256 over this epoch's DX message bytes, or None when the epoch
    /// inherits the base tables because no DX block preceded its data.
    embedded_table_sha256: Option<String>,
}

/// Partition a message stream into embedded-table epochs, in source order.
///
/// A run of DX messages closes the preceding epoch and opens a new one. Data
/// messages appearing before any DX block form an epoch governed by
/// `base_tables` alone.
fn extract_dx_table_epochs(
    messages: &[BufrMessage],
    base_tables: &TableSet,
) -> Result<Vec<TableEpoch>, String> {
    let mut epochs: Vec<TableEpoch> = Vec::new();
    let mut pending_dx: Vec<usize> = Vec::new();
    let mut current: Option<TableEpoch> = None;

    for (index, message) in messages.iter().enumerate() {
        match quick_data_category(message) {
            Some(DX_TABLE_DATA_CATEGORY) => {
                // A new block ends whatever epoch was in force.
                if let Some(epoch) = current.take() {
                    if !epoch.data_message_indices.is_empty() {
                        epochs.push(epoch);
                    }
                }
                pending_dx.push(index);
            }
            Some(_) => {
                // Zero-subset messages carry no data and do not open or
                // populate an epoch; the decode paths skip them, so counting
                // them here would misalign the indices.
                if quick_num_subsets(message).is_none_or(|count| count == 0) {
                    continue;
                }
                if current.is_none() {
                    let mut tables = base_tables.clone();
                    let sha = if pending_dx.is_empty() {
                        None
                    } else {
                        tables.merge(extract_dx_tables(messages, &pending_dx)?);
                        let mut digest = Sha256::new();
                        for &dx in &pending_dx {
                            digest.update(messages[dx].bytes());
                        }
                        Some(format!("{:x}", digest.finalize()))
                    };
                    current = Some(TableEpoch {
                        tables,
                        data_message_indices: Vec::new(),
                        dx_message_indices: std::mem::take(&mut pending_dx),
                        embedded_table_sha256: sha,
                    });
                }
                if let Some(epoch) = current.as_mut() {
                    epoch.data_message_indices.push(index);
                }
            }
            None => {}
        }
    }
    if let Some(epoch) = current.take() {
        if !epoch.data_message_indices.is_empty() {
            epochs.push(epoch);
        }
    }
    Ok(epochs)
}

/// Extract local Table B/D definitions from DX-table messages.
///
/// DX-table messages in NCEP PrepBUFR files use a fixed binary format
/// that encodes Table A/B/D entries as positional text/numeric fields.
/// This function directly parses that format rather than using standard
/// BUFR descriptor expansion (which would require the DX-specific
/// descriptors 3-00-003 and 3-00-004 to be pre-defined).
///
/// # DX Message Data Format
///
/// The Section 4 data of each DX message contains:
/// - 1 byte: n_a (Table A entry count, NUMERIC)
/// - n_a * 67 bytes: Table A entries (3-char category + 8-char mnemonic + 56-char description)
/// - 1 byte: n_b (Table B entry count, NUMERIC)
/// - n_b * 112 bytes: Table B entries (fixed format, all CCITT IA5 text)
/// - 1 byte: n_d (Table D entry count, NUMERIC)
/// - Variable: Table D entries (header + member list)
///
/// Returns the Table B and Table D entries of the DX-table messages
/// (data category 11) at `dx_indices` in `messages`. A DX message whose
/// header cannot be parsed is an error naming its index.
fn extract_dx_tables(messages: &[BufrMessage], dx_indices: &[usize]) -> Result<TableSet, String> {
    let mut combined_b: FxHashMap<u32, Arc<TableBEntry>> = FxHashMap::default();
    let mut combined_d: FxHashMap<u32, Arc<TableDEntry>> = FxHashMap::default();

    for &index in dx_indices {
        let parsed = parse_message(&messages[index])
            .map_err(|error| format!("DX table message {index}: {error}"))?;

        let data = &parsed.data;
        if data.is_empty() {
            continue;
        }

        let mut offset: usize = 0;

        // --- Table A: skip (category, mnemonic, description) ---
        if offset >= data.len() {
            continue;
        }
        let n_a = data[offset] as usize;
        offset += 1;
        // Each Table A entry is 67 bytes (3 + 8 + 56 chars)
        offset += n_a * 67;
        if offset >= data.len() {
            continue;
        }

        // --- Table B entries: 112 bytes each ---
        let n_b = data[offset] as usize;
        offset += 1;

        for _ in 0..n_b {
            if offset + 112 > data.len() {
                break;
            }
            let entry_bytes = &data[offset..offset + 112];
            offset += 112;

            // Parse the 112-byte Table B record:
            // [0:1]   F (1 char)
            // [1:3]   XX (2 chars)
            // [3:6]   YYY (3 chars)
            // [6:14]  mnemonic (8 chars)
            // [14:70] name (56 chars)
            // [70:94] units (24 chars)
            // [94:95] sign_scale (1 char: '+' or '-')
            // [95:98] scale_magnitude (3 chars)
            // [98:99] sign_ref (1 char: '+' or '-')
            // [99:109] ref_magnitude (10 chars)
            // [109:112] bit_width (3 chars)

            let f_str = std::str::from_utf8(&entry_bytes[0..1])
                .unwrap_or("0")
                .trim();
            let x_str = std::str::from_utf8(&entry_bytes[1..3])
                .unwrap_or("00")
                .trim();
            let y_str = std::str::from_utf8(&entry_bytes[3..6])
                .unwrap_or("000")
                .trim();
            let mnemonic = std::str::from_utf8(&entry_bytes[6..14])
                .unwrap_or("")
                .trim()
                .to_string();
            let name = std::str::from_utf8(&entry_bytes[14..70])
                .unwrap_or("")
                .trim()
                .to_string();
            let units = std::str::from_utf8(&entry_bytes[70..94])
                .unwrap_or("")
                .trim()
                .to_string();
            let sign_scale = entry_bytes[94];
            let scale_str = std::str::from_utf8(&entry_bytes[95..98])
                .unwrap_or("0")
                .trim();
            let sign_ref = entry_bytes[98];
            let ref_str = std::str::from_utf8(&entry_bytes[99..109])
                .unwrap_or("0")
                .trim();
            let width_str = std::str::from_utf8(&entry_bytes[109..112])
                .unwrap_or("0")
                .trim();

            let f: u32 = f_str.parse().unwrap_or(0);
            let x: u32 = x_str.parse().unwrap_or(0);
            let y: u32 = y_str.parse().unwrap_or(0);
            let desc_id = (f << 14) | (x << 8) | y;
            if desc_id == 0 {
                continue;
            }

            let scale_mag: i32 = scale_str.parse().unwrap_or(0);
            let scale = if sign_scale == b'-' {
                -scale_mag
            } else {
                scale_mag
            };

            let ref_mag: i64 = ref_str.parse().unwrap_or(0);
            let reference_value = if sign_ref == b'-' { -ref_mag } else { ref_mag };

            let bit_width: u32 = width_str.parse().unwrap_or(0);

            let entry = Arc::new(TableBEntry {
                mnemonic: Arc::from(mnemonic.as_str()),
                name,
                units: Arc::from(units.as_str()),
                scale,
                reference_value,
                bit_width,
                is_string: is_string_units(&units),
                is_code_or_flag: is_code_or_flag_units(&units),
            });

            combined_b.insert(desc_id, entry);
        }

        // --- Table D entries: variable length ---
        if offset >= data.len() {
            continue;
        }
        let n_d = data[offset] as usize;
        offset += 1;

        for _ in 0..n_d {
            if offset + 71 > data.len() {
                break;
            }

            // Header: 6 chars FXXYYY + 64 chars (mnemonic + description) = 70 bytes
            let fxxyyy_bytes = &data[offset..offset + 6];
            let mnem_desc_bytes = &data[offset + 6..offset + 70];
            offset += 70;

            let fxxyyy = std::str::from_utf8(fxxyyy_bytes).unwrap_or("000000").trim();
            let mnem_desc = std::str::from_utf8(mnem_desc_bytes).unwrap_or("");
            let seq_mnemonic = mnem_desc[..8.min(mnem_desc.len())].trim().to_string();

            // Parse FXXYYY
            let f: u32 = fxxyyy[0..1].parse().unwrap_or(0);
            let x: u32 = fxxyyy[1..3].parse().unwrap_or(0);
            let y: u32 = fxxyyy[3..6].parse().unwrap_or(0);
            let desc_id = (f << 14) | (x << 8) | y;
            if desc_id == 0 {
                offset += 1; // skip n_members byte
                continue;
            }

            // Member count: 1 byte NUMERIC
            if offset >= data.len() {
                break;
            }
            let n_members = data[offset] as usize;
            offset += 1;

            // Members: each is 6 chars (FXXYYY format)
            let mut descriptors: SmallVec<[u32; 16]> = SmallVec::with_capacity(n_members);
            for _ in 0..n_members {
                if offset + 6 > data.len() {
                    break;
                }
                let member_str = std::str::from_utf8(&data[offset..offset + 6]).unwrap_or("000000");
                offset += 6;

                // Parse member FXXYYY to descriptor ID
                if member_str.len() >= 6 {
                    let mf: u32 = member_str[0..1].parse().unwrap_or(0);
                    let mx: u32 = member_str[1..3].parse().unwrap_or(0);
                    let my: u32 = member_str[3..6].parse().unwrap_or(0);
                    let member_desc = (mf << 14) | (mx << 8) | my;
                    if member_desc != 0 {
                        descriptors.push(member_desc);
                    }
                }
            }

            if !descriptors.is_empty() {
                combined_d.insert(
                    desc_id,
                    Arc::new(TableDEntry {
                        mnemonic: Arc::from(seq_mnemonic.as_str()),
                        descriptors,
                    }),
                );
            }
        }
    }

    if combined_b.is_empty() && combined_d.is_empty() {
        return Err("No table entries extracted from DX messages".to_string());
    }

    Ok(TableSet {
        table_b: combined_b,
        table_d: combined_d,
    })
}

// =============================================================================
// Descriptor Expansion
// =============================================================================

/// Expand descriptors recursively using Table D.
fn expand_descriptors(
    descriptors: &[u32],
    tables: &TableSet,
    strict: bool,
    fixed_replication: FixedReplicationLayout,
) -> Result<Vec<ExpandedItem>, String> {
    expand_descriptors_inner(
        descriptors,
        tables,
        &mut OperatorState::default(),
        strict,
        fixed_replication == FixedReplicationLayout::Structural,
    )
}

fn expand_descriptors_inner(
    descriptors: &[u32],
    tables: &TableSet,
    state: &mut OperatorState,
    strict: bool,
    structural_fixed_replication: bool,
) -> Result<Vec<ExpandedItem>, String> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < descriptors.len() {
        let desc = descriptors[i];
        let f = (desc >> 14) & 0x03;
        let x = (desc >> 8) & 0x3F;
        let y = desc & 0xFF;

        match f {
            0 => {
                // Table B element
                if let Some(entry) = tables.table_b.get(&desc) {
                    // Operator 204: Insert associated field before each element
                    // descriptor (except class 31).
                    if !state.associated_field_stack.is_empty() && x != 31 {
                        let total_bits: u32 = state.associated_field_stack.iter().sum();
                        let assoc_entry = TableBEntry {
                            mnemonic: format!("A{}", descriptor_to_string(desc).replace('-', ""))
                                .into(),
                            name: format!("Associated field for {}", descriptor_to_string(desc)),
                            units: "CODE TABLE".into(),
                            scale: 0,
                            reference_value: 0,
                            bit_width: total_bits,
                            is_string: false,
                            is_code_or_flag: true,
                        };
                        result.push(ExpandedItem::Descriptor(desc, Arc::new(assoc_entry)));
                    }

                    // Apply operator modifications
                    // Fast path: if no operators are active, avoid cloning the entry.
                    let accepts_numeric_operators = !entry.is_string && !entry.is_code_or_flag;
                    let needs_modification = accepts_numeric_operators
                        && (state.data_width_change != 0
                            || state.scale_change != 0
                            || state.increase_scale_width_y > 0);

                    if needs_modification {
                        let mut modified = (**entry).clone();
                        modified.bit_width =
                            (modified.bit_width as i32 + state.data_width_change) as u32;
                        modified.scale += state.scale_change;

                        // Operator 2-07-YYY changes scale, reference, and width
                        // for numeric descriptors. WMO exempts CCITT IA5,
                        // code-table, and flag-table descriptors.
                        if state.increase_scale_width_y > 0 {
                            let y = state.increase_scale_width_y;
                            modified.scale += y as i32;
                            modified.bit_width += (10 * y).div_ceil(3);
                            modified.reference_value *= 10i64.pow(y);
                        }

                        result.push(ExpandedItem::Descriptor(desc, Arc::new(modified)));
                    } else {
                        result.push(ExpandedItem::Descriptor(desc, Arc::clone(entry)));
                    }
                } else {
                    if strict {
                        return Err(format!(
                            "Unknown Table B descriptor {} in strict mode",
                            descriptor_to_string(desc)
                        ));
                    }
                    // Permissive reads keep unknown descriptors as zero-width placeholders.
                    let placeholder = TableBEntry {
                        mnemonic: format!("UNKNOWN_{}", descriptor_to_string(desc)).into(),
                        name: format!("Unknown descriptor {}", descriptor_to_string(desc)),
                        units: "UNKNOWN".into(),
                        scale: 0,
                        reference_value: 0,
                        bit_width: 0,
                        is_string: false,
                        is_code_or_flag: false,
                    };
                    result.push(ExpandedItem::Descriptor(desc, Arc::new(placeholder)));
                }
                i += 1;
            }
            1 => {
                // Replication operator
                let num_following = x as usize;
                let replication_count = y as usize;

                if replication_count == 0 {
                    // Delayed replication - next descriptor specifies count
                    if i + 2 + num_following > descriptors.len() {
                        return Err(format!(
                            "Missing delayed replication factor descriptor at {}",
                            descriptor_to_string(desc)
                        ));
                    }

                    let factor_desc = descriptors[i + 1];
                    let factor_entry =
                        tables.table_b.get(&factor_desc).cloned().ok_or_else(|| {
                            format!(
                                "Unknown factor descriptor {}",
                                descriptor_to_string(factor_desc)
                            )
                        })?;

                    let group_descs = &descriptors[i + 2..i + 2 + num_following];
                    let expanded_group = expand_descriptors_inner(
                        group_descs,
                        tables,
                        state,
                        strict,
                        structural_fixed_replication,
                    )?;

                    result.push(ExpandedItem::DelayedReplication {
                        factor_desc,
                        factor_entry,
                        group: expanded_group,
                    });

                    i += 2 + num_following;
                } else {
                    // Fixed replication
                    if i + 1 + num_following > descriptors.len() {
                        return Err(format!(
                            "Replication at {} exceeds descriptor list",
                            descriptor_to_string(desc)
                        ));
                    }
                    let group = &descriptors[i + 1..i + 1 + num_following];
                    let expanded_group = expand_descriptors_inner(
                        group,
                        tables,
                        state,
                        strict,
                        structural_fixed_replication,
                    )?;
                    if structural_fixed_replication {
                        result.push(ExpandedItem::FixedReplication {
                            descriptor: desc,
                            count: replication_count,
                            group: expanded_group,
                        });
                    } else {
                        for _ in 0..replication_count {
                            result.extend(expanded_group.clone());
                        }
                    }
                    i += 1 + num_following;
                }
            }
            2 => {
                // Operator descriptor
                match x {
                    1 => {
                        // Change numeric data width; WMO exempts strings and
                        // code/flag-table descriptors when entries are applied.
                        if y == 0 {
                            state.data_width_change = 0;
                        } else {
                            state.data_width_change = y as i32 - 128;
                        }
                    }
                    2 => {
                        // Change numeric scale; categorical code/flag-table
                        // data retain their original scale.
                        if y == 0 {
                            state.scale_change = 0;
                        } else {
                            state.scale_change = y as i32 - 128;
                        }
                    }
                    3 if y == 0 => {
                        // Cancels changed reference values, which are never active.
                    }
                    4 => {
                        // Operator 204: Add associated field.
                        // 2-04-YYY (Y>0): push Y bits onto associated field stack.
                        // 2-04-000: pop last entry (cancel innermost nesting).
                        if y == 0 {
                            state.associated_field_stack.pop();
                        } else {
                            state.associated_field_stack.push(y);
                        }
                    }
                    5 if y == 0 => {
                        // Zero characters: no data bits.
                    }
                    5 => {
                        // Signify character
                        let char_entry = TableBEntry {
                            mnemonic: "CHAR_DATA".into(),
                            name: "Character data".into(),
                            units: Arc::from("CCITT IA5"),
                            scale: 0,
                            reference_value: 0,
                            bit_width: y * 8,
                            is_string: true,
                            is_code_or_flag: false,
                        };
                        result.push(ExpandedItem::Descriptor(desc, Arc::new(char_entry)));
                    }
                    6 if i + 1 < descriptors.len() => {
                        // Signify data width for local descriptor
                        let local_desc = descriptors[i + 1];
                        let local_entry = TableBEntry {
                            mnemonic: format!("LOCAL_{}", descriptor_to_string(local_desc)).into(),
                            name: "Local descriptor".into(),
                            units: "NUMERIC".into(),
                            scale: 0,
                            reference_value: 0,
                            bit_width: y,
                            is_string: false,
                            is_code_or_flag: false,
                        };
                        result.push(ExpandedItem::Descriptor(local_desc, Arc::new(local_entry)));
                        i += 1;
                    }
                    7 => {
                        // Increase scale, data width, and change reference value
                        // for numeric data; WMO exempts strings and code/flag
                        // tables when entries are applied.
                        // 2-07-YYY: scale += Y, width += floor((10*Y + 2)/3)
                        // 2-07-000: cancel
                        if y == 0 {
                            state.increase_scale_width_y = 0;
                        } else {
                            state.increase_scale_width_y = y;
                        }
                    }
                    _ if !operator_leaves_bits_unchanged(x, y) => {
                        return Err(format!(
                            "Unsupported operator {}: it changes the Section 4 bits that follow",
                            descriptor_to_string(desc)
                        ));
                    }
                    _ => {
                        if strict {
                            return Err(format!(
                                "Unsupported operator {} in strict mode",
                                descriptor_to_string(desc)
                            ));
                        }
                        // Permissive reads skip operators that occupy no bits and leave
                        // later element widths unchanged.
                    }
                }
                i += 1;
            }
            3 => {
                // Table D sequence - expand inline
                if let Some(entry) = tables.table_d.get(&desc) {
                    // Inline the sequence's descriptors into the current level
                    // by building a combined list of the remaining descriptors
                    let mut combined =
                        Vec::with_capacity(entry.descriptors.len() + descriptors.len() - i - 1);
                    combined.extend_from_slice(&entry.descriptors);
                    combined.extend_from_slice(&descriptors[i + 1..]);

                    // Recursively expand the combined list
                    let expanded = expand_descriptors_inner(
                        &combined,
                        tables,
                        state,
                        strict,
                        structural_fixed_replication,
                    )?;
                    result.extend(expanded);
                    return Ok(result);
                } else {
                    return Err(format!(
                        "Unknown Table D descriptor {}",
                        descriptor_to_string(desc)
                    ));
                }
            }
            _ => {
                return Err(format!("Invalid F value in descriptor: {f}"));
            }
        }
    }

    Ok(result)
}

/// Whether an operator this decoder does not apply occupies no Section 4 bits
/// and leaves the widths of later elements unchanged: the 2-22..2-25 and 2-32
/// "values follow" markers (YYY = 0), the bitmap operators 2-35-000, 2-36-000,
/// 2-37-000 and 2-37-255, and the event operators 2-41 and 2-42 (YYY = 0 or
/// 255). The bitmaps and values these operators announce are ordinary Table B
/// elements and decode as such.
fn operator_leaves_bits_unchanged(x: u32, y: u32) -> bool {
    match x {
        22..=25 | 32 | 35 | 36 => y == 0,
        37 | 41 | 42 => y == 0 || y == 255,
        _ => false,
    }
}

// =============================================================================
// Bit-level Data Decoding
// =============================================================================

/// Fast bit reader using borrowed buffer with bounds-checked reads near end.
struct BitReader<'a> {
    /// Borrowed buffer (no copy, no padding)
    buffer: &'a [u8],
    /// Current bit position
    bit_pos: usize,
    /// Original data length in bits
    data_len_bits: usize,
    /// Reject non-printable IA5 bytes and compressed IA5 increments.
    strict: bool,
    reject_overrun: bool,
}

/// How a `BitReader` treats input that does not fit its template.
#[derive(Clone, Copy)]
struct ReadPolicy {
    /// Reject unknown descriptors and unsupported operators in templates,
    /// non-printable IA5 bytes, compressed IA5 increments and reads past a
    /// BYTCNT subset frame.
    strict: bool,
    /// Fail on reads past the payload instead of returning missing values.
    reject_overrun: bool,
}

/// Whether an all-ones value of this descriptor means missing. Class-31
/// descriptors (replication factors, indicators) do not use the convention.
fn uses_missing_value(descriptor: u32) -> bool {
    ((descriptor >> 8) & 0x3F) != 31
}

/// The all-ones value of a `num_bits`-wide field, the BUFR missing value.
fn all_ones(num_bits: u32) -> u64 {
    if num_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << num_bits) - 1
    }
}

impl<'a> BitReader<'a> {
    fn with_policy(data: &'a [u8], policy: ReadPolicy) -> Self {
        Self {
            buffer: data,
            bit_pos: 0,
            data_len_bits: data.len() * 8,
            strict: policy.strict,
            reject_overrun: policy.reject_overrun,
        }
    }

    /// Whether `num_bits` more bits are within the data.
    fn can_read(&self, num_bits: u32) -> bool {
        self.bit_pos + num_bits as usize <= self.data_len_bits
    }

    /// Read up to 64 bits as u64.
    #[inline]
    fn read_bits(&mut self, num_bits: u32) -> u64 {
        if num_bits == 0 {
            return 0;
        }

        if self.bit_pos + num_bits as usize > self.data_len_bits {
            self.bit_pos += num_bits as usize;
            return all_ones(num_bits);
        }

        let byte_offset = self.bit_pos / 8;
        let bit_offset = self.bit_pos % 8;
        self.bit_pos += num_bits as usize;

        // Fast path: 8-byte read when 8 bytes remain in the buffer
        if byte_offset + 8 <= self.buffer.len() {
            let bytes = &self.buffer[byte_offset..byte_offset + 8];
            let val = u64::from_be_bytes(bytes.try_into().unwrap());
            let shift = 64 - bit_offset as u32 - num_bits;
            let mask = all_ones(num_bits);
            return (val >> shift) & mask;
        }

        // Slow path: near end of buffer, accumulate byte-at-a-time
        if byte_offset >= self.buffer.len() {
            // Past end — return all-ones (missing value)
            return all_ones(num_bits);
        }

        // Read available bytes into a padded 8-byte array
        let mut padded = [0u8; 8];
        let available = self.buffer.len() - byte_offset;
        let to_copy = available.min(8);
        padded[..to_copy].copy_from_slice(&self.buffer[byte_offset..byte_offset + to_copy]);
        let val = u64::from_be_bytes(padded);
        let shift = 64 - bit_offset as u32 - num_bits;
        let mask = all_ones(num_bits);
        (val >> shift) & mask
    }

    /// Check if value represents missing data (all bits set).
    #[inline]
    fn is_missing(val: u64, num_bits: u32) -> bool {
        val == all_ones(num_bits)
    }

    /// Advance the bit position without reading data.
    #[inline]
    fn skip_bits(&mut self, num_bits: u32) {
        self.bit_pos += num_bits as usize;
    }

    /// Read an IA5 string.
    ///
    /// The contract, in order:
    ///
    /// 1. Consume the declared width, always.
    /// 2. The standard all-ones BUFR missing code is null -- in permissive mode
    ///    too, not only strict.
    /// 3. The NCEPLIBS `10E10` missing placeholder in a short character field
    ///    is null (see `NCEP_10E10_MISSING_STRING`).
    /// 4. Bytes that are not printable IA5 are replaced by spaces, not deleted,
    ///    so the value keeps its width and character positions.
    fn read_string_value(&mut self, num_bytes: u32) -> Result<Option<String>, String> {
        let num_bits = num_bytes
            .checked_mul(8)
            .ok_or("IA5 character field width overflow")?;
        if !self.can_read(num_bits) {
            self.skip_bits(num_bits);
            return Ok(None);
        }
        let mut buf = Vec::with_capacity(num_bytes as usize);
        for _ in 0..num_bytes {
            buf.push(self.read_bits(8) as u8);
        }
        if buf.iter().all(|byte| *byte == 0xFF) {
            return Ok(None);
        }
        if is_ncep_10e10_missing_string(&buf) {
            return Ok(None);
        }
        if self.strict {
            if let Some(byte) = buf.iter().find(|&&byte| !(0x20..=0x7E).contains(&byte)) {
                return Err(format!("Invalid IA5 byte 0x{byte:02X} in strict mode"));
            }
        } else {
            // Preserve position and width; do not silently shorten the value.
            for byte in buf.iter_mut() {
                if !(0x20..0x7F).contains(byte) {
                    *byte = b' ';
                }
            }
        }
        // Trim trailing padding in-place
        while buf.last() == Some(&b' ') {
            buf.pop();
        }
        // SAFETY: buf only contains bytes in 0x20..0x7F (valid ASCII subset of UTF-8)
        Ok(Some(unsafe { String::from_utf8_unchecked(buf) }))
    }

    fn validate_bounds(&self) -> Result<(), String> {
        if self.reject_overrun && self.bit_pos > self.data_len_bits {
            return Err(format!(
                "BUFR decode read {} bits beyond the {}{SECTION4_OVERRUN_SUFFIX}",
                self.bit_pos - self.data_len_bits,
                self.data_len_bits
            ));
        }
        Ok(())
    }
}

/// Big-endian IEEE-754 bytes of `10E10`, the NCEPLIBS missing value.
///
/// Files written by NCEPLIBS before 10.2 hold this pattern in short character
/// fields. A 4-8 byte field holding a prefix of it is missing, as in NCEPLIBS
/// `missing.F90`; its printable prefix is `B7Hv`.
const NCEP_10E10_MISSING_STRING: [u8; 8] = [0x42, 0x37, 0x48, 0x76, 0xE0, 0x00, 0x00, 0x00];

fn is_ncep_10e10_missing_string(bytes: &[u8]) -> bool {
    if !(4..=8).contains(&bytes.len())
        || bytes[..bytes.len().min(5)] != NCEP_10E10_MISSING_STRING[..bytes.len().min(5)]
    {
        return false;
    }

    // NCEPLIBS converts embedded NULs to blanks before its 10E10 test, so
    // NUL and blank tails both mark missing.
    bytes
        .get(5..)
        .unwrap_or_default()
        .iter()
        .all(|byte| matches!(*byte, 0x00 | b' '))
}

const NCEP_SUBSET_BYTE_COUNT_DESCRIPTOR: u32 = 63 << 8;

fn ncep_subset_byte_count_entry(expanded: &[ExpandedItem]) -> Option<&TableBEntry> {
    match expanded.first() {
        Some(ExpandedItem::Descriptor(NCEP_SUBSET_BYTE_COUNT_DESCRIPTOR, entry))
            if !entry.is_string && entry.scale == 0 && entry.reference_value == 0 =>
        {
            Some(entry)
        }
        _ => None,
    }
}

fn declared_subset_end(
    reader: &mut BitReader<'_>,
    expanded: &[ExpandedItem],
) -> Result<Option<usize>, String> {
    let Some(entry) = ncep_subset_byte_count_entry(expanded) else {
        return Ok(None);
    };
    if entry.bit_width == 0 || entry.bit_width > 64 || reader.bit_pos & 7 != 0 {
        return Err("Invalid NCEP BYTCNT subset framing descriptor".into());
    }

    let subset_start = reader.bit_pos;
    if !reader.can_read(entry.bit_width) {
        return Err(format!(
            "NCEP BYTCNT at bit {subset_start} lies past the end of the {}{SECTION4_OVERRUN_SUFFIX}",
            reader.data_len_bits
        ));
    }
    let byte_count = reader.read_bits(entry.bit_width);
    reader.bit_pos = subset_start;
    if BitReader::is_missing(byte_count, entry.bit_width) || byte_count == 0 {
        return Err("Invalid NCEP BYTCNT subset byte count".into());
    }
    let declared_end = subset_start
        .checked_add(
            usize::try_from(byte_count)
                .map_err(|_| "NCEP BYTCNT does not fit in memory address space")?
                .checked_mul(8)
                .ok_or("NCEP BYTCNT bit length overflow")?,
        )
        .ok_or("NCEP BYTCNT subset boundary overflow")?;
    if declared_end > reader.data_len_bits {
        return Err(format!(
            "NCEP BYTCNT subset boundary {declared_end} exceeds the {}{SECTION4_OVERRUN_SUFFIX}",
            reader.data_len_bits
        ));
    }
    Ok(Some(declared_end))
}

fn decode_with_subset_frame<T>(
    reader: &mut BitReader<'_>,
    expanded: &[ExpandedItem],
    enabled: bool,
    policy: ReadPolicy,
    decode: impl FnOnce(&mut BitReader<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let declared_end = if enabled {
        declared_subset_end(reader, expanded)?
    } else {
        None
    };
    let Some(declared_end) = declared_end else {
        return decode(reader);
    };

    let subset_start = reader.bit_pos;
    // A read past the declared frame means the subset is shorter than BYTCNT
    // claims. NCEPLIBS reads that tail as missing and real NCEP dumps carry it,
    // so only strict decoding rejects it; it is not message truncation.
    let frame_policy = ReadPolicy {
        strict: policy.strict,
        reject_overrun: policy.strict,
    };
    let mut subset_reader = BitReader::with_policy(
        &reader.buffer[subset_start / 8..declared_end / 8],
        frame_policy,
    );
    let result = decode(&mut subset_reader)?;
    subset_reader.validate_bounds()?;
    reader.bit_pos = declared_end;
    Ok(result)
}

// =============================================================================
// Arrow Output — Tree-guided conversion
// =============================================================================

/// Schema element derived from ExpandedItem tree.
/// Tells us what columns to expect and their structure.
#[derive(Debug, Clone)]
enum SchemaItem {
    /// A scalar column (numeric or string)
    Scalar {
        name: Arc<str>,
        is_string: bool,
        raw_code: Option<RawCodeSpec>,
    },
    /// A delayed replication group → becomes List<Struct<...>> in Arrow
    ReplicationGroup {
        name: Arc<str>,
        children: Vec<SchemaItem>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericOutput {
    PhysicalFloat64,
    RawInt32,
}

#[derive(Debug, Clone)]
struct RawCodeSpec {
    descriptor: u32,
    scale: i32,
    reference_value: i64,
    bit_width: u32,
    missing_code: u64,
    source_units: Arc<str>,
}

impl RawCodeSpec {
    /// Raw int32 output for a numeric, non-class-31 descriptor of 1..=31 bits.
    fn for_entry(path: &str, descriptor: u32, entry: &TableBEntry) -> Result<Self, String> {
        if entry.is_string {
            return Err(format!(
                "Raw-code path '{path}' refers to a string descriptor"
            ));
        }
        if !uses_missing_value(descriptor) {
            return Err(format!(
                "Raw-code path '{path}' is a structural class-31 descriptor"
            ));
        }
        if entry.bit_width == 0 || entry.bit_width > 31 {
            return Err(format!(
                "Raw-code path '{path}' has unsupported {}-bit storage; int32 requires 1..=31 bits",
                entry.bit_width
            ));
        }
        Ok(Self {
            descriptor,
            scale: entry.scale,
            reference_value: entry.reference_value,
            bit_width: entry.bit_width,
            missing_code: (1_u64 << entry.bit_width) - 1,
            source_units: Arc::clone(&entry.units),
        })
    }

    fn field_metadata(&self) -> HashMap<String, String> {
        HashMap::from([
            ("representation".to_string(), "bufr_raw_code".to_string()),
            (
                "descriptor".to_string(),
                descriptor_to_string(self.descriptor),
            ),
            ("scale".to_string(), self.scale.to_string()),
            ("reference".to_string(), self.reference_value.to_string()),
            ("bit_width".to_string(), self.bit_width.to_string()),
            ("missing_code".to_string(), self.missing_code.to_string()),
            ("source_units".to_string(), self.source_units.to_string()),
            (
                "physical_formula".to_string(),
                physical_formula(self.reference_value, self.scale),
            ),
        ])
    }
}

fn physical_formula(reference_value: i64, scale: i32) -> String {
    let reference = match reference_value.cmp(&0) {
        std::cmp::Ordering::Less => format!("code - {}", reference_value.unsigned_abs()),
        std::cmp::Ordering::Equal => "code".to_string(),
        std::cmp::Ordering::Greater => format!("code + {reference_value}"),
    };
    let multiplier = match scale.cmp(&0) {
        std::cmp::Ordering::Greater => format!("1e-{scale}"),
        std::cmp::Ordering::Equal => "1".to_string(),
        std::cmp::Ordering::Less => format!("1e{}", scale.unsigned_abs()),
    };
    format!("({reference}) * {multiplier}")
}

/// Derive schema from the expanded descriptor tree.
fn derive_schema(expanded: &[ExpandedItem]) -> Vec<SchemaItem> {
    let mut schema = Vec::new();
    let mut repl_names = HashMap::new();

    for item in expanded {
        match item {
            ExpandedItem::Descriptor(_, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }
                let is_string = entry.is_string;
                schema.push(SchemaItem::Scalar {
                    name: Arc::clone(&entry.mnemonic),
                    is_string,
                    raw_code: None,
                });
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication {
                factor_entry,
                group,
                ..
            } => {
                let name = delayed_group_name(&mut repl_names, &factor_entry.mnemonic);
                let children = derive_schema(group);
                schema.push(SchemaItem::ReplicationGroup { name, children });
            }
        }
    }
    schema
}

fn configure_raw_code_fields(
    expanded: &[ExpandedItem],
    schema: &mut [SchemaItem],
    prefix: Option<&str>,
    requested: &HashSet<String>,
    matched: &mut HashSet<String>,
) -> Result<(), String> {
    let mut schema_index = 0;
    for item in expanded {
        match item {
            ExpandedItem::Descriptor(descriptor, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }
                let SchemaItem::Scalar { name, raw_code, .. } = &mut schema[schema_index] else {
                    return Err("Raw-code schema does not match expanded descriptor".into());
                };
                let path = field_path(prefix, name);
                if requested.contains(&path) {
                    *raw_code = Some(RawCodeSpec::for_entry(&path, *descriptor, entry)?);
                    matched.insert(path);
                }
                schema_index += 1;
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication { group, .. } => {
                let SchemaItem::ReplicationGroup { name, children } = &mut schema[schema_index]
                else {
                    return Err("Raw-code schema does not match expanded replication".into());
                };
                let path = field_path(prefix, name);
                configure_raw_code_fields(group, children, Some(&path), requested, matched)?;
                schema_index += 1;
            }
        }
    }
    Ok(())
}

fn apply_raw_code_paths(
    expanded: &[ExpandedItem],
    schema: &mut [SchemaItem],
    paths: Option<&[String]>,
) -> Result<(), String> {
    let Some(paths) = paths else {
        return Ok(());
    };
    if paths.is_empty() {
        return Err("raw_code_paths must contain at least one field".into());
    }
    parse_projection_tree(paths, FixedReplicationLayout::Flat)?;
    let requested: HashSet<String> = paths.iter().cloned().collect();
    let mut matched = HashSet::new();
    configure_raw_code_fields(expanded, schema, None, &requested, &mut matched)?;
    let missing: Vec<&String> = requested
        .iter()
        .filter(|path| !matched.contains(*path))
        .collect();
    if !missing.is_empty() {
        return Err(format!("Raw-code fields not found: {missing:?}"));
    }
    Ok(())
}

// =============================================================================
// Arrow Output — Streaming decode to Arrow builders
// =============================================================================

/// A tree of Arrow array builders matching the descriptor schema, created
/// once per `sequential_decode_uncompressed` call and fed values directly
/// from the BitReader.
enum ColumnBuilder {
    /// A scalar Float64 column
    Float64 {
        name: Arc<str>,
        builder: Float64Builder,
    },
    /// A scalar raw BUFR integer-code column
    Int32 {
        name: Arc<str>,
        builder: Int32Builder,
        raw_code: RawCodeSpec,
    },
    /// A scalar Utf8 column
    Utf8 {
        name: Arc<str>,
        builder: StringBuilder,
    },
    /// A List<Struct<...>> column for delayed replication
    List {
        name: Arc<str>,
        /// Arrow fields for the struct children (needed for building the StructBuilder)
        struct_fields: Fields,
        /// Schema items of the struct children; the nested-replication decode
        /// path reads each child's output type and nested layout from them.
        children_schema: Vec<SchemaItem>,
        /// Offsets for each row's list (start, end pairs into the struct array)
        offsets: Vec<i32>,
        /// Current total number of struct elements appended
        current_offset: i32,
        /// The struct builder holding all iterations across all rows
        struct_builder: StructBuilder,
    },
}

/// Arrow type of a scalar column: strings are Utf8, raw codes Int32, other numbers Float64.
fn leaf_data_type(is_string: bool, raw_code: bool) -> DataType {
    if is_string {
        DataType::Utf8
    } else if raw_code {
        DataType::Int32
    } else {
        DataType::Float64
    }
}

/// Convert schema items to Arrow Field definitions (full depth, including nested lists).
fn schema_to_fields(schema: &[SchemaItem]) -> Vec<Field> {
    schema
        .iter()
        .map(|item| match item {
            SchemaItem::Scalar {
                name,
                is_string,
                raw_code,
            } => {
                let field = Field::new(
                    &**name,
                    leaf_data_type(*is_string, raw_code.is_some()),
                    true,
                );
                match raw_code {
                    Some(raw_code) => field.with_metadata(raw_code.field_metadata()),
                    None => field,
                }
            }
            SchemaItem::ReplicationGroup { name, children } => {
                let struct_fields = schema_to_fields(children);
                let inner_field =
                    Field::new("item", DataType::Struct(Fields::from(struct_fields)), true);
                Field::new(&**name, DataType::List(Arc::new(inner_field)), true)
            }
        })
        .collect()
}

/// Create a parallel tree of Arrow builders from the schema.
/// The `capacity` parameter pre-sizes scalar builders to avoid reallocations.
fn create_builders(schema: &[SchemaItem], capacity: usize) -> Vec<ColumnBuilder> {
    schema
        .iter()
        .map(|item| match item {
            SchemaItem::Scalar {
                name,
                is_string,
                raw_code,
            } => {
                if *is_string {
                    ColumnBuilder::Utf8 {
                        name: Arc::clone(name),
                        builder: StringBuilder::with_capacity(capacity, capacity * 24),
                    }
                } else if let Some(raw_code) = raw_code {
                    ColumnBuilder::Int32 {
                        name: Arc::clone(name),
                        builder: Int32Builder::with_capacity(capacity),
                        raw_code: raw_code.clone(),
                    }
                } else {
                    ColumnBuilder::Float64 {
                        name: Arc::clone(name),
                        builder: Float64Builder::with_capacity(capacity),
                    }
                }
            }
            SchemaItem::ReplicationGroup { name, children } => {
                let child_fields = schema_to_fields(children);
                let fields_arc = Fields::from(child_fields.clone());

                // Create field builders for the struct (recursively for nested lists)
                let field_builders: Vec<Box<dyn ArrayBuilder>> = child_fields
                    .iter()
                    .map(|f| create_field_builder(f))
                    .collect();

                let struct_builder = StructBuilder::new(fields_arc.clone(), field_builders);

                ColumnBuilder::List {
                    name: Arc::clone(name),
                    struct_fields: fields_arc,
                    children_schema: children.clone(),
                    offsets: {
                        let mut v = Vec::with_capacity(capacity + 1);
                        v.push(0);
                        v
                    },
                    current_offset: 0,
                    struct_builder,
                }
            }
        })
        .collect()
}

/// Recursively create an Arrow ArrayBuilder for a given Field.
/// Handles nested List<Struct<...>> at arbitrary depth.
fn create_field_builder(field: &Field) -> Box<dyn ArrayBuilder> {
    match field.data_type() {
        DataType::Utf8 => Box::new(StringBuilder::new()),
        DataType::Float64 => Box::new(Float64Builder::new()),
        DataType::Int32 => Box::new(Int32Builder::new()),
        DataType::List(inner) => {
            let DataType::Struct(sf) = inner.data_type() else {
                unreachable!("replication lists hold structs")
            };
            let nested_builders: Vec<Box<dyn ArrayBuilder>> =
                sf.iter().map(|nf| create_field_builder(nf)).collect();
            Box::new(ListBuilder::new(StructBuilder::new(
                sf.clone(),
                nested_builders,
            )))
        }
        other => unreachable!("schema fields have no {other} type"),
    }
}

/// Decode uncompressed BUFR data directly into Arrow builders: values go
/// straight from the BitReader into the Arrow column builders.
fn decode_to_arrow_uncompressed(
    data: &[u8],
    expanded: &[ExpandedItem],
    decode_plans: &[Option<DecodePlan>],
    num_subsets: u16,
    builders: &mut [ColumnBuilder],
    skip_mask: Option<&[bool]>,
    policy: ReadPolicy,
) -> Result<(), String> {
    let mut reader = BitReader::with_policy(data, policy);

    let use_subset_framing = num_subsets > 1;
    match skip_mask {
        Some(mask) => {
            for _subset_idx in 0..num_subsets {
                decode_with_subset_frame(
                    &mut reader,
                    expanded,
                    use_subset_framing,
                    policy,
                    |subset_reader| {
                        stream_subset_uncompressed_filtered(
                            subset_reader,
                            expanded,
                            decode_plans,
                            builders,
                            mask,
                        )
                    },
                )?;
            }
        }
        None => {
            for _subset_idx in 0..num_subsets {
                decode_with_subset_frame(
                    &mut reader,
                    expanded,
                    use_subset_framing,
                    policy,
                    |subset_reader| {
                        stream_subset_uncompressed(subset_reader, expanded, decode_plans, builders)
                    },
                )?;
            }
        }
    }

    reader.validate_bounds()
}

/// Stream-decode one subset directly into the column builders.
fn stream_subset_uncompressed(
    reader: &mut BitReader,
    expanded: &[ExpandedItem],
    decode_plans: &[Option<DecodePlan>],
    builders: &mut [ColumnBuilder],
) -> Result<(), String> {
    let mut schema_idx = 0;

    for (item, decode_plan) in expanded.iter().zip(decode_plans) {
        match item {
            ExpandedItem::Descriptor(descriptor, entry) => {
                if entry.bit_width == 0 {
                    continue; // skip zero-width, no schema entry
                }

                stream_scalar_value(reader, *descriptor, entry, &mut builders[schema_idx])?;
                schema_idx += 1;
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication {
                factor_entry,
                group,
                ..
            } => {
                let decode_plan = decode_plan
                    .as_ref()
                    .ok_or_else(|| "Missing uncompressed replication decode plan".to_string())?;
                // Read replication count (never "missing" for class-31 factors)
                let count = reader.read_bits(factor_entry.bit_width) as usize;

                stream_replication(reader, group, decode_plan, count, &mut builders[schema_idx])?;
                schema_idx += 1;
            }
        }
    }

    Ok(())
}

/// Read a scalar value from BitReader and append to the appropriate builder.
#[inline]
fn stream_scalar_value(
    reader: &mut BitReader,
    descriptor: u32,
    entry: &TableBEntry,
    builder: &mut ColumnBuilder,
) -> Result<(), String> {
    if entry.is_string {
        match builder {
            ColumnBuilder::Utf8 { builder: b, .. } => {
                if let Some(value) = reader.read_string_value(entry.bit_width / 8)? {
                    b.append_value(&value);
                } else {
                    b.append_null();
                }
            }
            _ => return Err(format!("Expected Utf8 builder for {}", entry.mnemonic)),
        }
    } else {
        let raw = reader.read_bits(entry.bit_width);
        match builder {
            ColumnBuilder::Float64 { builder: b, .. } => {
                if uses_missing_value(descriptor) && BitReader::is_missing(raw, entry.bit_width) {
                    b.append_null();
                } else {
                    let scaled =
                        (entry.reference_value + raw as i64) as f64 / 10f64.powi(entry.scale);
                    b.append_value(scaled);
                }
            }
            ColumnBuilder::Int32 { builder: b, .. } => {
                if uses_missing_value(descriptor) && BitReader::is_missing(raw, entry.bit_width) {
                    b.append_null();
                } else {
                    b.append_value(raw as i32);
                }
            }
            _ => return Err(format!("Expected numeric builder for {}", entry.mnemonic)),
        }
    }
    Ok(())
}

/// Pre-computed decode operation for one field in a replication group.
/// Avoids repeated string matching and pattern matching in the inner loop.
#[derive(Debug)]
enum FieldOp {
    Numeric {
        bit_width: u32,
        scale: i32,
        reference_value: i64,
        uses_missing_value: bool,
        output: NumericOutput,
    },
    String {
        byte_count: u32,
    },
    /// Nested delayed replication — must fall back to tree walk
    NestedReplication,
}

/// Immutable field operations compiled once for one replication group.
#[derive(Debug)]
struct DecodePlan {
    fields: Vec<FieldOp>,
    has_nested: bool,
}

/// Build a flat decode plan from an expanded item group.
/// Returns the plan and whether it contains any nested replications.
fn build_decode_plan(group: &[ExpandedItem], schema: &[SchemaItem]) -> DecodePlan {
    let mut fields = Vec::new();
    let mut has_nested = false;
    let mut schema_index = 0;
    for item in group {
        match item {
            ExpandedItem::Descriptor(desc, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }
                if entry.is_string {
                    fields.push(FieldOp::String {
                        byte_count: entry.bit_width / 8,
                    });
                } else {
                    let output = match &schema[schema_index] {
                        SchemaItem::Scalar {
                            raw_code: Some(_), ..
                        } => NumericOutput::RawInt32,
                        SchemaItem::Scalar { .. } => NumericOutput::PhysicalFloat64,
                        SchemaItem::ReplicationGroup { .. } => {
                            unreachable!("descriptor decode plan has replication schema")
                        }
                    };
                    fields.push(FieldOp::Numeric {
                        bit_width: entry.bit_width,
                        scale: entry.scale,
                        reference_value: entry.reference_value,
                        uses_missing_value: uses_missing_value(*desc),
                        output,
                    });
                }
                schema_index += 1;
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication { .. } => {
                has_nested = true;
                fields.push(FieldOp::NestedReplication);
                schema_index += 1;
            }
        }
    }
    DecodePlan { fields, has_nested }
}

/// Compile replication plans parallel to one expanded descriptor sequence.
fn build_decode_plans(expanded: &[ExpandedItem], schema: &[SchemaItem]) -> Vec<Option<DecodePlan>> {
    let mut schema_index = 0;
    expanded
        .iter()
        .map(|item| match item {
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication { group, .. } => {
                let children = match &schema[schema_index] {
                    SchemaItem::ReplicationGroup { children, .. } => children,
                    SchemaItem::Scalar { .. } => return None,
                };
                schema_index += 1;
                Some(build_decode_plan(group, children))
            }
            ExpandedItem::Descriptor(_, entry) => {
                if entry.bit_width > 0 {
                    schema_index += 1;
                }
                None
            }
        })
        .collect()
}

/// Stream-decode a delayed replication group directly into a ListBuilder<StructBuilder>.
/// For each iteration, decode the group items and append to the struct builder.
/// This avoids any per-iteration Vec allocation — values go straight to Arrow.
fn stream_replication(
    reader: &mut BitReader,
    group: &[ExpandedItem],
    decode_plan: &DecodePlan,
    count: usize,
    builder: &mut ColumnBuilder,
) -> Result<(), String> {
    let (struct_builder, children_schema_ref, offsets, current_offset) = match builder {
        ColumnBuilder::List {
            struct_builder,
            children_schema: cs,
            offsets,
            current_offset,
            ..
        } => (struct_builder, cs, offsets, current_offset),
        _ => return Err("Expected List builder for replication".into()),
    };

    if !decode_plan.has_nested {
        // Fast path: iterate plan instead of tree — no nested replications
        for _ in 0..count {
            let mut field_idx = 0;
            for op in &decode_plan.fields {
                match op {
                    FieldOp::Numeric {
                        bit_width,
                        scale,
                        reference_value,
                        uses_missing_value,
                        output,
                    } => {
                        let raw = reader.read_bits(*bit_width);
                        if *uses_missing_value && BitReader::is_missing(raw, *bit_width) {
                            match output {
                                NumericOutput::PhysicalFloat64 => struct_builder
                                    .field_builder::<Float64Builder>(field_idx)
                                    .ok_or_else(|| {
                                        format!("No Float64Builder at field {field_idx}")
                                    })?
                                    .append_null(),
                                NumericOutput::RawInt32 => struct_builder
                                    .field_builder::<Int32Builder>(field_idx)
                                    .ok_or_else(|| format!("No Int32Builder at field {field_idx}"))?
                                    .append_null(),
                            }
                        } else {
                            match output {
                                NumericOutput::PhysicalFloat64 => {
                                    let scaled =
                                        (*reference_value + raw as i64) as f64 / 10f64.powi(*scale);
                                    struct_builder
                                        .field_builder::<Float64Builder>(field_idx)
                                        .ok_or_else(|| {
                                            format!("No Float64Builder at field {field_idx}")
                                        })?
                                        .append_value(scaled);
                                }
                                NumericOutput::RawInt32 => struct_builder
                                    .field_builder::<Int32Builder>(field_idx)
                                    .ok_or_else(|| format!("No Int32Builder at field {field_idx}"))?
                                    .append_value(raw as i32),
                            }
                        }
                        field_idx += 1;
                    }
                    FieldOp::String { byte_count } => {
                        let builder = struct_builder
                            .field_builder::<StringBuilder>(field_idx)
                            .ok_or_else(|| format!("No StringBuilder at field {field_idx}"))?;
                        if let Some(value) = reader.read_string_value(*byte_count)? {
                            builder.append_value(&value);
                        } else {
                            builder.append_null();
                        }
                        field_idx += 1;
                    }
                    FieldOp::NestedReplication => unreachable!(),
                }
            }
            struct_builder.append(true);
            *current_offset += 1;
        }
    } else {
        // Slow path: has nested replications, fall back to tree walk
        for _ in 0..count {
            let mut field_idx = 0;
            for item in group {
                match item {
                    ExpandedItem::Descriptor(descriptor, entry) => {
                        if entry.bit_width == 0 {
                            continue;
                        }

                        if entry.is_string {
                            let builder = struct_builder
                                .field_builder::<StringBuilder>(field_idx)
                                .ok_or_else(|| format!("No StringBuilder at field {field_idx}"))?;
                            if let Some(value) = reader.read_string_value(entry.bit_width / 8)? {
                                builder.append_value(&value);
                            } else {
                                builder.append_null();
                            }
                        } else {
                            let raw = reader.read_bits(entry.bit_width);
                            let output = match &children_schema_ref[field_idx] {
                                SchemaItem::Scalar {
                                    raw_code: Some(_), ..
                                } => NumericOutput::RawInt32,
                                _ => NumericOutput::PhysicalFloat64,
                            };
                            if uses_missing_value(*descriptor)
                                && BitReader::is_missing(raw, entry.bit_width)
                            {
                                match output {
                                    NumericOutput::PhysicalFloat64 => struct_builder
                                        .field_builder::<Float64Builder>(field_idx)
                                        .ok_or_else(|| {
                                            format!("No Float64Builder at field {field_idx}")
                                        })?
                                        .append_null(),
                                    NumericOutput::RawInt32 => struct_builder
                                        .field_builder::<Int32Builder>(field_idx)
                                        .ok_or_else(|| {
                                            format!("No Int32Builder at field {field_idx}")
                                        })?
                                        .append_null(),
                                }
                            } else {
                                match output {
                                    NumericOutput::PhysicalFloat64 => {
                                        let scaled = (entry.reference_value + raw as i64) as f64
                                            / 10f64.powi(entry.scale);
                                        struct_builder
                                            .field_builder::<Float64Builder>(field_idx)
                                            .ok_or_else(|| {
                                                format!("No Float64Builder at field {field_idx}")
                                            })?
                                            .append_value(scaled);
                                    }
                                    NumericOutput::RawInt32 => struct_builder
                                        .field_builder::<Int32Builder>(field_idx)
                                        .ok_or_else(|| {
                                            format!("No Int32Builder at field {field_idx}")
                                        })?
                                        .append_value(raw as i32),
                                }
                            }
                        }
                        field_idx += 1;
                    }
                    ExpandedItem::FixedReplication { .. } => {
                        unreachable!("flat expansion inlines fixed replication")
                    }
                    ExpandedItem::DelayedReplication {
                        factor_entry,
                        group: inner_group,
                        ..
                    } => {
                        // Nested delayed replication inside a struct
                        // Replication counts are never "missing"
                        let inner_count = reader.read_bits(factor_entry.bit_width) as usize;

                        // Get the inner ListBuilder from the struct builder
                        let inner_children = match &children_schema_ref[field_idx] {
                            SchemaItem::ReplicationGroup { children, .. } => children,
                            _ => return Err("Schema mismatch at nested replication".into()),
                        };

                        let inner_list_builder = struct_builder
                            .field_builder::<ListBuilder<StructBuilder>>(field_idx)
                            .ok_or_else(|| {
                                format!("No ListBuilder<StructBuilder> at field {field_idx}")
                            })?;

                        // For each inner iteration, decode into the nested struct builder
                        for _ in 0..inner_count {
                            stream_nested_struct(
                                reader,
                                inner_group,
                                inner_children,
                                inner_list_builder.values(),
                            )?;
                            inner_list_builder.values().append(true);
                        }
                        inner_list_builder.append(true);

                        field_idx += 1;
                    }
                }
            }

            // Mark this struct row as valid
            struct_builder.append(true);
            *current_offset += 1;
        }
    }

    // Record the offset for this list entry
    offsets.push(*current_offset);

    Ok(())
}

/// Decode one iteration of a nested struct directly into a StructBuilder.
/// Handles further nesting recursively.
fn stream_nested_struct(
    reader: &mut BitReader,
    group: &[ExpandedItem],
    children_schema: &[SchemaItem],
    struct_builder: &mut StructBuilder,
) -> Result<(), String> {
    let mut field_idx = 0;
    for item in group {
        match item {
            ExpandedItem::Descriptor(descriptor, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }
                if entry.is_string {
                    let builder = struct_builder
                        .field_builder::<StringBuilder>(field_idx)
                        .ok_or_else(|| format!("No StringBuilder at nested field {field_idx}"))?;
                    if let Some(value) = reader.read_string_value(entry.bit_width / 8)? {
                        builder.append_value(&value);
                    } else {
                        builder.append_null();
                    }
                } else {
                    let raw = reader.read_bits(entry.bit_width);
                    if uses_missing_value(*descriptor)
                        && BitReader::is_missing(raw, entry.bit_width)
                    {
                        struct_builder
                            .field_builder::<Float64Builder>(field_idx)
                            .ok_or_else(|| {
                                format!("No Float64Builder at nested field {field_idx}")
                            })?
                            .append_null();
                    } else {
                        let scaled =
                            (entry.reference_value + raw as i64) as f64 / 10f64.powi(entry.scale);
                        struct_builder
                            .field_builder::<Float64Builder>(field_idx)
                            .ok_or_else(|| {
                                format!("No Float64Builder at nested field {field_idx}")
                            })?
                            .append_value(scaled);
                    }
                }
                field_idx += 1;
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication {
                factor_entry,
                group: inner_group,
                ..
            } => {
                // Replication counts are never "missing"
                let inner_count = reader.read_bits(factor_entry.bit_width) as usize;
                let inner_children = match &children_schema[field_idx] {
                    SchemaItem::ReplicationGroup { children, .. } => children,
                    _ => return Err("Schema mismatch at deeply nested replication".into()),
                };

                let inner_list_builder = struct_builder
                    .field_builder::<ListBuilder<StructBuilder>>(field_idx)
                    .ok_or_else(|| {
                        format!("No ListBuilder<StructBuilder> at nested field {field_idx}")
                    })?;

                for _ in 0..inner_count {
                    stream_nested_struct(
                        reader,
                        inner_group,
                        inner_children,
                        inner_list_builder.values(),
                    )?;
                    inner_list_builder.values().append(true);
                }
                inner_list_builder.append(true);

                field_idx += 1;
            }
        }
    }
    Ok(())
}

/// Skip an entire delayed replication group in the BitReader.
fn skip_replication(
    reader: &mut BitReader,
    group: &[ExpandedItem],
    count: usize,
) -> Result<(), String> {
    for _ in 0..count {
        for item in group {
            skip_expanded_item(reader, item)?;
        }
    }
    Ok(())
}

/// Skip a single expanded item (scalar or nested replication).
fn skip_expanded_item(reader: &mut BitReader, item: &ExpandedItem) -> Result<(), String> {
    match item {
        ExpandedItem::Descriptor(_, entry) => {
            reader.skip_bits(entry.bit_width);
        }
        ExpandedItem::FixedReplication { .. } => {
            unreachable!("flat expansion inlines fixed replication")
        }
        ExpandedItem::DelayedReplication {
            factor_entry,
            group,
            ..
        } => {
            // Replication counts are never "missing"
            let count = reader.read_bits(factor_entry.bit_width) as usize;
            skip_replication(reader, group, count)?;
        }
    }
    Ok(())
}

/// Stream-decode one subset with mnemonic filtering.
/// `skip_mask[i]` is true if schema item `i` should be skipped (not written to builder).
fn stream_subset_uncompressed_filtered(
    reader: &mut BitReader,
    expanded: &[ExpandedItem],
    decode_plans: &[Option<DecodePlan>],
    builders: &mut [ColumnBuilder],
    skip_mask: &[bool],
) -> Result<(), String> {
    let mut schema_idx = 0;

    for (item, decode_plan) in expanded.iter().zip(decode_plans) {
        match item {
            ExpandedItem::Descriptor(descriptor, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }

                if skip_mask[schema_idx] {
                    reader.skip_bits(entry.bit_width);
                } else {
                    stream_scalar_value(reader, *descriptor, entry, &mut builders[schema_idx])?;
                }
                schema_idx += 1;
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication {
                factor_entry,
                group,
                ..
            } => {
                let decode_plan = decode_plan
                    .as_ref()
                    .ok_or_else(|| "Missing filtered replication decode plan".to_string())?;
                // Replication counts are never "missing"
                let count = reader.read_bits(factor_entry.bit_width) as usize;

                if skip_mask[schema_idx] {
                    skip_replication(reader, group, count)?;
                } else {
                    stream_replication(
                        reader,
                        group,
                        decode_plan,
                        count,
                        &mut builders[schema_idx],
                    )?;
                }
                schema_idx += 1;
            }
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust-side flatten: performs row-expansion and unit-list unwrapping in a
// single pass before crossing the FFI boundary to Python.
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively flatten a RecordBatch until no List<Struct> columns remain.
///
/// Phase 1: Row-expanding flatten of the CAT-bearing list column (if any).
/// Phase 2: Iteratively unwrap List<Struct> columns by taking element[0] and
///          promoting struct fields to top-level columns.
fn flatten_record_batch(batch: RecordBatch) -> Result<RecordBatch, String> {
    // Phase 1: Find multi-element list column to expand (contains "CAT" field)
    let batch = phase1_row_expand(batch)?;

    // Phase 2: Iteratively unwrap list<struct> columns
    phase2_unwrap_lists(batch)
}

/// `name`, or `name_2`, `name_3`, ... when an earlier field already has it.
fn unique_field_name(fields: &[Field], name: &str) -> String {
    let mut candidate = name.to_string();
    let mut suffix = 2u32;
    while fields.iter().any(|field| field.name() == &candidate) {
        candidate = format!("{name}_{suffix}");
        suffix += 1;
    }
    candidate
}

/// Phase 1: Find a List<Struct> column whose inner struct has a "CAT" field
/// (NCEP PrepBUFR convention) and whose max list length > 1, and expand it into
/// rows, broadcasting other columns. Rows whose CAT list is empty or null are
/// dropped. The expanded column keeps its position as a list with one entry per
/// row, so phase 2 promotes its fields as it does for any other list. If no such
/// CAT list exists, Phase 1 is skipped (no fallback).
///
/// Non-CAT multi-element List<Struct> columns are not expanded. In PrepBUFR
/// they are ancillary replications (LOCAL descriptor arrays on profiler and
/// aircraft reports, present-weather and cloud groups on surface reports);
/// expanding one duplicates observations, and the non-empty filter below would
/// drop stations without the group.
fn phase1_row_expand(batch: RecordBatch) -> Result<RecordBatch, String> {
    let schema = batch.schema();

    // Find the expand target: a CAT-bearing multi-element list (and only that).
    let mut expand_target: Option<(usize, FieldRef)> = None;
    for i in 0..schema.fields().len() {
        let field = schema.field(i);
        if let DataType::List(inner_field) = field.data_type() {
            if let DataType::Struct(struct_fields) = inner_field.data_type() {
                // Check max length > 1
                let list_col = batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| "Expected ListArray".to_string())?;
                let max_len = (0..list_col.len())
                    .filter(|&r| list_col.is_valid(r))
                    .map(|r| list_col.value_length(r))
                    .max()
                    .unwrap_or(0);
                if max_len > 1 && struct_fields.iter().any(|f| f.name() == "CAT") {
                    expand_target = Some((i, inner_field.clone()));
                    break;
                }
            }
        }
    }

    let Some((expand_idx, item_field)) = expand_target else {
        return Ok(batch); // No CAT level list — skip phase 1
    };

    // Perform the row-expanding flatten
    let list_col = batch
        .column(expand_idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    // Filter to rows where list is non-empty and valid
    let num_rows = batch.num_rows();
    let valid_mask: BooleanArray = (0..num_rows)
        .map(|i| Some(list_col.is_valid(i) && list_col.value_length(i) > 0))
        .collect();

    let filtered =
        filter_record_batch(&batch, &valid_mask).map_err(|e| format!("filter error: {e}"))?;

    let filtered_list = filtered
        .column(expand_idx)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    // Compute parent indices: for each item in the flattened list, which row it came from
    let offsets = filtered_list.offsets();
    let first_item = offsets[0] as usize;
    let item_count = *offsets.last().unwrap() as usize - first_item;
    let mut parent_indices = Vec::with_capacity(item_count);
    for row in 0..filtered_list.len() {
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        for _ in start..end {
            parent_indices.push(row as u32);
        }
    }
    let parent_idx_array = UInt32Array::from(parent_indices);

    let expanded_list: ArrayRef = Arc::new(
        ListArray::try_new(
            item_field,
            OffsetBuffer::from_lengths(std::iter::repeat_n(1, item_count)),
            filtered_list.values().slice(first_item, item_count),
            None,
        )
        .map_err(|e| {
            format!(
                "list error expanding '{}': {e}",
                schema.field(expand_idx).name()
            )
        })?,
    );
    let mut new_arrays: Vec<ArrayRef> = Vec::with_capacity(filtered.num_columns());
    for i in 0..filtered.num_columns() {
        if i == expand_idx {
            new_arrays.push(expanded_list.clone());
            continue;
        }
        // Broadcast via take
        let taken = take(filtered.column(i).as_ref(), &parent_idx_array, None)
            .map_err(|e| format!("take error on column '{}': {e}", schema.field(i).name()))?;
        new_arrays.push(taken);
    }
    RecordBatch::try_new(schema, new_arrays).map_err(|e| format!("RecordBatch error: {e}"))
}

fn list_struct_fields(data_type: &DataType) -> Option<&Fields> {
    match data_type {
        DataType::List(inner) => match inner.data_type() {
            DataType::Struct(struct_fields) => Some(struct_fields),
            _ => None,
        },
        _ => None,
    }
}

/// List<Struct> nesting levels `phase2_unwrap_lists` unwraps.
const MAX_LIST_UNWRAP_ROUNDS: usize = 10;

/// Phase 2: Iteratively unwrap all List<Struct> columns by extracting element[0].
/// Struct fields are promoted to top-level columns, null where the list is empty
/// or null, including a list empty in every row. Repeats until no list<struct>
/// remains, and fails when lists nest deeper than `MAX_LIST_UNWRAP_ROUNDS`.
fn phase2_unwrap_lists(mut batch: RecordBatch) -> Result<RecordBatch, String> {
    for round in 0..=MAX_LIST_UNWRAP_ROUNDS {
        let schema = batch.schema();
        let mut has_list_struct = false;

        let mut new_fields: Vec<Field> = Vec::new();
        let mut new_arrays: Vec<ArrayRef> = Vec::new();

        for i in 0..schema.fields().len() {
            let field = schema.field(i);
            let col = batch.column(i);

            let Some(list_struct_fields) = list_struct_fields(field.data_type()) else {
                // Keep non-list columns as-is
                new_fields.push(field.clone());
                new_arrays.push(col.clone());
                continue;
            };

            let list_col = col
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| format!("Expected ListArray for '{}'", field.name()))?;

            // Check if any entries are non-empty
            let max_len = (0..list_col.len())
                .filter(|&r| list_col.is_valid(r))
                .map(|r| list_col.value_length(r))
                .max()
                .unwrap_or(0);

            has_list_struct = true;

            // A list empty in every row promotes its struct fields as all-null columns.
            if max_len == 0 {
                for sf in list_struct_fields.iter() {
                    let name = unique_field_name(&new_fields, sf.name());
                    new_fields.push(Field::new(name, sf.data_type().clone(), true));
                    new_arrays.push(new_null_array(sf.data_type(), list_col.len()));
                }
                continue;
            }

            // Extract element[0] from each list entry.
            // For valid non-empty lists: index = offsets[row]
            // For invalid/empty lists: index = 0 (will be nulled out)
            let offsets = list_col.offsets();
            let num_rows = list_col.len();

            let mut indices = Vec::with_capacity(num_rows);
            let mut valid_bits = Vec::with_capacity(num_rows);
            for row in 0..num_rows {
                let is_valid = list_col.is_valid(row) && list_col.value_length(row) > 0;
                valid_bits.push(is_valid);
                if is_valid {
                    indices.push(offsets[row] as u32);
                } else {
                    indices.push(0); // dummy index, will be nulled
                }
            }
            let idx_array = UInt32Array::from(indices);

            // Take from the values (struct) array
            let values = list_col.values();
            let taken = take(values.as_ref(), &idx_array, None)
                .map_err(|e| format!("take error unwrapping '{}': {e}", field.name()))?;

            // The taken array is a StructArray — extract fields
            let struct_arr = taken
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    format!(
                        "Expected StructArray from list values of '{}'",
                        field.name()
                    )
                })?;

            let struct_dt = struct_arr.data_type();
            let struct_fields = match struct_dt {
                DataType::Struct(sf) => sf,
                _ => return Err("Unexpected non-struct".into()),
            };

            // Create validity mask as BooleanArray for null propagation
            let valid_mask: BooleanArray = valid_bits.iter().map(|&v| Some(v)).collect();
            let has_nulls = valid_bits.iter().any(|&v| !v);

            for (j, sf) in struct_fields.iter().enumerate() {
                let name = unique_field_name(&new_fields, sf.name());
                let child = struct_arr.column(j).clone();

                // Apply null mask: where list was empty/null, child should be null
                let final_child = if has_nulls {
                    apply_null_mask(child, &valid_mask)?
                } else {
                    child
                };

                new_fields.push(Field::new(name, final_child.data_type().clone(), true));
                new_arrays.push(final_child);
            }
        }

        if !has_list_struct {
            // No List<Struct> columns remain
            return Ok(batch);
        }
        if round == MAX_LIST_UNWRAP_ROUNDS {
            break;
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        batch = RecordBatch::try_new(new_schema, new_arrays)
            .map_err(|e| format!("RecordBatch error in unwrap: {e}"))?;
    }

    Err(format!(
        "List<Struct> columns nest more than {MAX_LIST_UNWRAP_ROUNDS} levels deep"
    ))
}

/// Apply a boolean validity mask to an array: where mask is false, set to null.
fn apply_null_mask(array: ArrayRef, mask: &BooleanArray) -> Result<ArrayRef, String> {
    // Convert BooleanArray to Vec<bool> for NullBuffer
    let mask_bools: Vec<bool> = (0..mask.len())
        .map(|i| mask.is_valid(i) && mask.value(i))
        .collect();

    // If array already has nulls, combine them (AND the two masks)
    let combined_bools: Vec<bool> = match array.logical_nulls() {
        Some(existing) => existing
            .iter()
            .zip(mask_bools.iter())
            .map(|(a, &b)| a && b)
            .collect(),
        None => mask_bools,
    };
    let combined = NullBuffer::from(combined_bools);

    // Rebuild the array data with the new null buffer
    let data = array.to_data();
    let new_data = data
        .into_builder()
        .null_bit_buffer(Some(combined.into_inner().into_inner()))
        .build()
        .map_err(|e| format!("Failed to rebuild array with null mask: {e}"))?;

    Ok(make_array(new_data))
}

/// Convert the filled ColumnBuilder tree into an Arrow RecordBatch.
fn builders_to_record_batch(
    builders: Vec<ColumnBuilder>,
    num_rows: usize,
    projection_roots: Option<&[String]>,
) -> Result<RecordBatch, String> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for builder in builders {
        match builder {
            ColumnBuilder::Float64 { name, mut builder } => {
                if let Some(m) = projection_roots {
                    if !m.iter().any(|mn| mn.as_str() == &*name) {
                        continue;
                    }
                }
                fields.push(Field::new(&*name, DataType::Float64, true));
                arrays.push(Arc::new(builder.finish()));
            }
            ColumnBuilder::Int32 {
                name,
                mut builder,
                raw_code,
            } => {
                if let Some(m) = projection_roots {
                    if !m.iter().any(|mn| mn.as_str() == &*name) {
                        continue;
                    }
                }
                fields.push(
                    Field::new(&*name, DataType::Int32, true)
                        .with_metadata(raw_code.field_metadata()),
                );
                arrays.push(Arc::new(builder.finish()));
            }
            ColumnBuilder::Utf8 { name, mut builder } => {
                if let Some(m) = projection_roots {
                    if !m.iter().any(|mn| mn.as_str() == &*name) {
                        continue;
                    }
                }
                fields.push(Field::new(&*name, DataType::Utf8, true));
                arrays.push(Arc::new(builder.finish()));
            }
            ColumnBuilder::List {
                name,
                struct_fields,
                mut struct_builder,
                offsets,
                ..
            } => {
                if let Some(m) = projection_roots {
                    if !m.iter().any(|mn| mn.as_str() == &*name) {
                        continue;
                    }
                }

                let struct_array = struct_builder.finish();
                let offsets_buffer = OffsetBuffer::new(ScalarBuffer::from(offsets));
                let inner_field =
                    Arc::new(Field::new("item", DataType::Struct(struct_fields), true));
                let list_array = ListArray::try_new(
                    inner_field.clone(),
                    offsets_buffer,
                    Arc::new(struct_array),
                    None,
                )
                .map_err(|e| format!("Failed to create ListArray: {e}"))?;

                fields.push(Field::new(&*name, DataType::List(inner_field), true));
                arrays.push(Arc::new(list_array));
            }
        }
    }

    if arrays.is_empty() {
        return Err("No matching columns found".into());
    }

    // Validate all arrays have the same length
    let expected_len = num_rows;
    for (i, arr) in arrays.iter().enumerate() {
        if arr.len() != expected_len {
            return Err(format!(
                "Column '{}' has {} rows but expected {}",
                fields[i].name(),
                arr.len(),
                expected_len,
            ));
        }
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| format!("Arrow error: {e}"))
}

/// Read one compressed IA5 element and pass each subset's string to
/// `store`.
///
/// Without increments every subset holds the reference. With increments a
/// missing (all-ones) increment is null and a blank increment holds the
/// reference, which is null when the reference is missing.
fn read_compressed_strings(
    reader: &mut BitReader,
    entry: &TableBEntry,
    n: usize,
    mut store: impl FnMut(usize, Option<String>),
) -> Result<(), String> {
    let reference = reader.read_string_value(entry.bit_width / 8)?;
    let increment_bytes = reader.read_bits(6) as u32;
    if increment_bytes == 0 {
        for subset in 0..n {
            store(subset, reference.clone());
        }
        return Ok(());
    }
    if reader.strict {
        return Err(format!(
            "Compressed IA5 increments are unsupported in strict mode for {}",
            entry.mnemonic
        ));
    }
    for subset in 0..n {
        let value = match reader.read_string_value(increment_bytes)? {
            Some(local) if local.trim().is_empty() => reference.clone(),
            local => local,
        };
        store(subset, value);
    }
    Ok(())
}

/// Compressed delayed replication stores one count for all subsets; the
/// per-subset increments are skipped.
fn read_compressed_replication_count(reader: &mut BitReader, bit_width: u32, n: usize) -> usize {
    let count = reader.read_bits(bit_width) as usize;
    let increment_bits = reader.read_bits(6) as u32;
    reader.skip_bits(increment_bits * n as u32);
    count
}

#[inline]
fn skip_compressed_values(reader: &mut BitReader, reference_bits: u32, n: usize) {
    reader.skip_bits(reference_bits);
    let increment_bits = reader.read_bits(6) as u32;
    reader.skip_bits(increment_bits * n as u32);
}

/// Compressed strings carry their increment width (NBINC) in bytes, not bits.
fn skip_compressed_strings(reader: &mut BitReader, byte_count: u32, n: usize) {
    reader.skip_bits(byte_count * 8);
    let increment_bytes = reader.read_bits(6) as u32;
    reader.skip_bits(increment_bytes * 8 * n as u32);
}

/// Decode a slice of uncompressed messages sequentially into shared builders.
/// `rows` is the messages' total subset count.
fn sequential_decode_uncompressed(
    messages: &[&BufrMessage],
    rows: usize,
    expanded: &[ExpandedItem],
    schema: &[SchemaItem],
    data_category: u8,
    projection_roots: Option<&[String]>,
    policy: ReadPolicy,
) -> Result<RecordBatch, String> {
    let mut builders = create_builders(schema, rows);
    let builders_len = builders.len();
    let mut total_subsets: usize = 0;
    let decode_plans =
        build_decode_plans(&expanded[..expanded.len() - 1], &schema[..schema.len() - 1]);

    let skip_mask: Option<Vec<bool>> = projection_roots.map(|mlist| {
        let mset: HashSet<&str> = mlist.iter().map(|s| s.as_str()).collect();
        schema[..schema.len() - 1]
            .iter()
            .map(|item| {
                let name = match item {
                    SchemaItem::Scalar { name, .. } => name,
                    SchemaItem::ReplicationGroup { name, .. } => name,
                };
                !mset.contains(&**name) // true = skip this column
            })
            .collect()
    });

    for msg in messages {
        let parsed = parse_message(msg).map_err(|e| e.to_string())?;
        total_subsets += parsed.num_subsets as usize;

        decode_to_arrow_uncompressed(
            parsed.data,
            &expanded[..expanded.len() - 1],
            &decode_plans,
            parsed.num_subsets,
            &mut builders[..builders_len - 1],
            skip_mask.as_deref(),
            policy,
        )?;

        if let ColumnBuilder::Float64 { builder: b, .. } = builders.last_mut().unwrap() {
            for _ in 0..parsed.num_subsets {
                b.append_value(data_category as f64);
            }
        }
    }

    builders_to_record_batch(builders, total_subsets, projection_roots)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel concatenation: each output buffer is allocated once and every piece
// is copied into its own disjoint range of it on the rayon pool.
// ─────────────────────────────────────────────────────────────────────────────

const BITMAP_CHUNK_BYTES: usize = 1 << 16;

#[derive(Clone, Copy)]
struct OutputPtr(*mut u8);

// SAFETY: tasks holding an `OutputPtr` write only to disjoint byte ranges.
unsafe impl Send for OutputPtr {}
unsafe impl Sync for OutputPtr {}

impl OutputPtr {
    fn get(self) -> *mut u8 {
        self.0
    }
}

fn prefix_sums(lengths: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut starts = vec![0];
    for length in lengths {
        starts.push(starts[starts.len() - 1] + length);
    }
    starts
}

/// Allocates `len` bytes without initializing them; callers write every byte
/// through the returned pointer before the buffer is read.
fn uninit_buffer(len: usize) -> (MutableBuffer, OutputPtr) {
    let mut buffer = MutableBuffer::with_capacity(len);
    // SAFETY: the capacity covers `len`; every byte is written before it is read.
    unsafe { buffer.set_len(len) };
    let ptr = OutputPtr(buffer.as_mut_ptr());
    (buffer, ptr)
}

fn concat_byte_slices(slices: &[&[u8]]) -> Buffer {
    let starts = prefix_sums(slices.iter().map(|slice| slice.len()));
    let (buffer, out) = uninit_buffer(starts[slices.len()]);
    slices
        .par_iter()
        .zip(starts.par_iter())
        .for_each(|(slice, &start)| {
            // SAFETY: `start..start + slice.len()` lies inside the buffer and no
            // other piece writes to it.
            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), out.get().add(start), slice.len())
            };
        });
    buffer.into()
}

fn set_bits_true(bytes: &mut [u8], start: usize, len: usize) {
    let end = start + len;
    let mut bit = start;
    while bit < end && !bit.is_multiple_of(8) {
        bytes[bit / 8] |= 1 << (bit % 8);
        bit += 1;
    }
    let whole_bytes_end = end / 8 * 8;
    if bit < whole_bytes_end {
        bytes[bit / 8..whole_bytes_end / 8].fill(0xFF);
        bit = whole_bytes_end;
    }
    while bit < end {
        bytes[bit / 8] |= 1 << (bit % 8);
        bit += 1;
    }
}

/// Concatenates bit ranges given as `(bytes, bit_offset)`, where `None` means
/// all bits set. Each task owns a range of output bytes, so pieces that share a
/// boundary byte are written by the same task.
fn concat_bits(sources: &[Option<(&[u8], usize)>], starts: &[usize]) -> Buffer {
    let total_bits = starts[sources.len()];
    let byte_len = total_bits.div_ceil(8);
    let (buffer, out) = uninit_buffer(byte_len);
    (0..byte_len.div_ceil(BITMAP_CHUNK_BYTES))
        .into_par_iter()
        .for_each(|chunk| {
            let first_byte = chunk * BITMAP_CHUNK_BYTES;
            let chunk_len = BITMAP_CHUNK_BYTES.min(byte_len - first_byte);
            // SAFETY: chunks cover disjoint byte ranges inside the buffer, and
            // the range is zeroed before the slice is formed.
            let bytes = unsafe {
                let ptr = out.get().add(first_byte);
                std::ptr::write_bytes(ptr, 0, chunk_len);
                std::slice::from_raw_parts_mut(ptr, chunk_len)
            };
            let bit_start = first_byte * 8;
            let bit_end = (bit_start + chunk_len * 8).min(total_bits);
            let mut piece = starts.partition_point(|&start| start <= bit_start) - 1;
            while piece < sources.len() && starts[piece] < bit_end {
                let from = starts[piece].max(bit_start);
                let to = starts[piece + 1].min(bit_end);
                if from < to {
                    match sources[piece] {
                        Some((data, offset)) => {
                            set_bits(
                                bytes,
                                data,
                                from - bit_start,
                                offset + from - starts[piece],
                                to - from,
                            );
                        }
                        None => set_bits_true(bytes, from - bit_start, to - from),
                    }
                }
                piece += 1;
            }
        });
    buffer.into()
}

fn concat_nulls(pieces: &[ArrayRef], starts: &[usize]) -> Option<NullBuffer> {
    let null_count: usize = pieces.iter().map(|piece| piece.null_count()).sum();
    if null_count == 0 {
        return None;
    }
    let sources: Vec<Option<(&[u8], usize)>> = pieces
        .iter()
        .map(|piece| {
            piece
                .nulls()
                .map(|nulls| (nulls.validity(), nulls.offset()))
        })
        .collect();
    let validity = BooleanBuffer::new(concat_bits(&sources, starts), 0, starts[pieces.len()]);
    // SAFETY: `null_count` is the sum of the pieces' null counts.
    Some(unsafe { NullBuffer::new_unchecked(validity, null_count) })
}

/// Rebases each piece's offsets onto its child start in the concatenated child.
fn concat_offsets(
    offsets: &[&[i32]],
    starts: &[usize],
    child_starts: &[usize],
) -> Result<OffsetBuffer<i32>, String> {
    let total = starts[offsets.len()];
    let child_total = i32::try_from(child_starts[offsets.len()])
        .map_err(|_| "Concatenated offsets overflow i32".to_string())?;
    let (buffer, out) = uninit_buffer((total + 1) * std::mem::size_of::<i32>());
    offsets
        .par_iter()
        .zip(starts.par_iter().zip(child_starts.par_iter()))
        .for_each(|(piece, (&start, &child_start))| {
            let target = out.get() as *mut i32;
            for (index, &offset) in piece[..piece.len() - 1].iter().enumerate() {
                // SAFETY: rows `start..start + piece.len() - 1` belong to this
                // piece; the buffer is aligned for i32 and the value is at most
                // `child_total`.
                unsafe {
                    target
                        .add(start + index)
                        .write(offset - piece[0] + child_start as i32)
                };
            }
        });
    // SAFETY: index `total` is the last slot of the buffer.
    unsafe { (out.get() as *mut i32).add(total).write(child_total) };
    let offsets = ScalarBuffer::new(buffer.into(), 0, total + 1);
    // SAFETY: offsets start at zero and are non-decreasing because each piece's
    // are, and consecutive pieces meet at their child start.
    Ok(unsafe { OffsetBuffer::new_unchecked(offsets) })
}

fn concat_byte_arrays<T: ByteArrayType<Offset = i32>>(
    pieces: &[ArrayRef],
    starts: &[usize],
) -> Result<ArrayRef, String> {
    let arrays: Vec<&GenericByteArray<T>> = pieces.iter().map(|piece| piece.as_bytes()).collect();
    let offsets: Vec<&[i32]> = arrays.iter().map(|array| array.value_offsets()).collect();
    let values: Vec<&[u8]> = arrays
        .iter()
        .zip(&offsets)
        .map(|(array, offsets)| {
            &array.values()[offsets[0] as usize..offsets[offsets.len() - 1] as usize]
        })
        .collect();
    let child_starts = prefix_sums(values.iter().map(|values| values.len()));
    let (nulls, (offsets, values)) = rayon::join(
        || concat_nulls(pieces, starts),
        || {
            rayon::join(
                || concat_offsets(&offsets, starts, &child_starts),
                || concat_byte_slices(&values),
            )
        },
    );
    // SAFETY: every value is copied whole from a valid piece of the same type.
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(offsets?, values, nulls)
    }))
}

fn primitive_value_bytes(array: &dyn Array) -> Option<&[u8]> {
    downcast_primitive_array!(
        array => Some(array.values().inner().as_slice()),
        _ => None
    )
}

/// Concatenates arrays of one type. Primitive, Boolean, Utf8, Binary, List,
/// FixedSizeList and Struct are copied in parallel; other types use arrow's
/// `concat`.
fn concat_arrays(pieces: &[ArrayRef]) -> Result<ArrayRef, String> {
    let data_type = pieces[0].data_type();
    if let Some(piece) = pieces.iter().find(|piece| piece.data_type() != data_type) {
        return Err(format!(
            "Cannot concatenate {data_type} with {}",
            piece.data_type()
        ));
    }
    if pieces.len() == 1 {
        return Ok(pieces[0].clone());
    }
    let starts = prefix_sums(pieces.iter().map(|piece| piece.len()));
    let total = starts[pieces.len()];
    if let Some(slices) = pieces
        .iter()
        .map(|piece| primitive_value_bytes(piece.as_ref()))
        .collect::<Option<Vec<_>>>()
    {
        let (nulls, values) = rayon::join(
            || concat_nulls(pieces, &starts),
            || concat_byte_slices(&slices),
        );
        let builder = ArrayData::builder(data_type.clone())
            .len(total)
            .add_buffer(values)
            .nulls(nulls);
        // SAFETY: the values buffer holds the `total` values of the pieces.
        return Ok(make_array(unsafe { builder.build_unchecked() }));
    }
    match data_type {
        DataType::Boolean => {
            let sources: Vec<Option<(&[u8], usize)>> = pieces
                .iter()
                .map(|piece| {
                    let values = piece.as_boolean().values();
                    Some((values.values(), values.offset()))
                })
                .collect();
            let (nulls, values) = rayon::join(
                || concat_nulls(pieces, &starts),
                || concat_bits(&sources, &starts),
            );
            Ok(Arc::new(BooleanArray::new(
                BooleanBuffer::new(values, 0, total),
                nulls,
            )))
        }
        DataType::Utf8 => concat_byte_arrays::<Utf8Type>(pieces, &starts),
        DataType::Binary => concat_byte_arrays::<BinaryType>(pieces, &starts),
        DataType::List(field) => {
            let lists: Vec<&ListArray> = pieces.iter().map(|piece| piece.as_list()).collect();
            let offsets: Vec<&[i32]> = lists.iter().map(|list| list.value_offsets()).collect();
            let children: Vec<ArrayRef> = lists
                .iter()
                .zip(&offsets)
                .map(|(list, offsets)| {
                    let first = offsets[0] as usize;
                    list.values()
                        .slice(first, offsets[offsets.len() - 1] as usize - first)
                })
                .collect();
            let (nulls, (offsets, values)) = rayon::join(
                || concat_nulls(pieces, &starts),
                || {
                    rayon::join(
                        || {
                            OffsetBuffer::<i32>::from_lengths(
                                lists.iter().flat_map(|list| list.offsets().lengths()),
                            )
                        },
                        || concat_arrays(&children),
                    )
                },
            );
            ListArray::try_new(field.clone(), offsets, values?, nulls)
                .map(|array| Arc::new(array) as ArrayRef)
                .map_err(|error| error.to_string())
        }
        DataType::FixedSizeList(field, size) => {
            let children: Vec<ArrayRef> = pieces
                .iter()
                .map(|piece| piece.as_fixed_size_list().values().clone())
                .collect();
            let (nulls, values) = rayon::join(
                || concat_nulls(pieces, &starts),
                || concat_arrays(&children),
            );
            FixedSizeListArray::try_new(field.clone(), *size, values?, nulls)
                .map(|array| Arc::new(array) as ArrayRef)
                .map_err(|error| error.to_string())
        }
        DataType::Struct(fields) => {
            let (nulls, columns) = rayon::join(
                || concat_nulls(pieces, &starts),
                || {
                    (0..fields.len())
                        .into_par_iter()
                        .map(|index| {
                            let children: Vec<ArrayRef> = pieces
                                .iter()
                                .map(|piece| piece.as_struct().column(index).clone())
                                .collect();
                            concat_arrays(&children)
                        })
                        .collect::<Result<Vec<_>, _>>()
                },
            );
            // SAFETY: each column concatenates the matching column of valid
            // pieces of this struct type, so lengths and types match.
            Ok(Arc::new(unsafe {
                StructArray::new_unchecked_with_length(fields.clone(), columns?, nulls, total)
            }))
        }
        _ => {
            let arrays: Vec<&dyn Array> = pieces.iter().map(|piece| piece.as_ref()).collect();
            concat(&arrays).map_err(|error| error.to_string())
        }
    }
}

/// Concatenates batches that share the first batch's schema. The input batches
/// are dropped on the rayon pool.
fn parallel_concat_batches(batches: Vec<RecordBatch>) -> Result<RecordBatch, String> {
    let Some(first) = batches.first() else {
        return Err("No batches to concatenate".into());
    };
    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }
    let schema = first.schema();
    let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
    let columns = (0..schema.fields().len())
        .into_par_iter()
        .map(|index| {
            let pieces: Vec<ArrayRef> = batches
                .iter()
                .map(|batch| batch.column(index).clone())
                .collect();
            concat_arrays(&pieces)
        })
        .collect::<Result<Vec<_>, _>>()?;
    batches.into_par_iter().for_each(drop);
    let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
    RecordBatch::try_new_with_options(schema, columns, &options).map_err(|error| error.to_string())
}

/// Least encoded message bytes per chunk when uncompressed messages are
/// decoded in parallel; each chunk has its own builders and is concatenated.
const MIN_UNCOMPRESSED_CHUNK_BYTES: usize = 1 << 20;

/// Decode uncompressed messages in parallel, returning a single RecordBatch.
/// Splits messages into at most one chunk per thread, each with at least
/// `MIN_UNCOMPRESSED_CHUNK_BYTES` of messages and its own builders, then
/// concatenates results.
/// `subset_counts` holds each message's subset count.
fn parallel_decode_uncompressed(
    messages: &[&BufrMessage],
    subset_counts: &[u16],
    expanded: &[ExpandedItem],
    schema: &[SchemaItem],
    data_category: u8,
    projection_roots: Option<&[String]>,
    policy: ReadPolicy,
) -> Result<RecordBatch, String> {
    let rows = |counts: &[u16]| counts.iter().map(|count| *count as usize).sum::<usize>();
    let encoded_bytes: usize = messages.iter().map(|message| message.bytes().len()).sum();
    let num_chunks = rayon::current_num_threads()
        .min(messages.len())
        .min(encoded_bytes / MIN_UNCOMPRESSED_CHUNK_BYTES);
    if num_chunks <= 1 {
        return sequential_decode_uncompressed(
            messages,
            rows(subset_counts),
            expanded,
            schema,
            data_category,
            projection_roots,
            policy,
        );
    }
    let chunk_size = messages.len().div_ceil(num_chunks);

    let chunks: Vec<(&[&BufrMessage], &[u16])> = messages
        .chunks(chunk_size)
        .zip(subset_counts.chunks(chunk_size))
        .collect();

    let chunk_batches: Vec<Result<RecordBatch, String>> = chunks
        .par_iter()
        .map(|(chunk, counts)| {
            sequential_decode_uncompressed(
                chunk,
                rows(counts),
                expanded,
                schema,
                data_category,
                projection_roots,
                policy,
            )
        })
        .collect();

    // Collect results
    let mut batches = Vec::with_capacity(chunk_batches.len());
    for result in chunk_batches {
        batches.push(result?);
    }

    parallel_concat_batches(batches).map_err(|e| format!("Arrow concat error: {e}"))
}

#[derive(Debug, Clone)]
enum StructuralReplication {
    Fixed {
        descriptor: u32,
        count: usize,
    },
    Delayed {
        factor_descriptor: u32,
        factor_mnemonic: Arc<str>,
    },
}

#[derive(Debug, Clone)]
enum StructuralSchemaItem {
    Scalar {
        descriptor: u32,
        entry: Arc<TableBEntry>,
        path: String,
        raw_code: Option<RawCodeSpec>,
    },
    Replication {
        name: Arc<str>,
        path: String,
        kind: StructuralReplication,
        children: Vec<StructuralSchemaItem>,
    },
}

impl StructuralSchemaItem {
    fn name(&self) -> &str {
        match self {
            Self::Scalar { entry, .. } => &entry.mnemonic,
            Self::Replication { name, .. } => name,
        }
    }
}

#[derive(Debug, Default)]
struct ProjectionTree {
    whole_field: bool,
    children: HashMap<String, ProjectionTree>,
}

fn fixed_replication_name(descriptor: u32, occurrence: usize) -> String {
    let f = (descriptor >> 14) & 0x03;
    let x = (descriptor >> 8) & 0x3f;
    let y = descriptor & 0xff;
    format!("fixed_{f}_{x:02}_{y:03}__occ_{occurrence}")
}

/// Column name of a delayed group within one sequence: the factor mnemonic,
/// then `MNEMONIC_2`, `MNEMONIC_3`, ... for later groups with the same factor.
fn delayed_group_name(seen: &mut HashMap<Arc<str>, usize>, factor_mnemonic: &Arc<str>) -> Arc<str> {
    let count = seen.entry(Arc::clone(factor_mnemonic)).or_insert(0);
    *count += 1;
    if *count == 1 {
        Arc::clone(factor_mnemonic)
    } else {
        Arc::from(format!("{factor_mnemonic}_{count}"))
    }
}

fn field_path(prefix: Option<&str>, name: &str) -> String {
    match prefix {
        Some(parent) => format!("{parent}.{name}"),
        None => name.to_string(),
    }
}

fn derive_structural_schema(
    expanded: &[ExpandedItem],
    prefix: Option<&str>,
) -> Vec<StructuralSchemaItem> {
    let mut schema = Vec::new();
    let mut delayed_names = HashMap::new();
    let mut fixed_names: HashMap<u32, usize> = HashMap::new();
    for item in expanded {
        match item {
            ExpandedItem::Descriptor(descriptor, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }
                schema.push(StructuralSchemaItem::Scalar {
                    descriptor: *descriptor,
                    entry: Arc::clone(entry),
                    path: field_path(prefix, &entry.mnemonic),
                    raw_code: None,
                });
            }
            ExpandedItem::FixedReplication {
                descriptor,
                count,
                group,
            } => {
                let occurrence = fixed_names.entry(*descriptor).or_insert(0);
                let name: Arc<str> = Arc::from(fixed_replication_name(*descriptor, *occurrence));
                *occurrence += 1;
                let path = field_path(prefix, &name);
                schema.push(StructuralSchemaItem::Replication {
                    name,
                    path: path.clone(),
                    kind: StructuralReplication::Fixed {
                        descriptor: *descriptor,
                        count: *count,
                    },
                    children: derive_structural_schema(group, Some(&path)),
                });
            }
            ExpandedItem::DelayedReplication {
                factor_desc,
                factor_entry,
                group,
            } => {
                let name = delayed_group_name(&mut delayed_names, &factor_entry.mnemonic);
                let path = field_path(prefix, &name);
                schema.push(StructuralSchemaItem::Replication {
                    name,
                    path: path.clone(),
                    kind: StructuralReplication::Delayed {
                        factor_descriptor: *factor_desc,
                        factor_mnemonic: Arc::clone(&factor_entry.mnemonic),
                    },
                    children: derive_structural_schema(group, Some(&path)),
                });
            }
        }
    }
    schema
}

/// Parse projection paths into a tree.
///
/// The flat layout accepts top-level fields and one nested child below a
/// `List<Struct<...>>`, for example `DRF16BIT.TMANT`; the structural layout
/// accepts paths of any depth. The decoder still walks every BUFR descriptor
/// to retain the correct bit position; nested projection limits the Arrow
/// values retained across the Python boundary.
fn parse_projection_tree(
    paths: &[String],
    layout: FixedReplicationLayout,
) -> Result<ProjectionTree, String> {
    if paths.is_empty() {
        return Err("projection must contain at least one field".into());
    }
    let mut root = ProjectionTree::default();
    for path in paths {
        let parts: Vec<&str> = path.split('.').collect();
        let valid = parts.iter().all(|part| !part.is_empty());
        match layout {
            FixedReplicationLayout::Flat if !valid || parts.len() > 2 => {
                return Err(format!(
                    "Invalid projection path '{path}'; expected FIELD or FIELD.CHILD"
                ));
            }
            FixedReplicationLayout::Structural if !valid => {
                return Err(format!("Invalid structural projection path '{path}'"));
            }
            _ => {}
        }
        let mut node = &mut root;
        for part in parts {
            node = node.children.entry(part.to_string()).or_default();
        }
        node.whole_field = true;
    }
    Ok(root)
}

fn find_structural_item<'a>(
    schema: &'a [StructuralSchemaItem],
    parts: &[&str],
) -> Option<&'a StructuralSchemaItem> {
    let item = schema.iter().find(|item| item.name() == parts[0])?;
    if parts.len() == 1 {
        return Some(item);
    }
    match item {
        StructuralSchemaItem::Replication { children, .. } => {
            find_structural_item(children, &parts[1..])
        }
        StructuralSchemaItem::Scalar { .. } => None,
    }
}

fn validate_structural_paths(
    schema: &[StructuralSchemaItem],
    paths: &[String],
    option_name: &str,
) -> Result<(), String> {
    let missing: Vec<&String> = paths
        .iter()
        .filter(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            parts.is_empty()
                || parts.iter().any(|part| part.is_empty())
                || find_structural_item(schema, &parts).is_none()
        })
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("{option_name} fields not found: {missing:?}"))
    }
}

fn configure_structural_raw_codes(
    schema: &mut [StructuralSchemaItem],
    requested: &HashSet<String>,
) -> Result<(), String> {
    for item in schema {
        match item {
            StructuralSchemaItem::Scalar {
                descriptor,
                entry,
                path,
                raw_code,
            } => {
                if requested.contains(path) {
                    *raw_code = Some(RawCodeSpec::for_entry(path, *descriptor, entry)?);
                }
            }
            StructuralSchemaItem::Replication { children, .. } => {
                configure_structural_raw_codes(children, requested)?;
            }
        }
    }
    Ok(())
}

fn apply_structural_raw_code_paths(
    schema: &mut [StructuralSchemaItem],
    paths: Option<&[String]>,
) -> Result<(), String> {
    let Some(paths) = paths else {
        return Ok(());
    };
    if paths.is_empty() {
        return Err("raw_code_paths must contain at least one field".into());
    }
    validate_structural_paths(schema, paths, "Raw-code")?;
    let requested: HashSet<String> = paths.iter().cloned().collect();
    configure_structural_raw_codes(schema, &requested)
}

fn scalar_structural_metadata(
    descriptor: u32,
    entry: &TableBEntry,
    path: &str,
    raw_code: Option<&RawCodeSpec>,
) -> HashMap<String, String> {
    let mut metadata = HashMap::from([
        (
            "bufr.descriptor".to_string(),
            descriptor_to_string(descriptor),
        ),
        ("bufr.mnemonic".to_string(), entry.mnemonic.to_string()),
        ("bufr.scale".to_string(), entry.scale.to_string()),
        (
            "bufr.reference".to_string(),
            entry.reference_value.to_string(),
        ),
        ("bufr.bit_width".to_string(), entry.bit_width.to_string()),
        ("bufr.units".to_string(), entry.units.to_string()),
        ("bufr.structural_path".to_string(), path.to_string()),
        ("bufr.value_kind".to_string(), value_kind(entry).to_string()),
    ]);
    if let Some(raw_code) = raw_code {
        metadata.extend(raw_code.field_metadata());
    }
    metadata
}

fn structural_item_field(item: &StructuralSchemaItem, data_type: DataType) -> Field {
    match item {
        StructuralSchemaItem::Scalar {
            descriptor,
            entry,
            path,
            raw_code,
        } => Field::new(&*entry.mnemonic, data_type, true).with_metadata(
            scalar_structural_metadata(*descriptor, entry, path, raw_code.as_ref()),
        ),
        StructuralSchemaItem::Replication {
            name, path, kind, ..
        } => {
            let descriptor = match kind {
                StructuralReplication::Fixed { descriptor, .. } => *descriptor,
                StructuralReplication::Delayed {
                    factor_descriptor, ..
                } => *factor_descriptor,
            };
            let metadata = HashMap::from([
                ("bufr.structural_path".to_string(), path.clone()),
                (
                    "bufr.descriptor".to_string(),
                    descriptor_to_string(descriptor),
                ),
            ]);
            Field::new(&**name, data_type, true).with_metadata(metadata)
        }
    }
}

fn structural_child_selection<'a>(
    selection: Option<&'a ProjectionTree>,
    child_name: &str,
) -> Option<Option<&'a ProjectionTree>> {
    match selection {
        None => Some(None),
        Some(node) if node.whole_field => Some(None),
        Some(node) => node.children.get(child_name).map(Some),
    }
}

/// Projection paths of one `iter_batches` call, parsed once.
struct Projection {
    paths: Vec<String>,
    tree: ProjectionTree,
}

impl Projection {
    fn parse(paths: Vec<String>, layout: FixedReplicationLayout) -> Result<Self, String> {
        let tree = parse_projection_tree(&paths, layout)?;
        Ok(Self { paths, tree })
    }

    /// The top-level fields needed while decoding a projected flat batch.
    fn roots(&self) -> Vec<String> {
        let mut roots: Vec<String> = self.tree.children.keys().cloned().collect();
        roots.sort();
        roots.push(DATA_CATEGORY_COLUMN.to_string());
        roots
    }
}

/// Project selected children from a `List<Struct<...>>` without materializing
/// Python objects or changing list offsets.
fn project_list_struct_children(
    field: &Field,
    array: &ArrayRef,
    requested: &HashMap<String, ProjectionTree>,
) -> Result<(Field, ArrayRef, HashSet<String>), String> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| format!("Field '{}' is not a List array", field.name()))?;
    let values = list
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| format!("Field '{}' is not a List<Struct<...>>", field.name()))?;

    let struct_fields = match values.data_type() {
        DataType::Struct(fields) => fields,
        _ => {
            return Err(format!(
                "Field '{}' is not a List<Struct<...>>",
                field.name()
            ))
        }
    };

    let mut selected_fields: Vec<Field> = Vec::new();
    let mut selected_arrays: Vec<ArrayRef> = Vec::new();
    let mut matched: HashSet<String> = HashSet::new();
    for (index, child) in struct_fields.iter().enumerate() {
        if requested.contains_key(child.name()) {
            selected_fields.push(child.as_ref().clone());
            selected_arrays.push(values.column(index).clone());
            matched.insert(child.name().clone());
        }
    }

    if selected_fields.is_empty() {
        return Err(format!(
            "None of the requested children exist under '{}'",
            field.name()
        ));
    }

    let selected_fields = Fields::from(selected_fields);
    let selected_struct = StructArray::try_new(
        selected_fields.clone(),
        selected_arrays,
        values.nulls().cloned(),
    )
    .map_err(|error| format!("Failed to project '{}': {error}", field.name()))?;
    let item_field = Arc::new(Field::new("item", DataType::Struct(selected_fields), true));
    let selected_list = ListArray::try_new(
        Arc::clone(&item_field),
        list.offsets().clone(),
        Arc::new(selected_struct),
        list.nulls().cloned(),
    )
    .map_err(|error| format!("Failed to rebuild projected '{}': {error}", field.name()))?;

    Ok((
        Field::new(
            field.name(),
            DataType::List(item_field),
            field.is_nullable(),
        ),
        Arc::new(selected_list),
        matched,
    ))
}

/// Apply top-level and one-level nested projection to an Arrow batch.
fn project_record_batch(
    batch: RecordBatch,
    projection: Option<&Projection>,
) -> Result<RecordBatch, String> {
    let Some(projection) = projection else {
        return Ok(batch);
    };
    let mut matched_paths: HashSet<String> = HashSet::new();
    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
        let name = field.name();
        if APPENDED_COLUMNS.contains(&name.as_str()) {
            fields.push(field.as_ref().clone());
            arrays.push(array.clone());
            continue;
        }

        let Some(entry) = projection.tree.children.get(name) else {
            continue;
        };
        if entry.whole_field {
            fields.push(field.as_ref().clone());
            arrays.push(array.clone());
            matched_paths.insert(name.clone());
            for child in entry.children.keys() {
                matched_paths.insert(format!("{name}.{child}"));
            }
            continue;
        }

        let (projected_field, projected_array, matched_children) =
            project_list_struct_children(field, array, &entry.children)?;
        fields.push(projected_field);
        arrays.push(projected_array);
        for child in matched_children {
            matched_paths.insert(format!("{name}.{child}"));
        }
    }

    let missing: Vec<&String> = projection
        .paths
        .iter()
        .filter(|path| !matched_paths.contains(*path))
        .collect();
    if !missing.is_empty() {
        return Err(format!("Projection fields not found: {missing:?}"));
    }

    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        batch.schema().metadata().clone(),
    ));
    RecordBatch::try_new(schema, arrays).map_err(|error| format!("Arrow projection error: {error}"))
}

#[derive(Debug, Clone)]
struct ListValueFilterSpec {
    path: String,
    key: String,
    values: HashSet<u64>,
}

fn numeric_filter_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn parse_list_value_filters(
    filters: Vec<(String, String, Vec<f64>)>,
) -> Result<Vec<ListValueFilterSpec>, String> {
    let mut paths = HashSet::new();
    filters
        .into_iter()
        .map(|(path, key, values)| {
            if path.is_empty() || path.contains('.') || key.is_empty() || key.contains('.') {
                return Err(format!(
                    "Invalid list filter '{path}.{key}'; path and key must be single field names"
                ));
            }
            if !paths.insert(path.clone()) {
                return Err(format!(
                    "Multiple list filters for '{path}' are not supported; combine values into one filter"
                ));
            }
            if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "List filter '{path}.{key}' requires at least one finite numeric value"
                ));
            }
            Ok(ListValueFilterSpec {
                path,
                key,
                values: values.into_iter().map(numeric_filter_bits).collect(),
            })
        })
        .collect()
}

/// How the stored values of a filter key give the physical values a filter
/// compares: as decoded, or as raw codes with this packing.
#[derive(Clone, Copy)]
enum KeyValues {
    Physical,
    RawCode { reference: i64, divisor: f64 },
}

/// Index of the filter key among the struct children of a `List<Struct>`
/// field; the key must be a Float64 child or a raw-code Int32 child.
fn list_filter_key_index(
    field: &Field,
    filter: &ListValueFilterSpec,
) -> Result<(usize, KeyValues), String> {
    let DataType::List(item) = field.data_type() else {
        return Err(format!("Filter path '{}' is not a List array", filter.path));
    };
    let DataType::Struct(children) = item.data_type() else {
        return Err(format!(
            "Filter path '{}' is not a List<Struct<...>>",
            filter.path
        ));
    };
    let key = children
        .iter()
        .position(|child| *child.name() == filter.key)
        .ok_or_else(|| format!("Filter key '{}.{}' was not found", filter.path, filter.key))?;
    let not_numeric = || format!("Filter key '{}.{}' is not numeric", filter.path, filter.key);
    match children[key].data_type() {
        DataType::Float64 => Ok((key, KeyValues::Physical)),
        DataType::Int32 => {
            let metadata = children[key].metadata();
            let packing = |name: &str| metadata.get(name).ok_or_else(not_numeric);
            let reference = packing("reference")?.parse().map_err(|_| not_numeric())?;
            let scale: i32 = packing("scale")?.parse().map_err(|_| not_numeric())?;
            Ok((
                key,
                KeyValues::RawCode {
                    reference,
                    divisor: 10f64.powi(scale),
                },
            ))
        }
        _ => Err(not_numeric()),
    }
}

/// Filter entries of one `List<Struct>` while preserving outer row identity.
fn filter_list_struct_values(
    field: &Field,
    array: &ArrayRef,
    filter: &ListValueFilterSpec,
) -> Result<ArrayRef, String> {
    let (key_index, key_values) = list_filter_key_index(field, filter)?;
    let list = array.as_list::<i32>();
    let keys = list.values().as_struct().column(key_index);
    let physical = |index: usize| -> f64 {
        match key_values {
            KeyValues::Physical => keys.as_primitive::<Float64Type>().value(index),
            KeyValues::RawCode { reference, divisor } => {
                let code = keys.as_primitive::<Int32Type>().value(index);
                (reference + i64::from(code)) as f64 / divisor
            }
        }
    };

    let mut selected = Vec::new();
    let mut offsets = Vec::with_capacity(list.len() + 1);
    offsets.push(0i32);
    for row in 0..list.len() {
        if list.is_valid(row) {
            let start = list.offsets()[row] as usize;
            let end = list.offsets()[row + 1] as usize;
            for index in start..end {
                if keys.is_valid(index)
                    && filter
                        .values
                        .contains(&numeric_filter_bits(physical(index)))
                {
                    selected.push(index as u64);
                }
            }
        }
        offsets.push(selected.len() as i32);
    }

    let indices = UInt64Array::from(selected);
    let selected_values = take(list.values().as_ref(), &indices, None)
        .map_err(|error| format!("Failed to filter '{}': {error}", filter.path))?;
    let item_field = match field.data_type() {
        DataType::List(item) => Arc::clone(item),
        _ => return Err(format!("Filter path '{}' is not a List", filter.path)),
    };
    let offset_buffer = OffsetBuffer::new(ScalarBuffer::from(offsets));
    let filtered = ListArray::try_new(
        item_field,
        offset_buffer,
        selected_values,
        list.nulls().cloned(),
    )
    .map_err(|error| format!("Failed to rebuild filtered '{}': {error}", filter.path))?;
    Ok(Arc::new(filtered))
}

fn apply_list_value_filters(
    batch: RecordBatch,
    filters: Option<&[ListValueFilterSpec]>,
) -> Result<RecordBatch, String> {
    let Some(filters) = filters else {
        return Ok(batch);
    };
    let mut arrays = batch.columns().to_vec();
    for filter in filters {
        let index = batch
            .schema()
            .index_of(&filter.path)
            .map_err(|_| format!("List filter path '{}' was not decoded", filter.path))?;
        arrays[index] =
            filter_list_struct_values(batch.schema().field(index), &arrays[index], filter)?;
    }
    RecordBatch::try_new(batch.schema(), arrays)
        .map_err(|error| format!("Arrow list filter error: {error}"))
}

/// Append deterministic source identity to a decoded batch.
fn append_source_provenance(
    batch: RecordBatch,
    message_indices: &[usize],
    subset_counts: &[u16],
) -> Result<RecordBatch, String> {
    if message_indices.len() != subset_counts.len() {
        return Err("Message provenance length mismatch".into());
    }
    if batch
        .schema()
        .fields()
        .iter()
        .any(|field| PROVENANCE_COLUMNS.contains(&field.name().as_str()))
    {
        return Err("Decoded schema collides with source provenance fields".into());
    }

    let mut source_messages: Vec<u64> = Vec::with_capacity(batch.num_rows());
    let mut source_subsets: Vec<u32> = Vec::with_capacity(batch.num_rows());
    for (message_index, subset_count) in message_indices.iter().zip(subset_counts) {
        source_messages.extend(std::iter::repeat_n(
            *message_index as u64,
            *subset_count as usize,
        ));
        source_subsets.extend(0..u32::from(*subset_count));
    }
    if source_messages.len() != batch.num_rows() {
        return Err(format!(
            "Decoded {} rows but provenance describes {} rows",
            batch.num_rows(),
            source_messages.len()
        ));
    }

    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    fields.push(Field::new(SOURCE_MESSAGE_COLUMN, DataType::UInt64, false));
    fields.push(Field::new(SOURCE_SUBSET_COLUMN, DataType::UInt32, false));

    let mut arrays = batch.columns().to_vec();
    arrays.push(Arc::new(UInt64Array::from(source_messages)));
    arrays.push(Arc::new(UInt32Array::from(source_subsets)));

    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        batch.schema().metadata().clone(),
    ));
    RecordBatch::try_new(schema, arrays).map_err(|error| format!("Arrow provenance error: {error}"))
}

fn append_data_category(expanded: &mut Vec<ExpandedItem>) {
    expanded.push(ExpandedItem::Descriptor(
        0,
        Arc::new(TableBEntry {
            mnemonic: Arc::from(DATA_CATEGORY_COLUMN),
            name: "Data category".to_string(),
            units: Arc::from("NUMERIC"),
            scale: 0,
            reference_value: 0,
            bit_width: 8,
            is_string: false,
            is_code_or_flag: false,
        }),
    ));
}

const DATA_CATEGORY_COLUMN: &str = "_data_category";
const SOURCE_MESSAGE_COLUMN: &str = "_source_message_index";
const SOURCE_SUBSET_COLUMN: &str = "_source_subset_index";
/// Columns that identify each row's source message and subset.
const PROVENANCE_COLUMNS: [&str; 2] = [SOURCE_MESSAGE_COLUMN, SOURCE_SUBSET_COLUMN];
/// Columns every decoded batch carries after the template's fields.
const APPENDED_COLUMNS: [&str; 3] = [
    DATA_CATEGORY_COLUMN,
    SOURCE_MESSAGE_COLUMN,
    SOURCE_SUBSET_COLUMN,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixedReplicationLayout {
    Flat,
    Structural,
}

impl FixedReplicationLayout {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "flat" => Ok(Self::Flat),
            "structural" => Ok(Self::Structural),
            _ => Err(format!(
                "fixed_replication must be 'flat' or 'structural', got '{value}'"
            )),
        }
    }
}

#[derive(Clone, Copy)]
struct BoundedDecodeOptions<'a> {
    projection: Option<&'a Projection>,
    raw_code_paths: Option<&'a [String]>,
    list_value_filters: Option<&'a [ListValueFilterSpec]>,
    policy: ReadPolicy,
    fixed_replication: FixedReplicationLayout,
    projection_missing: ProjectionMissing,
}

/// What a decode does with projected fields a template does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionMissing {
    Raise,
    Skip,
}

impl ProjectionMissing {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "raise" => Ok(Self::Raise),
            "skip" => Ok(Self::Skip),
            other => Err(format!(
                "projection_missing must be 'raise' or 'skip', got '{other}'"
            )),
        }
    }
}

/// The projection, raw-code paths and list filters of one decode that name
/// fields its template has.
struct PresentFields {
    projection: Option<Projection>,
    raw_code_paths: Option<Vec<String>>,
    list_value_filters: Option<Vec<ListValueFilterSpec>>,
}

impl<'a> BoundedDecodeOptions<'a> {
    fn present_fields(&self, present: impl Fn(&str) -> bool) -> Result<PresentFields, String> {
        // A projected filtered list counts as present only if it also has the
        // filter key; without a projection a missing key still raises.
        let present = |path: &str| {
            let root = path.split('.').next().unwrap_or(path);
            present(path)
                && (self.projection.is_none()
                    || self
                        .list_value_filters
                        .into_iter()
                        .flatten()
                        .filter(|filter| filter.path == root)
                        .all(|filter| present(&format!("{}.{}", filter.path, filter.key))))
        };
        let projection = self
            .projection
            .map(|projection| {
                let paths: Vec<String> = projection
                    .paths
                    .iter()
                    .filter(|path| present(path))
                    .cloned()
                    .collect();
                let tree = if paths.is_empty() {
                    ProjectionTree::default()
                } else {
                    parse_projection_tree(&paths, self.fixed_replication)?
                };
                Ok::<_, String>(Projection { paths, tree })
            })
            .transpose()?;
        let raw_code_paths = self
            .raw_code_paths
            .map(|paths| -> Vec<String> {
                paths.iter().filter(|path| present(path)).cloned().collect()
            })
            .filter(|paths| !paths.is_empty());
        let list_value_filters = self
            .list_value_filters
            .map(|filters| -> Vec<ListValueFilterSpec> {
                filters
                    .iter()
                    .filter(|filter| present(&filter.path))
                    .cloned()
                    .collect()
            })
            .filter(|filters| !filters.is_empty());
        Ok(PresentFields {
            projection,
            raw_code_paths,
            list_value_filters,
        })
    }

    fn with_fields(self, fields: &'a PresentFields) -> Self {
        Self {
            projection: fields.projection.as_ref(),
            raw_code_paths: fields.raw_code_paths.as_deref(),
            list_value_filters: fields.list_value_filters.as_deref(),
            ..self
        }
    }
}

/// Decode options shared by `iter_batches` and `read_bufr`, validated and
/// parsed once per call.
struct ParsedDecodeOptions {
    projection: Option<Projection>,
    raw_code_paths: Option<Vec<String>>,
    list_value_filters: Option<Vec<ListValueFilterSpec>>,
    fixed_replication: FixedReplicationLayout,
    skip_truncated_messages: bool,
    projection_missing: ProjectionMissing,
}

impl ParsedDecodeOptions {
    fn parse(
        projection: Option<Vec<String>>,
        raw_code_paths: Option<Vec<String>>,
        list_value_filters: Option<Vec<(String, String, Vec<f64>)>>,
        fixed_replication: &str,
        skip_truncated_messages: bool,
        projection_missing: &str,
    ) -> Result<Self, String> {
        let fixed_replication = FixedReplicationLayout::parse(fixed_replication)?;
        let projection_missing = ProjectionMissing::parse(projection_missing)?;
        let projection = projection
            .map(|paths| Projection::parse(paths, fixed_replication))
            .transpose()?;
        if let Some(paths) = raw_code_paths.as_deref() {
            let raw_tree = parse_projection_tree(paths, fixed_replication)?;
            match fixed_replication {
                FixedReplicationLayout::Flat => {
                    if let Some(projection) = &projection {
                        for (root, raw_entry) in &raw_tree.children {
                            let covered =
                                projection
                                    .tree
                                    .children
                                    .get(root)
                                    .is_some_and(|projected_entry| {
                                        projected_entry.whole_field
                                            || (!raw_entry.whole_field
                                                && raw_entry.children.keys().all(|child| {
                                                    projected_entry.children.contains_key(child)
                                                }))
                                    });
                            if !covered {
                                return Err(format!(
                                    "Raw-code path under '{root}' must also appear in projection"
                                ));
                            }
                        }
                    }
                }
                FixedReplicationLayout::Structural => {
                    if let Some(projection) = &projection {
                        for raw_path in paths {
                            let covered = projection.paths.iter().any(|projected_path| {
                                raw_path == projected_path
                                    || raw_path.starts_with(&format!("{projected_path}."))
                            });
                            if !covered {
                                return Err(format!(
                                    "Raw-code path '{raw_path}' must also appear in projection"
                                ));
                            }
                        }
                    }
                }
            }
        }
        let list_value_filters = list_value_filters
            .map(parse_list_value_filters)
            .transpose()?;
        if fixed_replication == FixedReplicationLayout::Structural && list_value_filters.is_some() {
            return Err(
                "list filters are not supported with fixed_replication='structural'".into(),
            );
        }
        if let (Some(projection), Some(filters)) = (&projection, list_value_filters.as_deref()) {
            for filter in filters {
                if !projection.paths.iter().any(|path| {
                    path == &filter.path || path.starts_with(&format!("{}.", filter.path))
                }) {
                    return Err(format!(
                        "List filter path '{}' must also appear in projection",
                        filter.path
                    ));
                }
            }
        }
        Ok(Self {
            projection,
            raw_code_paths,
            list_value_filters,
            fixed_replication,
            skip_truncated_messages,
            projection_missing,
        })
    }

    fn bounded(&self, strict: bool) -> BoundedDecodeOptions<'_> {
        BoundedDecodeOptions {
            projection: self.projection.as_ref(),
            raw_code_paths: self.raw_code_paths.as_deref(),
            list_value_filters: self.list_value_filters.as_deref(),
            // Quarantine needs overruns to surface as errors, which permissive
            // decoding otherwise turns into missing values.
            policy: ReadPolicy {
                strict,
                reject_overrun: strict || self.skip_truncated_messages,
            },
            fixed_replication: self.fixed_replication,
            projection_missing: self.projection_missing,
        }
    }
}

#[derive(Debug, Serialize)]
struct DecodeFailure {
    source_message_index: usize,
    reason: String,
}

/// Decode shape of one compressed-template item. `id` indexes the
/// per-message occurrence counts; a List node also counts its struct items
/// under `items_id`, and its descendants take the ids before `end_id`. A Skip
/// node is an unprojected item whose bits are stepped over. `has_list` marks
/// a node that is or contains a decoded List, whose length only the message
/// data gives.
struct ColumnNode {
    id: usize,
    kind: ColumnKind,
    children: Vec<ColumnNode>,
    has_list: bool,
}

enum ColumnKind {
    Skip,
    Float64,
    Int32,
    Utf8,
    FixedList(usize),
    List {
        items_id: usize,
        end_id: usize,
        filter: Option<KeyFilter>,
    },
}

/// Keeps the list entries whose child `key` holds one of `values`, compared
/// as `numeric_filter_bits` of the physical value.
struct KeyFilter {
    key: usize,
    values: HashSet<u64>,
}

impl ColumnNode {
    /// The node of `field` with its output field, or a Skip node and `None`
    /// when `selection` leaves the field out.
    fn new(
        field: &Field,
        selection: Option<Option<&ProjectionTree>>,
        next_id: &mut usize,
    ) -> Result<(Self, Option<Field>), String> {
        let id = *next_id;
        *next_id += 1;
        let node = |kind, children: Vec<Self>| Self {
            has_list: matches!(kind, ColumnKind::List { .. })
                || children.iter().any(|child| child.has_list),
            id,
            kind,
            children,
        };
        let Some(selection) = selection else {
            return Ok((node(ColumnKind::Skip, Vec::new()), None));
        };
        let leaf = |kind| Ok((node(kind, Vec::new()), Some(field.clone())));
        match field.data_type() {
            DataType::Float64 => leaf(ColumnKind::Float64),
            DataType::Int32 => leaf(ColumnKind::Int32),
            DataType::Utf8 => leaf(ColumnKind::Utf8),
            DataType::FixedSizeList(item, size) => {
                let count = usize::try_from(*size).map_err(|_| BATCH_LAYOUT_MISMATCH)?;
                let (children, item) = Self::struct_children(item, selection, next_id)?;
                Ok((
                    node(ColumnKind::FixedList(count), children),
                    Some(
                        field
                            .clone()
                            .with_data_type(DataType::FixedSizeList(item, *size)),
                    ),
                ))
            }
            DataType::List(item) => {
                let items_id = *next_id;
                *next_id += 1;
                let (children, item) = Self::struct_children(item, selection, next_id)?;
                let kind = ColumnKind::List {
                    items_id,
                    end_id: *next_id,
                    filter: None,
                };
                Ok((
                    node(kind, children),
                    Some(field.clone().with_data_type(DataType::List(item))),
                ))
            }
            other => Err(format!(
                "Compressed decoding does not write {other} field '{}'",
                field.name()
            )),
        }
    }

    /// Nodes of every struct child, and the item field of the selected ones.
    fn struct_children(
        item: &FieldRef,
        selection: Option<&ProjectionTree>,
        next_id: &mut usize,
    ) -> Result<(Vec<Self>, FieldRef), String> {
        let DataType::Struct(fields) = item.data_type() else {
            return Err(BATCH_LAYOUT_MISMATCH.into());
        };
        let mut children = Vec::with_capacity(fields.len());
        let mut selected = Vec::with_capacity(fields.len());
        for field in fields {
            let (child, output) = Self::new(
                field,
                structural_child_selection(selection, field.name()),
                next_id,
            )?;
            children.push(child);
            selected.extend(output);
        }
        let item = item
            .as_ref()
            .clone()
            .with_data_type(DataType::Struct(Fields::from(selected)));
        Ok((children, Arc::new(item)))
    }
}

const BATCH_LAYOUT_MISMATCH: &str = "Batch column layout does not match the expanded template";

fn is_zero_width(item: &ExpandedItem) -> bool {
    matches!(item, ExpandedItem::Descriptor(_, entry) if entry.bit_width == 0)
}

#[inline]
fn physical_value(entry: &TableBEntry, divisor: f64, code: u64) -> f64 {
    (entry.reference_value + code as i64) as f64 / divisor
}

/// Step over one compressed item without storing it. Strict mode still
/// validates IA5 references and rejects IA5 increments.
fn skip_compressed_item(
    reader: &mut BitReader,
    item: &ExpandedItem,
    n: usize,
) -> Result<(), String> {
    match item {
        ExpandedItem::Descriptor(_, entry) if entry.bit_width == 0 => {}
        ExpandedItem::Descriptor(_, entry) if entry.is_string && reader.strict => {
            read_compressed_strings(reader, entry, n, |_, _| {})?
        }
        ExpandedItem::Descriptor(_, entry) if entry.is_string => {
            skip_compressed_strings(reader, entry.bit_width / 8, n)
        }
        ExpandedItem::Descriptor(_, entry) => skip_compressed_values(reader, entry.bit_width, n),
        ExpandedItem::FixedReplication { count, group, .. } => {
            for _ in 0..*count {
                for item in group {
                    skip_compressed_item(reader, item, n)?;
                }
            }
        }
        ExpandedItem::DelayedReplication {
            factor_entry,
            group,
            ..
        } => {
            let count = read_compressed_replication_count(reader, factor_entry.bit_width, n);
            for _ in 0..count {
                for item in group {
                    skip_compressed_item(reader, item, n)?;
                }
            }
        }
    }
    Ok(())
}

/// Add `times` occurrences of a node without a decoded list, and the
/// occurrences of its descendants that follow from the template.
fn count_template_occurrences(
    node: &ColumnNode,
    times: usize,
    base: usize,
    occurrences: &mut [usize],
) {
    occurrences[node.id - base] += times;
    let per_occurrence = match node.kind {
        ColumnKind::FixedList(size) => size,
        _ => 1,
    };
    for child in &node.children {
        count_template_occurrences(child, times * per_occurrence, base, occurrences);
    }
}

/// Count every node's occurrences per subset of one compressed message,
/// reading only replication counts and list filter keys. Subsets of a
/// compressed message share each replication count, so `occurrences[id - base]`
/// holds for every subset; a filtered list's items and their descendants
/// are counted per subset in `filtered`. Nodes without a decoded list are
/// counted from the template and their bits stepped over; after the last
/// node with a decoded list the reader stops unless `read_all` asks for it
/// to end past `expanded`.
#[allow(clippy::too_many_arguments)]
fn count_compressed_occurrences(
    reader: &mut BitReader,
    expanded: &[ExpandedItem],
    nodes: &[ColumnNode],
    n: usize,
    base: usize,
    occurrences: &mut [usize],
    filtered: &mut Vec<FilteredLayout>,
    read_all: bool,
) -> Result<(), String> {
    let last_list = nodes.iter().rposition(|node| node.has_list);
    let read_past = |position: usize| read_all || last_list.is_some_and(|last| position < last);
    let mut nodes = nodes.iter().enumerate();
    for item in expanded {
        if is_zero_width(item) {
            continue;
        }
        let (position, node) = nodes.next().ok_or(BATCH_LAYOUT_MISMATCH)?;
        if !node.has_list {
            count_template_occurrences(node, 1, base, occurrences);
            if read_past(position) {
                skip_compressed_item(reader, item, n)?;
            }
            continue;
        }
        occurrences[node.id - base] += 1;
        match (item, &node.kind) {
            (ExpandedItem::FixedReplication { count, group, .. }, ColumnKind::FixedList(size))
                if count == size =>
            {
                for _ in 0..*count {
                    count_compressed_occurrences(
                        reader,
                        group,
                        &node.children,
                        n,
                        base,
                        occurrences,
                        filtered,
                        true,
                    )?;
                }
            }
            (
                ExpandedItem::DelayedReplication {
                    factor_entry,
                    group,
                    ..
                },
                ColumnKind::List {
                    items_id,
                    end_id,
                    filter,
                },
            ) => {
                let count = read_compressed_replication_count(reader, factor_entry.bit_width, n);
                if let Some(filter) = filter {
                    filtered.push(count_filtered_list(
                        reader,
                        group,
                        &node.children,
                        (*items_id, *end_id),
                        filter,
                        count,
                        n,
                    )?);
                    continue;
                }
                occurrences[items_id - base] += count;
                if node.children.iter().any(|child| child.has_list) {
                    for _ in 0..count {
                        count_compressed_occurrences(
                            reader,
                            group,
                            &node.children,
                            n,
                            base,
                            occurrences,
                            filtered,
                            true,
                        )?;
                    }
                    continue;
                }
                for child in &node.children {
                    count_template_occurrences(child, count, base, occurrences);
                }
                if read_past(position) {
                    for _ in 0..count {
                        for item in group {
                            skip_compressed_item(reader, item, n)?;
                        }
                    }
                }
            }
            _ => return Err(BATCH_LAYOUT_MISMATCH.into()),
        }
    }
    if nodes.next().is_some() {
        return Err(BATCH_LAYOUT_MISMATCH.into());
    }
    Ok(())
}

/// Size one filtered list of a compressed message: each element's key
/// decides which subsets keep it, and the nodes of a kept element count
/// toward that subset only.
fn count_filtered_list(
    reader: &mut BitReader,
    group: &[ExpandedItem],
    children: &[ColumnNode],
    (items_id, end_id): (usize, usize),
    filter: &KeyFilter,
    count: usize,
    n: usize,
) -> Result<FilteredLayout, String> {
    let width = end_id - items_id;
    let mut layout = FilteredLayout {
        first_id: items_id,
        counts: vec![0; width * n],
        kept: vec![false; count * n],
    };
    let mut element = vec![0; width];
    let mut nested = Vec::new();
    for kept in layout.kept.chunks_mut(n) {
        element.fill(0);
        element[0] = 1;
        let mut nodes = children.iter().enumerate();
        for item in group {
            if is_zero_width(item) {
                continue;
            }
            let (index, node) = nodes.next().ok_or(BATCH_LAYOUT_MISMATCH)?;
            match item {
                ExpandedItem::Descriptor(descriptor, entry) if index == filter.key => {
                    element[node.id - items_id] += 1;
                    let divisor = 10f64.powi(entry.scale);
                    let mut last: Option<(Option<u64>, bool)> = None;
                    for_each_compressed_code(
                        reader,
                        entry.bit_width,
                        uses_missing_value(*descriptor),
                        n,
                        |subset, code| {
                            kept[subset] = match last {
                                Some((last_code, keep)) if last_code == code => keep,
                                _ => {
                                    let keep = code.is_some_and(|code| {
                                        filter.values.contains(&numeric_filter_bits(
                                            physical_value(entry, divisor, code),
                                        ))
                                    });
                                    last = Some((code, keep));
                                    keep
                                }
                            };
                        },
                    );
                }
                _ => count_compressed_occurrences(
                    reader,
                    std::slice::from_ref(item),
                    std::slice::from_ref(node),
                    n,
                    items_id,
                    &mut element,
                    &mut nested,
                    true,
                )?,
            }
        }
        if nodes.next().is_some() {
            return Err(BATCH_LAYOUT_MISMATCH.into());
        }
        for subset in (0..n).filter(|subset| kept[*subset]) {
            for (offset, occurrences) in element.iter().enumerate() {
                layout.counts[offset * n + subset] += occurrences;
            }
        }
    }
    Ok(layout)
}

struct MessageLayout {
    rows: usize,
    occurrences: Vec<usize>,
    filtered: Vec<FilteredLayout>,
}

/// Per-subset counts of the nodes under one filtered list of a message: the
/// list's items and their descendants, ids from `first_id`.
struct FilteredLayout {
    first_id: usize,
    /// Count of node `first_id + k` in subset `s`, at `k * rows + s`.
    counts: Vec<usize>,
    /// Whether subset `s` keeps element `e`, at `e * rows + s`.
    kept: Vec<bool>,
}

impl MessageLayout {
    fn subset_counts(&self, id: usize) -> Option<&[usize]> {
        self.filtered.iter().find_map(|filtered| {
            let offset = id.checked_sub(filtered.first_id)?;
            filtered
                .counts
                .get(offset * self.rows..(offset + 1) * self.rows)
        })
    }

    fn length(&self, id: usize) -> usize {
        match self.subset_counts(id) {
            Some(counts) => counts.iter().sum(),
            None => self.rows * self.occurrences[id],
        }
    }

    fn positions(&self, id: usize) -> Positions {
        match self.subset_counts(id) {
            Some(counts) => {
                let mut next = prefix_sums(counts.iter().copied());
                next.pop();
                Positions::Kept { next }
            }
            None => Positions::Strided {
                occurrences: self.occurrences[id],
                next: 0,
            },
        }
    }

    fn kept_elements(&self, items_id: usize) -> Option<&[bool]> {
        self.filtered
            .iter()
            .find(|filtered| filtered.first_id == items_id)
            .map(|filtered| filtered.kept.as_slice())
    }
}

/// Batch-wide column storage.
enum BatchColumn {
    Float64(LeafStorage<f64>),
    Int32(LeafStorage<i32>),
    Utf8 {
        values: Vec<Option<String>>,
    },
    FixedList {
        items: usize,
        children: Vec<BatchColumn>,
    },
    List {
        offsets: Vec<i32>,
        children: Vec<BatchColumn>,
    },
}

/// Values and validity bits of one numeric column, split into one window
/// per message. The values start uninitialised and decoding writes every
/// slot once, through the `LeafWindow` of the message that holds it.
///
/// A window allocates its own validity bits at its first missing value, so
/// parallel windows share no byte and a window without missing values
/// allocates none. Finishing the column packs the windows' bits together.
struct LeafStorage<T> {
    values: Box<[MaybeUninit<T>]>,
    valid: Vec<Option<Vec<u8>>>,
    lengths: Vec<usize>,
}

impl<T: Default> LeafStorage<T> {
    fn new(lengths: Vec<usize>) -> Self {
        Self {
            values: Box::new_uninit_slice(lengths.iter().sum()),
            valid: vec![None; lengths.len()],
            lengths,
        }
    }

    fn windows(&mut self) -> Vec<LeafWindow<'_, T>> {
        split_by_lengths(&mut self.values, self.lengths.iter().copied())
            .into_iter()
            .zip(self.valid.iter_mut())
            .map(|(values, valid)| LeafWindow::new(values, valid))
            .collect()
    }

    /// The written values and their validity, `None` without missing values.
    ///
    /// # Safety
    ///
    /// Every slot was written through a `LeafWindow`. Columns are finished
    /// only after every message of the batch decoded into its windows
    /// without error, and debug builds assert that each window is filled.
    unsafe fn assume_written(self) -> (Vec<T>, Option<NullBuffer>) {
        let nulls = window_nulls(&self.valid, &self.lengths);
        (self.values.assume_init().into_vec(), nulls)
    }
}

/// Packs the windows' validity bits into one bitmap; a window without bits
/// is all valid.
fn window_nulls(valid: &[Option<Vec<u8>>], lengths: &[usize]) -> Option<NullBuffer> {
    if valid.iter().all(Option::is_none) {
        return None;
    }
    let starts = prefix_sums(lengths.iter().copied());
    let sources: Vec<Option<(&[u8], usize)>> = valid
        .iter()
        .map(|bits| bits.as_deref().map(|bits| (bits, 0)))
        .collect();
    let length = starts[lengths.len()];
    Some(NullBuffer::new(BooleanBuffer::new(
        concat_bits(&sources, &starts),
        0,
        length,
    )))
}

/// One message's disjoint window into a `LeafStorage`.
struct LeafWindow<'a, T> {
    values: &'a mut [MaybeUninit<T>],
    valid: &'a mut Option<Vec<u8>>,
    #[cfg(debug_assertions)]
    written: Vec<bool>,
}

impl<'a, T: Default> LeafWindow<'a, T> {
    fn new(values: &'a mut [MaybeUninit<T>], valid: &'a mut Option<Vec<u8>>) -> Self {
        Self {
            #[cfg(debug_assertions)]
            written: vec![false; values.len()],
            values,
            valid,
        }
    }

    /// Store `value` at `index`; a missing value is stored as the default
    /// under a cleared validity bit.
    #[inline(always)]
    fn write(&mut self, index: usize, value: Option<T>) {
        let value = value.unwrap_or_else(|| {
            match self.valid {
                Some(valid) => valid[index / 8] &= !(1 << (index % 8)),
                None => self.first_missing(index),
            }
            T::default()
        });
        self.values[index].write(value);
        #[cfg(debug_assertions)]
        {
            self.written[index] = true;
        }
    }

    #[cold]
    #[inline(never)]
    fn first_missing(&mut self, index: usize) {
        let mut valid = vec![u8::MAX; self.values.len().div_ceil(8)];
        valid[index / 8] &= !(1 << (index % 8));
        *self.valid = Some(valid);
    }

    #[cfg(debug_assertions)]
    fn is_filled(&self) -> bool {
        self.written.iter().all(|written| *written)
    }
}

#[cfg(debug_assertions)]
fn slots_filled(slots: &[ColumnSlot<'_>]) -> bool {
    slots.iter().all(|slot| match slot {
        ColumnSlot::Float64 { window, .. } => window.is_filled(),
        ColumnSlot::Int32 { window, .. } => window.is_filled(),
        ColumnSlot::FixedList { children } => slots_filled(children),
        ColumnSlot::List(list) => slots_filled(&list.children),
        ColumnSlot::Skip | ColumnSlot::Utf8 { .. } => true,
    })
}

fn allocate_batch_columns(
    nodes: &[ColumnNode],
    layouts: &[MessageLayout],
) -> Result<Vec<BatchColumn>, String> {
    nodes
        .iter()
        .filter(|node| !matches!(node.kind, ColumnKind::Skip))
        .map(|node| allocate_batch_column(node, layouts))
        .collect()
}

fn allocate_batch_column(
    node: &ColumnNode,
    layouts: &[MessageLayout],
) -> Result<BatchColumn, String> {
    let lengths =
        |id: usize| -> Vec<usize> { layouts.iter().map(|layout| layout.length(id)).collect() };
    let total = |id: usize| -> usize { lengths(id).iter().sum() };
    let length = total(node.id);
    Ok(match node.kind {
        ColumnKind::Skip => unreachable!("skipped items have no column"),
        ColumnKind::Float64 => BatchColumn::Float64(LeafStorage::new(lengths(node.id))),
        ColumnKind::Int32 => BatchColumn::Int32(LeafStorage::new(lengths(node.id))),
        ColumnKind::Utf8 => BatchColumn::Utf8 {
            values: vec![None; length],
        },
        ColumnKind::FixedList(size) => BatchColumn::FixedList {
            items: length * size,
            children: allocate_batch_columns(&node.children, layouts)?,
        },
        ColumnKind::List { items_id, .. } => {
            if i32::try_from(total(items_id)).is_err() {
                return Err("Delayed replication items exceed Arrow i32 list offsets".into());
            }
            BatchColumn::List {
                offsets: vec![0; length + 1],
                children: allocate_batch_columns(&node.children, layouts)?,
            }
        }
    })
}

/// Where one message writes the values of one node.
enum Positions {
    /// Value `k` of subset `s` is at `s * occurrences + k`.
    Strided { occurrences: usize, next: usize },
    /// The next value of subset `s` is at `next[s]`. Under a filtered list
    /// only the subsets that keep the current element advance.
    Kept { next: Vec<usize> },
}

/// The indices of one occurrence of a node.
enum Placement<'p> {
    Strided {
        stride: usize,
        occurrence: usize,
    },
    Kept {
        next: &'p mut [usize],
        kept: &'p [bool],
    },
}

impl Positions {
    #[inline(always)]
    fn place<'p>(&'p mut self, kept: Option<&'p [bool]>) -> Result<Placement<'p>, String> {
        match (self, kept) {
            (Positions::Strided { occurrences, next }, None) => Ok(Placement::Strided {
                stride: *occurrences,
                occurrence: take_occurrence(next, *occurrences)?,
            }),
            (Positions::Kept { next }, Some(kept)) => Ok(Placement::Kept { next, kept }),
            _ => Err(BATCH_LAYOUT_MISMATCH.into()),
        }
    }
}

impl Placement<'_> {
    /// Index of `subset`'s value, `None` for a subset that drops it.
    #[inline(always)]
    fn index(&mut self, subset: usize) -> Option<usize> {
        match self {
            Placement::Strided { stride, occurrence } => Some(subset * *stride + *occurrence),
            Placement::Kept { next, kept } => kept[subset].then(|| {
                next[subset] += 1;
                next[subset] - 1
            }),
        }
    }
}

/// One message's disjoint window into a `BatchColumn`.
enum ColumnSlot<'a> {
    Skip,
    Float64 {
        window: LeafWindow<'a, f64>,
        positions: Positions,
    },
    Int32 {
        window: LeafWindow<'a, i32>,
        positions: Positions,
    },
    Utf8 {
        values: &'a mut [Option<String>],
        positions: Positions,
    },
    FixedList {
        children: Vec<ColumnSlot<'a>>,
    },
    List(Box<ListSlot<'a>>),
}

/// `ends` are the list offsets after each entry; `items_base` is the
/// batch-wide index of this message's first struct item and `items` the
/// positions of the struct items within the message. A filtered list's
/// `kept` flags are element-major.
struct ListSlot<'a> {
    ends: &'a mut [i32],
    entries: Positions,
    items_base: usize,
    items: Positions,
    kept: Option<&'a [bool]>,
    children: Vec<ColumnSlot<'a>>,
}

fn split_by_lengths<T>(mut slice: &mut [T], lengths: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    lengths
        .map(|length| {
            let (head, tail) = std::mem::take(&mut slice).split_at_mut(length);
            slice = tail;
            head
        })
        .collect()
}

fn take_occurrence(next: &mut usize, occurrences: usize) -> Result<usize, String> {
    if *next >= occurrences {
        return Err(BATCH_LAYOUT_MISMATCH.into());
    }
    *next += 1;
    Ok(*next - 1)
}

/// Split columns into per-message slots: `result[message][node]`. Each
/// column is split on the thread pool.
fn message_slots<'a>(
    columns: &'a mut [BatchColumn],
    nodes: &[ColumnNode],
    layouts: &'a [MessageLayout],
) -> Vec<Vec<ColumnSlot<'a>>> {
    let decoded_nodes: Vec<&ColumnNode> = nodes
        .iter()
        .filter(|node| !matches!(node.kind, ColumnKind::Skip))
        .collect();
    let mut per_column: Vec<std::vec::IntoIter<ColumnSlot<'a>>> = columns
        .par_iter_mut()
        .zip(decoded_nodes.into_par_iter())
        .map(|(column, node)| column_slots(column, node, layouts).into_iter())
        .collect();
    layouts
        .iter()
        .map(|_| {
            let mut column_slots = per_column.iter_mut();
            nodes
                .iter()
                .map(|node| match node.kind {
                    ColumnKind::Skip => ColumnSlot::Skip,
                    _ => column_slots
                        .next()
                        .and_then(Iterator::next)
                        .expect("one slot per message in each decoded column"),
                })
                .collect()
        })
        .collect()
}

fn column_slots<'a>(
    column: &'a mut BatchColumn,
    node: &ColumnNode,
    layouts: &'a [MessageLayout],
) -> Vec<ColumnSlot<'a>> {
    let lengths = || layouts.iter().map(|layout| layout.length(node.id));
    let positions = || layouts.iter().map(|layout| layout.positions(node.id));
    match (column, &node.kind) {
        (BatchColumn::Float64(storage), _) => storage
            .windows()
            .into_iter()
            .zip(positions())
            .map(|(window, positions)| ColumnSlot::Float64 { window, positions })
            .collect(),
        (BatchColumn::Int32(storage), _) => storage
            .windows()
            .into_iter()
            .zip(positions())
            .map(|(window, positions)| ColumnSlot::Int32 { window, positions })
            .collect(),
        (BatchColumn::Utf8 { values }, _) => split_by_lengths(values, lengths())
            .into_iter()
            .zip(positions())
            .map(|(values, positions)| ColumnSlot::Utf8 { values, positions })
            .collect(),
        (BatchColumn::FixedList { children, .. }, _) => {
            message_slots(children, &node.children, layouts)
                .into_iter()
                .map(|children| ColumnSlot::FixedList { children })
                .collect()
        }
        (BatchColumn::List { offsets, children }, ColumnKind::List { items_id, .. }) => {
            let mut items_base = 0;
            split_by_lengths(&mut offsets[1..], lengths())
                .into_iter()
                .zip(message_slots(children, &node.children, layouts))
                .zip(layouts)
                .map(|((ends, children), layout)| {
                    let slot = ColumnSlot::List(Box::new(ListSlot {
                        ends,
                        entries: layout.positions(node.id),
                        items_base,
                        items: layout.positions(*items_id),
                        kept: layout.kept_elements(*items_id),
                        children,
                    }));
                    items_base += layout.length(*items_id);
                    slot
                })
                .collect()
        }
        (BatchColumn::List { .. }, _) => unreachable!("list storage follows a list node"),
    }
}

/// Call `write(subset, code)` for the `n` codes of one compressed numeric
/// element; `code` is `None` for a missing value.
#[inline(always)]
fn for_each_compressed_code(
    reader: &mut BitReader,
    bit_width: u32,
    uses_missing_value: bool,
    n: usize,
    mut write: impl FnMut(usize, Option<u64>),
) {
    let reference = reader.read_bits(bit_width);
    let increment_bits = reader.read_bits(6) as u32;
    if increment_bits == 0 {
        let missing = uses_missing_value && BitReader::is_missing(reference, bit_width);
        let code = (!missing).then_some(reference);
        for subset in 0..n {
            write(subset, code);
        }
        return;
    }
    for subset in 0..n {
        let increment = reader.read_bits(increment_bits);
        let missing = uses_missing_value && BitReader::is_missing(increment, increment_bits);
        write(subset, (!missing).then_some(reference + increment));
    }
}

/// Call `store(index, code)` for each placed code of one compressed numeric
/// element; `code` is `None` for a missing value.
#[inline(always)]
fn place_compressed_codes(
    reader: &mut BitReader,
    bit_width: u32,
    uses_missing_value: bool,
    n: usize,
    placement: Placement<'_>,
    mut store: impl FnMut(usize, Option<u64>),
) {
    match placement {
        Placement::Strided { stride, occurrence } => {
            for_each_compressed_code(reader, bit_width, uses_missing_value, n, |subset, code| {
                store(subset * stride + occurrence, code)
            })
        }
        mut placement => {
            for_each_compressed_code(reader, bit_width, uses_missing_value, n, |subset, code| {
                if let Some(index) = placement.index(subset) {
                    store(index, code);
                }
            })
        }
    }
}

/// Decode one compressed message into its column slots. `kept` flags the
/// subsets that keep the current element of an enclosing filtered list.
fn decode_compressed_into_slots(
    reader: &mut BitReader,
    expanded: &[ExpandedItem],
    slots: &mut [ColumnSlot<'_>],
    n: usize,
    kept: Option<&[bool]>,
) -> Result<(), String> {
    let mut slots = slots.iter_mut();
    for item in expanded {
        if is_zero_width(item) {
            continue;
        }
        let slot = slots.next().ok_or(BATCH_LAYOUT_MISMATCH)?;
        match (item, slot) {
            (_, ColumnSlot::Skip) => skip_compressed_item(reader, item, n)?,
            (
                ExpandedItem::Descriptor(descriptor, entry),
                ColumnSlot::Float64 { window, positions },
            ) if !entry.is_string => {
                let divisor = 10f64.powi(entry.scale);
                place_compressed_codes(
                    reader,
                    entry.bit_width,
                    uses_missing_value(*descriptor),
                    n,
                    positions.place(kept)?,
                    |index, code| {
                        window.write(index, code.map(|code| physical_value(entry, divisor, code)))
                    },
                );
            }
            (
                ExpandedItem::Descriptor(descriptor, entry),
                ColumnSlot::Int32 { window, positions },
            ) if !entry.is_string => {
                place_compressed_codes(
                    reader,
                    entry.bit_width,
                    uses_missing_value(*descriptor),
                    n,
                    positions.place(kept)?,
                    |index, code| window.write(index, code.map(|code| code as i32)),
                );
            }
            (ExpandedItem::Descriptor(_, entry), ColumnSlot::Utf8 { values, positions })
                if entry.is_string =>
            {
                let mut placement = positions.place(kept)?;
                read_compressed_strings(reader, entry, n, |subset, value| {
                    if let Some(index) = placement.index(subset) {
                        values[index] = value;
                    }
                })?;
            }
            (
                ExpandedItem::FixedReplication { count, group, .. },
                ColumnSlot::FixedList { children },
            ) => {
                for _ in 0..*count {
                    decode_compressed_into_slots(reader, group, children, n, kept)?;
                }
            }
            (
                ExpandedItem::DelayedReplication {
                    factor_entry,
                    group,
                    ..
                },
                ColumnSlot::List(list),
            ) => {
                let ListSlot {
                    ends,
                    entries,
                    items_base,
                    items,
                    kept: list_kept,
                    children,
                } = &mut **list;
                let count = read_compressed_replication_count(reader, factor_entry.bit_width, n);
                let list_kept = *list_kept;
                for element in 0..count {
                    let element_kept = match list_kept {
                        Some(flags) => Some(
                            flags
                                .get(element * n..(element + 1) * n)
                                .ok_or(BATCH_LAYOUT_MISMATCH)?,
                        ),
                        None => kept,
                    };
                    // An entry no subset keeps writes nothing and is stepped over.
                    if element_kept.is_some_and(|flags| !flags.contains(&true)) {
                        for item in group.iter() {
                            skip_compressed_item(reader, item, n)?;
                        }
                        continue;
                    }
                    decode_compressed_into_slots(reader, group, children, n, element_kept)?;
                }
                let mut placement = entries.place(kept)?;
                match (items, list_kept) {
                    (
                        Positions::Strided {
                            occurrences: per_subset,
                            next: written,
                        },
                        None,
                    ) => {
                        *written += count;
                        if *written > *per_subset {
                            return Err(BATCH_LAYOUT_MISMATCH.into());
                        }
                        for subset in 0..n {
                            if let Some(entry) = placement.index(subset) {
                                // Offsets fit i32: allocation checked the batch item total.
                                ends[entry] =
                                    (*items_base + subset * *per_subset + *written) as i32;
                            }
                        }
                    }
                    (Positions::Kept { next }, list_kept) => {
                        let mut added = vec![0; n];
                        match list_kept {
                            Some(flags) => {
                                for element_flags in flags.chunks_exact(n) {
                                    for (added, keep) in added.iter_mut().zip(element_flags) {
                                        *added += usize::from(*keep);
                                    }
                                }
                            }
                            None => added.fill(count),
                        }
                        for subset in 0..n {
                            if let Some(entry) = placement.index(subset) {
                                next[subset] += added[subset];
                                ends[entry] = (*items_base + next[subset]) as i32;
                            }
                        }
                    }
                    _ => return Err(BATCH_LAYOUT_MISMATCH.into()),
                }
            }
            _ => return Err(BATCH_LAYOUT_MISMATCH.into()),
        }
    }
    if slots.next().is_some() {
        return Err(BATCH_LAYOUT_MISMATCH.into());
    }
    Ok(())
}

fn finish_batch_column(column: BatchColumn, field: &Field) -> Result<ArrayRef, String> {
    let struct_array =
        |children: Vec<BatchColumn>, item: &Field, length: usize| -> Result<StructArray, String> {
            let DataType::Struct(fields) = item.data_type() else {
                unreachable!("batch list items are structs")
            };
            let arrays = children
                .into_par_iter()
                .zip(fields.par_iter())
                .map(|(child, field)| finish_batch_column(child, field))
                .collect::<Result<Vec<_>, _>>()?;
            StructArray::try_new_with_length(fields.clone(), arrays, None, length)
                .map_err(|error| format!("Batch struct '{}' error: {error}", field.name()))
        };
    Ok(match (column, field.data_type()) {
        (BatchColumn::Float64(storage), _) => {
            // SAFETY: the caller finishes columns only after a complete decode.
            let (values, nulls) = unsafe { storage.assume_written() };
            Arc::new(Float64Array::new(ScalarBuffer::from(values), nulls))
        }
        (BatchColumn::Int32(storage), _) => {
            // SAFETY: the caller finishes columns only after a complete decode.
            let (values, nulls) = unsafe { storage.assume_written() };
            Arc::new(Int32Array::new(ScalarBuffer::from(values), nulls))
        }
        (BatchColumn::Utf8 { values }, _) => Arc::new(StringArray::from_iter(values)),
        (BatchColumn::FixedList { items, children }, DataType::FixedSizeList(item, size)) => {
            Arc::new(
                FixedSizeListArray::try_new(
                    Arc::clone(item),
                    *size,
                    Arc::new(struct_array(children, item, items)?),
                    None,
                )
                .map_err(|error| format!("Batch fixed list '{}' error: {error}", field.name()))?,
            )
        }
        (BatchColumn::List { offsets, children }, DataType::List(item)) => {
            let items = offsets.last().map_or(0, |end| *end as usize);
            Arc::new(
                ListArray::try_new(
                    Arc::clone(item),
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    Arc::new(struct_array(children, item, items)?),
                    None,
                )
                .map_err(|error| format!("Batch list '{}' error: {error}", field.name()))?,
            )
        }
        _ => unreachable!("batch column storage follows its field"),
    })
}

/// Decode compressed messages that share one template into batch-wide
/// columns: one array per field of `fields` that `selection` keeps, returned
/// with its output field. A filtered list keeps only the entries whose key
/// holds one of the filter's values.
///
/// A sizing pass reads each message's replication counts and filter keys,
/// the columns are allocated once for the whole batch, and each message is
/// decoded in parallel into its disjoint row range, so no per-message arrays
/// are concatenated. Unselected items and dropped list entries are stepped
/// over without being stored.
fn compressed_messages_to_columns(
    messages: &[&BufrMessage],
    expanded: &[ExpandedItem],
    fields: &[Field],
    selection: Option<&ProjectionTree>,
    filters: Option<&[ListValueFilterSpec]>,
    policy: ReadPolicy,
) -> Result<(Vec<Field>, Vec<ArrayRef>), String> {
    let mut node_count = 0;
    let mut nodes = Vec::with_capacity(fields.len());
    let mut output_fields = Vec::with_capacity(fields.len());
    for field in fields {
        let (node, output) = ColumnNode::new(
            field,
            structural_child_selection(selection, field.name()),
            &mut node_count,
        )?;
        nodes.push(node);
        output_fields.extend(output);
    }
    for filter in filters.unwrap_or_default() {
        let index = fields
            .iter()
            .position(|field| *field.name() == filter.path)
            .ok_or_else(|| format!("List filter path '{}' was not decoded", filter.path))?;
        let (key, _) = list_filter_key_index(&fields[index], filter)?;
        if let ColumnKind::List { filter: kept, .. } = &mut nodes[index].kind {
            *kept = Some(KeyFilter {
                key,
                values: filter.values.clone(),
            });
        }
    }
    let parsed: Vec<ParsedMessage<'_>> = messages
        .par_iter()
        .map(|message| parse_message(message))
        .collect::<Result<_, _>>()?;
    let layouts: Vec<MessageLayout> = parsed
        .par_iter()
        .map(|message| {
            let rows = message.num_subsets as usize;
            let mut occurrences = vec![0; node_count];
            let mut filtered = Vec::new();
            let mut reader = BitReader::with_policy(message.data, policy);
            count_compressed_occurrences(
                &mut reader,
                expanded,
                &nodes,
                rows,
                0,
                &mut occurrences,
                &mut filtered,
                false,
            )?;
            reader.validate_bounds()?;
            Ok(MessageLayout {
                rows,
                occurrences,
                filtered,
            })
        })
        .collect::<Result<_, String>>()?;
    // Unless overruns are rejected, the items after the last decoded node
    // change no output and are not read.
    let read_nodes = if policy.reject_overrun {
        nodes.len()
    } else {
        nodes
            .iter()
            .rposition(|node| !matches!(node.kind, ColumnKind::Skip))
            .map_or(0, |last| last + 1)
    };
    let read_items = expanded
        .iter()
        .enumerate()
        .filter(|(_, item)| !is_zero_width(item))
        .nth(read_nodes)
        .map_or(expanded.len(), |(index, _)| index);
    let mut columns = allocate_batch_columns(&nodes, &layouts)?;
    parsed
        .par_iter()
        .zip(message_slots(&mut columns, &nodes, &layouts).into_par_iter())
        .try_for_each(|(message, mut slots)| {
            let mut reader = BitReader::with_policy(message.data, policy);
            decode_compressed_into_slots(
                &mut reader,
                &expanded[..read_items],
                &mut slots[..read_nodes],
                message.num_subsets as usize,
                None,
            )?;
            #[cfg(debug_assertions)]
            assert!(
                slots_filled(&slots),
                "a decoded message left column slots unwritten"
            );
            reader.validate_bounds()
        })?;
    let arrays = columns
        .into_par_iter()
        .zip(output_fields.par_iter())
        .map(|(column, field)| finish_batch_column(column, field))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((output_fields, arrays))
}

fn structural_field(item: &StructuralSchemaItem) -> Result<Field, String> {
    let data_type = match item {
        StructuralSchemaItem::Scalar {
            entry, raw_code, ..
        } => leaf_data_type(entry.is_string, raw_code.is_some()),
        StructuralSchemaItem::Replication { kind, children, .. } => {
            let child_fields = children
                .iter()
                .map(structural_field)
                .collect::<Result<Vec<_>, _>>()?;
            let item_field = Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(child_fields)),
                true,
            ));
            match kind {
                StructuralReplication::Fixed { count, .. } => DataType::FixedSizeList(
                    item_field,
                    i32::try_from(*count)
                        .map_err(|_| "Fixed replication count exceeds Arrow i32".to_string())?,
                ),
                StructuralReplication::Delayed { .. } => DataType::List(item_field),
            }
        }
    };
    Ok(structural_item_field(item, data_type))
}

struct FlatArrowCursor<'a> {
    fields: &'a Fields,
    arrays: &'a [ArrayRef],
    index: usize,
}

impl<'a> FlatArrowCursor<'a> {
    fn next(&mut self) -> Result<(&'a Field, &'a ArrayRef), String> {
        let field = self
            .fields
            .get(self.index)
            .ok_or_else(|| "Structural layout consumed beyond the flat schema".to_string())?;
        let array = self
            .arrays
            .get(self.index)
            .ok_or_else(|| "Structural layout consumed beyond the flat arrays".to_string())?;
        self.index += 1;
        Ok((field.as_ref(), array))
    }
}

fn structural_raw_paths_for_flat_decode(
    schema: &[StructuralSchemaItem],
    requested: Option<&[String]>,
) -> Result<Option<Vec<String>>, String> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    let requested: HashSet<&str> = requested.iter().map(String::as_str).collect();
    let mut paths = Vec::new();

    fn visit(
        schema: &[StructuralSchemaItem],
        requested: &HashSet<&str>,
        delayed_prefix: Option<&str>,
        flat_names: &mut HashMap<Arc<str>, usize>,
        paths: &mut Vec<String>,
    ) {
        for item in schema {
            match item {
                StructuralSchemaItem::Scalar { entry, path, .. } => {
                    if requested.contains(path.as_str()) {
                        paths.push(field_path(delayed_prefix, &entry.mnemonic));
                    }
                }
                StructuralSchemaItem::Replication { kind, children, .. } => match kind {
                    StructuralReplication::Fixed { count, .. } => {
                        for _ in 0..*count {
                            visit(children, requested, delayed_prefix, flat_names, paths);
                        }
                    }
                    StructuralReplication::Delayed {
                        factor_mnemonic, ..
                    } => {
                        let name = delayed_group_name(flat_names, factor_mnemonic);
                        let prefix = field_path(delayed_prefix, &name);
                        visit(
                            children,
                            requested,
                            Some(&prefix),
                            &mut HashMap::new(),
                            paths,
                        );
                    }
                },
            }
        }
    }

    visit(schema, &requested, None, &mut HashMap::new(), &mut paths);
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        return Err("Raw-code structural paths did not resolve to flat leaves".into());
    }
    Ok(Some(paths))
}

fn interleave_fixed_occurrences(
    arrays: &[ArrayRef],
    row_count: usize,
    path: &str,
) -> Result<ArrayRef, String> {
    if arrays.is_empty() {
        return Err(format!("Fixed replication '{path}' has no child arrays"));
    }
    if arrays
        .iter()
        .any(|array| array.len() != row_count || array.data_type() != arrays[0].data_type())
    {
        return Err(format!(
            "Fixed replication '{path}' child arrays are not type/length compatible"
        ));
    }
    let array_refs: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    let concatenated =
        concat(&array_refs).map_err(|error| format!("Failed to stage '{path}': {error}"))?;
    let mut indices = Vec::with_capacity(row_count * arrays.len());
    for row in 0..row_count {
        for occurrence in 0..arrays.len() {
            indices.push((occurrence * row_count + row) as u64);
        }
    }
    take(concatenated.as_ref(), &UInt64Array::from(indices), None)
        .map_err(|error| format!("Failed to transpose '{path}': {error}"))
}

/// Restack one flat sequence into `schema`. The flat layout inlines fixed
/// replications, so delayed groups inside every fixed occurrence share the
/// numbering of their enclosing flat sequence, counted in `flat_names`.
fn restack_structural_sequence(
    schema: &[StructuralSchemaItem],
    selection: Option<&ProjectionTree>,
    cursor: &mut FlatArrowCursor<'_>,
    row_count: usize,
    flat_names: &mut HashMap<Arc<str>, usize>,
) -> Result<(Vec<Field>, Vec<ArrayRef>), String> {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for item in schema {
        let item_selection = structural_child_selection(selection, item.name());
        match item {
            StructuralSchemaItem::Scalar {
                descriptor,
                entry,
                path,
                raw_code,
            } => {
                let (flat_field, array) = cursor.next()?;
                if flat_field.name() != &*entry.mnemonic {
                    return Err(format!(
                        "Structural leaf '{path}' expected flat field '{}', found '{}'",
                        entry.mnemonic,
                        flat_field.name()
                    ));
                }
                let expected_type = leaf_data_type(entry.is_string, raw_code.is_some());
                if array.data_type() != &expected_type {
                    return Err(format!(
                        "Structural leaf '{path}' expected {expected_type}, found {}",
                        array.data_type()
                    ));
                }
                if item_selection.is_some() {
                    fields.push(
                        Field::new(&*entry.mnemonic, expected_type, true).with_metadata(
                            scalar_structural_metadata(*descriptor, entry, path, raw_code.as_ref()),
                        ),
                    );
                    arrays.push(array.clone());
                }
            }
            StructuralSchemaItem::Replication {
                kind,
                children,
                path,
                ..
            } => match kind {
                StructuralReplication::Fixed { count, .. } => {
                    let mut occurrence_fields = Vec::with_capacity(*count);
                    let mut occurrence_arrays = Vec::with_capacity(*count);
                    for _ in 0..*count {
                        let (child_fields, child_arrays) = restack_structural_sequence(
                            children,
                            item_selection.flatten(),
                            cursor,
                            row_count,
                            flat_names,
                        )?;
                        occurrence_fields.push(child_fields);
                        occurrence_arrays.push(child_arrays);
                    }
                    if item_selection.is_none() {
                        continue;
                    }
                    let child_fields = occurrence_fields
                        .first()
                        .cloned()
                        .ok_or_else(|| format!("Fixed replication '{path}' has zero count"))?;
                    if occurrence_fields
                        .iter()
                        .any(|candidate| candidate != &child_fields)
                    {
                        return Err(format!(
                            "Fixed replication '{path}' occurrences have different schemas"
                        ));
                    }
                    let mut child_arrays = Vec::with_capacity(child_fields.len());
                    for child_index in 0..child_fields.len() {
                        let arrays: Vec<ArrayRef> = occurrence_arrays
                            .iter()
                            .map(|occurrence| occurrence[child_index].clone())
                            .collect();
                        child_arrays.push(interleave_fixed_occurrences(&arrays, row_count, path)?);
                    }
                    let struct_fields = Fields::from(child_fields);
                    let struct_array =
                        StructArray::try_new(struct_fields.clone(), child_arrays, None).map_err(
                            |error| format!("Failed to build fixed struct '{path}': {error}"),
                        )?;
                    let item_field =
                        Arc::new(Field::new("item", DataType::Struct(struct_fields), true));
                    let list_size = i32::try_from(*count)
                        .map_err(|_| format!("Fixed replication '{path}' exceeds Arrow i32"))?;
                    let array = FixedSizeListArray::try_new(
                        Arc::clone(&item_field),
                        list_size,
                        Arc::new(struct_array),
                        None,
                    )
                    .map_err(|error| format!("Failed to build fixed list '{path}': {error}"))?;
                    let data_type = DataType::FixedSizeList(item_field, list_size);
                    fields.push(structural_item_field(item, data_type));
                    arrays.push(Arc::new(array));
                }
                StructuralReplication::Delayed {
                    factor_mnemonic, ..
                } => {
                    let (flat_field, flat_array) = cursor.next()?;
                    let flat_name = delayed_group_name(flat_names, factor_mnemonic);
                    if flat_field.name() != &*flat_name {
                        return Err(format!(
                            "Structural delayed group '{path}' expected flat field '{flat_name}', found '{}'",
                            flat_field.name()
                        ));
                    }
                    let list = flat_array
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .ok_or_else(|| format!("Delayed group '{path}' is not List<Struct>"))?;
                    let values = list
                        .values()
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .ok_or_else(|| format!("Delayed group '{path}' is not List<Struct>"))?;
                    let struct_fields = match values.data_type() {
                        DataType::Struct(fields) => fields,
                        _ => unreachable!(),
                    };
                    let mut child_cursor = FlatArrowCursor {
                        fields: struct_fields,
                        arrays: values.columns(),
                        index: 0,
                    };
                    let (child_fields, child_arrays) = restack_structural_sequence(
                        children,
                        item_selection.flatten(),
                        &mut child_cursor,
                        values.len(),
                        &mut HashMap::new(),
                    )?;
                    if child_cursor.index != values.num_columns() {
                        return Err(format!(
                            "Delayed group '{path}' left {} flat children unused",
                            values.num_columns() - child_cursor.index
                        ));
                    }
                    if item_selection.is_none() {
                        continue;
                    }
                    let child_fields = Fields::from(child_fields);
                    let struct_array = StructArray::try_new(
                        child_fields.clone(),
                        child_arrays,
                        values.nulls().cloned(),
                    )
                    .map_err(|error| format!("Failed to build delayed struct '{path}': {error}"))?;
                    let item_field =
                        Arc::new(Field::new("item", DataType::Struct(child_fields), true));
                    let array = ListArray::try_new(
                        Arc::clone(&item_field),
                        list.offsets().clone(),
                        Arc::new(struct_array),
                        list.nulls().cloned(),
                    )
                    .map_err(|error| format!("Failed to build delayed list '{path}': {error}"))?;
                    fields.push(structural_item_field(item, DataType::List(item_field)));
                    arrays.push(Arc::new(array));
                }
            },
        }
    }
    Ok((fields, arrays))
}

fn structural_schema_metadata() -> HashMap<String, String> {
    HashMap::from([(
        "earth2bufr.fixed_replication".to_string(),
        "structural".to_string(),
    )])
}

fn restack_flat_batch_structurally(
    flat: RecordBatch,
    schema: &[StructuralSchemaItem],
    projection: Option<&ProjectionTree>,
) -> Result<RecordBatch, String> {
    let flat_schema = flat.schema();
    let mut cursor = FlatArrowCursor {
        fields: flat_schema.fields(),
        arrays: flat.columns(),
        index: 0,
    };
    let (mut fields, mut arrays) = restack_structural_sequence(
        schema,
        projection,
        &mut cursor,
        flat.num_rows(),
        &mut HashMap::new(),
    )?;
    while cursor.index < flat.num_columns() {
        let (field, array) = cursor.next()?;
        if !APPENDED_COLUMNS.contains(&field.name().as_str()) {
            return Err(format!(
                "Structural layout left unexpected flat field '{}' unused",
                field.name()
            ));
        }
        fields.push(field.clone());
        arrays.push(array.clone());
    }
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        structural_schema_metadata(),
    ));
    RecordBatch::try_new(schema, arrays)
        .map_err(|error| format!("Structural Arrow restack error: {error}"))
}

/// Messages decoded together, with what the file index holds about them.
/// They share data category, compression, descriptors and table epoch, so
/// `template` describes every message.
struct MessageGroup<'a> {
    messages: Vec<&'a BufrMessage>,
    indices: &'a [usize],
    template: &'a IndexedDataMessage,
    subset_counts: Vec<u16>,
    tables: &'a TableSet,
}

impl MessageGroup<'_> {
    fn rows(&self) -> usize {
        self.subset_counts.iter().map(|count| *count as usize).sum()
    }
}

fn with_data_category(mut arrays: Vec<ArrayRef>, data_category: u8, rows: usize) -> Vec<ArrayRef> {
    arrays.push(Arc::new(Float64Array::from_value(
        data_category as f64,
        rows,
    )));
    arrays
}

fn decode_structural_bounded_group(
    group: &MessageGroup<'_>,
    options: BoundedDecodeOptions<'_>,
) -> Result<RecordBatch, String> {
    debug_assert!(
        options.list_value_filters.is_none(),
        "iter_batches rejects structural list filters"
    );
    let expanded = expand_descriptors(
        &group.template.descriptors,
        group.tables,
        options.policy.strict,
        FixedReplicationLayout::Structural,
    )?;
    let mut schema = derive_structural_schema(&expanded, None);
    let present;
    let options = if options.projection_missing == ProjectionMissing::Skip {
        present = options.present_fields(|path| {
            find_structural_item(&schema, &path.split('.').collect::<Vec<_>>()).is_some()
        })?;
        options.with_fields(&present)
    } else {
        options
    };
    apply_structural_raw_code_paths(&mut schema, options.raw_code_paths)?;
    if let Some(projection) = options.projection {
        validate_structural_paths(&schema, &projection.paths, "Projection")?;
    }
    let selection = options.projection.map(|projection| &projection.tree);
    if group.template.is_compressed {
        let fields = schema
            .iter()
            .map(structural_field)
            .collect::<Result<Vec<_>, _>>()?;
        let (mut fields, arrays) = compressed_messages_to_columns(
            &group.messages,
            &expanded,
            &fields,
            selection,
            None,
            options.policy,
        )?;
        fields.push(Field::new(DATA_CATEGORY_COLUMN, DataType::Float64, true));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(
                fields,
                structural_schema_metadata(),
            )),
            with_data_category(arrays, group.template.data_category, group.rows()),
        )
        .map_err(|error| format!("Direct structural Arrow error: {error}"))?;
        return append_source_provenance(batch, group.indices, &group.subset_counts);
    }
    let flat_raw_paths = structural_raw_paths_for_flat_decode(&schema, options.raw_code_paths)?;
    let flat = decode_bounded_group(
        group,
        BoundedDecodeOptions {
            projection: None,
            raw_code_paths: flat_raw_paths.as_deref(),
            list_value_filters: None,
            policy: options.policy,
            fixed_replication: FixedReplicationLayout::Flat,
            projection_missing: ProjectionMissing::Raise,
        },
    )?;
    restack_flat_batch_structurally(flat, &schema, selection)
}

fn schema_item_name(item: &SchemaItem) -> &str {
    match item {
        SchemaItem::Scalar { name, .. } | SchemaItem::ReplicationGroup { name, .. } => name,
    }
}

/// Whether a flat `FIELD` or `FIELD.CHILD` path names a field of `schema`.
fn flat_path_present(schema: &[SchemaItem], path: &str) -> bool {
    let (root, child) = match path.split_once('.') {
        Some((root, child)) => (root, Some(child)),
        None => (path, None),
    };
    schema.iter().any(|item| match (item, child) {
        (item, None) => schema_item_name(item) == root,
        (SchemaItem::ReplicationGroup { name, children }, Some(child)) => {
            &**name == root && children.iter().any(|item| schema_item_name(item) == child)
        }
        (SchemaItem::Scalar { .. }, Some(_)) => false,
    })
}

/// Decode one bounded group of compatible, nonzero-subset messages.
fn decode_bounded_group(
    group: &MessageGroup<'_>,
    options: BoundedDecodeOptions<'_>,
) -> Result<RecordBatch, String> {
    if options.fixed_replication == FixedReplicationLayout::Structural {
        return decode_structural_bounded_group(group, options);
    }

    let template = group.template;
    let mut expanded = expand_descriptors(
        &template.descriptors,
        group.tables,
        options.policy.strict,
        FixedReplicationLayout::Flat,
    )?;
    append_data_category(&mut expanded);
    let mut schema = derive_schema(&expanded);
    let present;
    let options = if options.projection_missing == ProjectionMissing::Skip {
        present = options.present_fields(|path| flat_path_present(&schema, path))?;
        options.with_fields(&present)
    } else {
        options
    };
    let BoundedDecodeOptions {
        projection,
        raw_code_paths,
        list_value_filters,
        policy,
        ..
    } = options;
    apply_raw_code_paths(&expanded, &mut schema, raw_code_paths)?;
    let batch = if template.is_compressed {
        let (mut fields, arrays) = compressed_messages_to_columns(
            &group.messages,
            &expanded[..expanded.len() - 1],
            &schema_to_fields(&schema[..schema.len() - 1]),
            projection.map(|projection| &projection.tree),
            list_value_filters,
            policy,
        )?;
        fields.push(Field::new(DATA_CATEGORY_COLUMN, DataType::Float64, true));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            with_data_category(arrays, template.data_category, group.rows()),
        )
        .map_err(|error| format!("Arrow error: {error}"))?;
        append_source_provenance(batch, group.indices, &group.subset_counts)?
    } else {
        let projection_roots = projection.map(Projection::roots);
        let batch = parallel_decode_uncompressed(
            &group.messages,
            &group.subset_counts,
            &expanded,
            &schema,
            template.data_category,
            projection_roots.as_deref(),
            policy,
        )?;
        let batch = append_source_provenance(batch, group.indices, &group.subset_counts)?;
        apply_list_value_filters(batch, list_value_filters)?
    };
    project_record_batch(batch, projection)
}

// =============================================================================
// Python Interface
// =============================================================================

#[derive(Debug)]
struct IndexedDataMessage {
    data_category: u8,
    num_subsets: u16,
    is_compressed: bool,
    descriptors: Vec<u32>,
}

#[derive(Debug)]
struct BufrFileState {
    /// The file the index was built from, read again for message bytes.
    file: fs::File,
    /// Each message up to its Section 4 header; DX table messages whole.
    headers: Vec<BufrMessage>,
    /// Byte range of each message in the file.
    spans: Vec<std::ops::Range<usize>>,
    data_message_indices: Vec<usize>,
    message_index: Vec<Option<IndexedDataMessage>>,
    /// One entry per DX table epoch in the file, in stream order.
    epochs: Vec<TableEpoch>,
    /// Message index -> index into `epochs`. `usize::MAX` for DX/empty messages.
    message_epoch: Vec<usize>,
    strict: bool,
}

impl BufrFileState {
    fn indexed(&self, message_index: usize) -> Result<&IndexedDataMessage, String> {
        self.message_index
            .get(message_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("Source message index {message_index} is not a data message"))
    }

    /// Tables in force at the point in the stream where this message sits.
    ///
    /// See `TableEpoch`.
    fn tables_for(&self, message_index: usize) -> Result<&TableSet, String> {
        let epoch = self
            .message_epoch
            .get(message_index)
            .copied()
            .unwrap_or(usize::MAX);
        self.epochs
            .get(epoch)
            .map(|epoch| &epoch.tables)
            .ok_or_else(|| format!("BUFR message {message_index} has no DX table epoch"))
    }

    /// Read the whole messages at `indices` from the file, in parallel reads of
    /// up to `FETCH_READ_SIZE` bytes that each cover nearby messages.
    fn read_messages(&self, indices: &[usize]) -> Result<Vec<BufrMessage>, String> {
        let mut reads: Vec<std::ops::Range<usize>> = Vec::new();
        let mut first = 0;
        for position in 1..=indices.len() {
            let split = position == indices.len() || {
                let read_start = self.spans[indices[first]].start;
                let previous = &self.spans[indices[position - 1]];
                let next = &self.spans[indices[position]];
                next.start < previous.end
                    || next.start - previous.end > FETCH_MAX_GAP
                    || next.end - read_start > FETCH_READ_SIZE
            };
            if split {
                reads.push(first..position);
                first = position;
            }
        }
        let messages: Vec<Vec<BufrMessage>> = reads
            .par_iter()
            .map(|positions| {
                let read_indices = &indices[positions.clone()];
                let start = self.spans[read_indices[0]].start;
                let end = self.spans[read_indices[read_indices.len() - 1]].end;
                let mut buffer: Arc<[u8]> = std::iter::repeat_n(0, end - start).collect();
                let bytes = Arc::get_mut(&mut buffer).expect("a new buffer is unshared");
                read_exact_at(&self.file, bytes, start as u64).map_err(|error| {
                    format!("Failed to read bytes {start}..{end} of the opened BUFR file: {error}")
                })?;
                read_indices
                    .iter()
                    .map(|&index| {
                        let span = &self.spans[index];
                        let message = BufrMessage {
                            buffer: Arc::clone(&buffer),
                            start: span.start - start,
                            end: span.end - start,
                        };
                        let bytes = message.bytes();
                        if !bytes.starts_with(self.headers[index].bytes())
                            || !bytes.ends_with(b"7777")
                        {
                            return Err(format!(
                                "BUFR message {index} at bytes {}..{} changed after the file \
                                 was opened",
                                span.start, span.end
                            ));
                        }
                        Ok(message)
                    })
                    .collect()
            })
            .collect::<Result<_, String>>()?;
        Ok(messages.into_iter().flatten().collect())
    }
}

/// Upper bound on one read of `BufrFileState::read_messages`, unless a single
/// message is larger.
const FETCH_READ_SIZE: usize = 1 << 20;

/// Largest gap between two messages that one read of
/// `BufrFileState::read_messages` spans.
const FETCH_MAX_GAP: usize = 4096;

/// Read `buf` from `offset` in `SCAN_CHUNK_SIZE` pieces, concurrently on the
/// decode pool when it has more than one thread.
fn read_chunks_at(file: &fs::File, buf: &mut [u8], offset: usize) -> std::io::Result<()> {
    let pool = decode_pool().map_err(std::io::Error::other)?;
    if buf.len() <= SCAN_CHUNK_SIZE || pool.current_num_threads() == 1 {
        return read_exact_at(file, buf, offset as u64);
    }
    pool.install(|| {
        buf.par_chunks_mut(SCAN_CHUNK_SIZE)
            .enumerate()
            .try_for_each(|(index, chunk)| {
                read_exact_at(file, chunk, (offset + index * SCAN_CHUNK_SIZE) as u64)
            })
    })
}

fn read_exact_at(file: &fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        let mut buf = buf;
        let mut offset = offset;
        while !buf.is_empty() {
            match std::os::windows::fs::FileExt::seek_read(file, buf, offset) {
                Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                Ok(read) => {
                    buf = &mut buf[read..];
                    offset += read as u64;
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

/// Cached BUFR file index and tables for bounded reads.
#[pyclass]
struct BufrFileHandle {
    state: Arc<BufrFileState>,
}

fn collect_active_descriptors(
    descriptors: &[u32],
    tables: &TableSet,
    sequences: &mut FxHashMap<u32, Value>,
    elements: &mut FxHashMap<u32, Value>,
) -> Result<(), String> {
    let mut index = 0;
    while index < descriptors.len() {
        let descriptor = descriptors[index];
        let f = (descriptor >> 14) & 0x03;
        match f {
            0 => {
                if let Some(entry) = tables.table_b.get(&descriptor) {
                    elements.entry(descriptor).or_insert_with(|| {
                        json!({
                            "descriptor": descriptor_to_string(descriptor),
                            "mnemonic": entry.mnemonic,
                            "description": entry.name,
                            "units": entry.units,
                            "base_scale": entry.scale,
                            "base_reference": entry.reference_value,
                            "base_width": entry.bit_width,
                            "value_kind": value_kind(entry),
                        })
                    });
                }
                index += 1;
            }
            // Active-table discovery only needs the descriptor closure. Factor
            // and replicated members are visited as subsequent tokens; actual
            // replication/operator execution is owned by expand_descriptors.
            1 | 2 => index += 1,
            3 => {
                if sequences.contains_key(&descriptor) {
                    index += 1;
                    continue;
                }
                let entry = tables.table_d.get(&descriptor).ok_or_else(|| {
                    format!("Unknown sequence {}", descriptor_to_string(descriptor))
                })?;
                sequences.entry(descriptor).or_insert_with(|| {
                    json!({
                        "descriptor": descriptor_to_string(descriptor),
                        "mnemonic": entry.mnemonic,
                        "members": entry.descriptors.iter().map(|value| descriptor_to_string(*value)).collect::<Vec<_>>(),
                    })
                });
                collect_active_descriptors(&entry.descriptors, tables, sequences, elements)?;
                index += 1;
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn collect_effective_occurrences(
    expanded: &[ExpandedItem],
    prefix: Option<&str>,
    occurrences: &mut Vec<Value>,
    mnemonic_counts: &mut HashMap<String, usize>,
) {
    let mut replication_names = HashMap::new();
    for item in expanded {
        match item {
            ExpandedItem::Descriptor(descriptor, entry) => {
                if entry.bit_width == 0 {
                    continue;
                }
                let mnemonic = entry.mnemonic.to_string();
                let occurrence = mnemonic_counts.entry(mnemonic.clone()).or_insert(0);
                let decoder_path = field_path(prefix, &mnemonic);
                let missing_code = if !uses_missing_value(*descriptor) || entry.bit_width >= 64 {
                    None
                } else {
                    Some((1_u64 << entry.bit_width) - 1)
                };
                occurrences.push(json!({
                    "descriptor": descriptor_to_string(*descriptor),
                    "mnemonic": mnemonic,
                    "decoder_path": decoder_path,
                    "global_mnemonic_occurrence": *occurrence,
                    "effective_scale": entry.scale,
                    "effective_reference": entry.reference_value,
                    "effective_width": entry.bit_width,
                    "source_missing_code": missing_code,
                    "units": entry.units,
                    "value_kind": value_kind(entry),
                }));
                *occurrence += 1;
            }
            ExpandedItem::FixedReplication { .. } => {
                unreachable!("flat expansion inlines fixed replication")
            }
            ExpandedItem::DelayedReplication {
                factor_desc,
                factor_entry,
                group,
            } => {
                let name = delayed_group_name(&mut replication_names, &factor_entry.mnemonic);
                let path = field_path(prefix, &name);
                occurrences.push(json!({
                    "descriptor": descriptor_to_string(*factor_desc),
                    "mnemonic": factor_entry.mnemonic,
                    "decoder_path": path,
                    "structural_replication_factor": true,
                    "effective_scale": factor_entry.scale,
                    "effective_reference": factor_entry.reference_value,
                    "effective_width": factor_entry.bit_width,
                    "source_missing_code": Value::Null,
                    "units": factor_entry.units,
                }));
                collect_effective_occurrences(group, Some(&path), occurrences, mnemonic_counts);
            }
        }
    }
}

fn schema_item_payload(item: &SchemaItem) -> Value {
    match item {
        SchemaItem::Scalar {
            name, is_string, ..
        } => json!({
            "name": name,
            "arrow_type": if *is_string { "string" } else { "double" },
        }),
        SchemaItem::ReplicationGroup { name, children } => json!({
            "name": name,
            "arrow_type": "list<struct>",
            "children": children.iter().map(schema_item_payload).collect::<Vec<_>>(),
        }),
    }
}

/// The rayon pool decodes run on. Built on first use and replaced by
/// `set_num_threads`; a decode keeps the pool it started on.
static DECODE_POOL: RwLock<Option<Arc<rayon::ThreadPool>>> = RwLock::new(None);

/// Thread count from `EARTH2BUFR_NUM_THREADS`, else `RAYON_NUM_THREADS`, else the
/// available parallelism. Unset, unparsable and zero values are skipped.
fn default_num_threads() -> usize {
    ["EARTH2BUFR_NUM_THREADS", "RAYON_NUM_THREADS"]
        .iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|&threads| threads > 0)
        })
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

fn build_decode_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("earth2bufr-decode-{index}"))
        .build()
        .map(Arc::new)
        .map_err(|error| {
            format!(
                "Cannot start {threads} BUFR decode threads: {error}; set a smaller count with \
                 earth2bufr.set_num_threads or EARTH2BUFR_NUM_THREADS"
            )
        })
}

/// The current decode pool, built with `default_num_threads` on first use.
fn decode_pool() -> Result<Arc<rayon::ThreadPool>, String> {
    if let Some(pool) = DECODE_POOL
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
    {
        return Ok(Arc::clone(pool));
    }
    let mut slot = DECODE_POOL.write().unwrap_or_else(PoisonError::into_inner);
    if let Some(pool) = slot.as_ref() {
        return Ok(Arc::clone(pool));
    }
    let pool = build_decode_pool(default_num_threads())?;
    *slot = Some(Arc::clone(&pool));
    Ok(pool)
}

thread_local! {
    /// A one-thread pool made of the calling thread, so that a one-thread decode
    /// runs on the thread that asks for the batch. `None` when the thread is
    /// already a worker of another rayon pool.
    static CALLER_POOL: Option<rayon::ThreadPool> = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .ok();
}

/// Message grouping and decode options of one `iter_batches` call, owned by
/// the prefetch thread once iteration starts, or by the iterator when the
/// decode pool has one thread.
struct BatchPlanner {
    state: Arc<BufrFileState>,
    message_indices: Vec<usize>,
    cursor: usize,
    messages_per_batch: usize,
    target_decoded_bytes: Option<usize>,
    estimated_template_index: Option<usize>,
    estimated_bytes_per_subset: Option<f64>,
    options: ParsedDecodeOptions,
}

impl BatchPlanner {
    fn compatible(&self, left_index: usize, right_index: usize) -> Result<bool, String> {
        let left = self.state.indexed(left_index)?;
        let right = self.state.indexed(right_index)?;
        // Identical descriptors can still mean different things under different
        // DX table epochs, so the epoch is part of the decoding identity.
        Ok(left.data_category == right.data_category
            && left.is_compressed == right.is_compressed
            && left.descriptors == right.descriptors
            && self.state.message_epoch[left_index] == self.state.message_epoch[right_index])
    }

    fn take_next_group(&mut self) -> Result<Option<Vec<usize>>, String> {
        if self.cursor >= self.message_indices.len() {
            return Ok(None);
        }

        let start = self.cursor;
        let first_index = self.message_indices[start];
        let first = self.state.indexed(first_index)?;
        let estimated_bytes_per_subset = match self.estimated_template_index {
            Some(index) if self.compatible(index, first_index)? => self.estimated_bytes_per_subset,
            _ => None,
        };
        let message_limit = self.messages_per_batch;
        let mut subsets = first.num_subsets as usize;
        let mut encoded_bytes = self.state.spans[first_index].len();
        let mut end = start + 1;
        while end < self.message_indices.len() && end - start < message_limit {
            if let (Some(target), Some(bytes_per_subset)) =
                (self.target_decoded_bytes, estimated_bytes_per_subset)
            {
                // The group's message bytes are held while it decodes, so they
                // count against the target as well.
                if bytes_per_subset * subsets as f64 >= target as f64 || encoded_bytes >= target {
                    break;
                }
            } else if self.target_decoded_bytes.is_some() {
                // One complete message calibrates a new compatible template.
                break;
            }
            let candidate_index = self.message_indices[end];
            let candidate = self.state.indexed(candidate_index)?;
            if !self.compatible(first_index, candidate_index)? {
                break;
            }
            subsets += candidate.num_subsets as usize;
            encoded_bytes += self.state.spans[candidate_index].len();
            end += 1;
        }
        self.cursor = end;
        Ok(Some(self.message_indices[start..end].to_vec()))
    }

    fn update_decoded_size_estimate(&mut self, group: &[usize], batch: &RecordBatch) {
        if self.target_decoded_bytes.is_none() || group.is_empty() || batch.num_rows() == 0 {
            return;
        }
        self.estimated_template_index = Some(group[0]);
        self.estimated_bytes_per_subset =
            Some(decoded_bytes(batch) as f64 / batch.num_rows() as f64);
    }

    fn decode_groups(&self, groups: &[Vec<usize>]) -> Vec<DecodedGroup> {
        let options = self.options.bounded(self.state.strict);
        let decode = |indices: &Vec<usize>| {
            decode_bounded_indices_resilient(
                &self.state,
                indices,
                options,
                self.options.skip_truncated_messages,
            )
        };
        if groups.len() == 1 {
            vec![decode(&groups[0])]
        } else {
            groups.par_iter().map(decode).collect()
        }
    }

    /// Plan and decode the next groups in source order, at most `group_limit`
    /// of them, or one with a byte target since each group is then sized from
    /// the decode before it. A planning error is the last group returned and
    /// ends the iteration; an empty result means the iteration is done.
    fn decode_next(&mut self, group_limit: usize, pool: &rayon::ThreadPool) -> Vec<DecodedGroup> {
        let group_limit = if self.target_decoded_bytes.is_some() {
            1
        } else {
            group_limit
        };
        let mut groups = Vec::with_capacity(group_limit);
        let mut planning_error = None;
        while groups.len() < group_limit {
            match self.take_next_group() {
                Ok(Some(group)) => groups.push(group),
                Ok(None) => break,
                Err(error) => {
                    planning_error = Some(error);
                    self.cursor = self.message_indices.len();
                    break;
                }
            }
        }
        let mut decoded = pool.install(|| self.decode_groups(&groups));
        if let (Some(group), Some(Ok((batches, failures)))) = (groups.first(), decoded.first()) {
            if failures.is_empty() && batches.len() == 1 {
                self.update_decoded_size_estimate(group, &batches[0]);
            }
        }
        decoded.extend(planning_error.map(Err));
        decoded
    }
}

type DecodedGroup = Result<(Vec<RecordBatch>, Vec<DecodeFailure>), String>;

/// Decoded groups waiting for the consumer, in source order, with their
/// decoded bytes when a byte target is set.
#[derive(Default)]
struct PrefetchQueue {
    groups: VecDeque<(DecodedGroup, usize)>,
    queued_bytes: usize,
    producer_done: bool,
    consumer_gone: bool,
}

#[derive(Default)]
struct PrefetchChannel {
    queue: Mutex<PrefetchQueue>,
    changed: Condvar,
}

impl PrefetchChannel {
    fn lock(&self) -> MutexGuard<'_, PrefetchQueue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'a>(&self, queue: MutexGuard<'a, PrefetchQueue>) -> MutexGuard<'a, PrefetchQueue> {
        self.changed
            .wait(queue)
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Marks the producer done when its thread returns or unwinds.
struct ProducerDone(Arc<PrefetchChannel>);

impl Drop for ProducerDone {
    fn drop(&mut self) {
        self.0.lock().producer_done = true;
        self.0.changed.notify_all();
    }
}

/// Decode groups in source order ahead of the consumer. Before each decode the
/// producer waits while `prefetch_batches` groups are queued or, with a byte
/// target, while queued bytes exceed `(prefetch_batches - 1) * target`, so
/// queued plus in-progress groups stay within `prefetch_batches` groups and
/// about `prefetch_batches * target` bytes.
fn run_prefetch(mut planner: BatchPlanner, channel: &PrefetchChannel, prefetch_batches: usize) {
    let queued_byte_limit = planner
        .target_decoded_bytes
        .map(|target| target.saturating_mul(prefetch_batches - 1));
    loop {
        let free_slots = {
            let mut queue = channel.lock();
            loop {
                if queue.consumer_gone {
                    return;
                }
                let full = queue.groups.len() >= prefetch_batches
                    || queued_byte_limit.is_some_and(|limit| queue.queued_bytes > limit);
                if !full {
                    break prefetch_batches - queue.groups.len();
                }
                queue = channel.wait(queue);
            }
        };
        // `__next__` built the pool before starting this thread, and a pool is
        // only ever replaced by another one.
        let pool = decode_pool().expect("the decode pool exists once a prefetch thread runs");
        let decoded = planner.decode_next(free_slots, &pool);
        if decoded.is_empty() {
            return;
        }
        let mut queue = channel.lock();
        for group in decoded {
            let bytes = match (&group, planner.target_decoded_bytes) {
                (Ok((batches, _)), Some(_)) => batches.iter().map(decoded_bytes).sum(),
                _ => 0,
            };
            queue.queued_bytes += bytes;
            queue.groups.push_back((group, bytes));
        }
        channel.changed.notify_all();
    }
}

/// The prefetch thread of one iterator; dropping it stops and joins the thread.
struct Prefetch {
    channel: Arc<PrefetchChannel>,
    thread: Option<JoinHandle<()>>,
}

impl Prefetch {
    fn start(planner: BatchPlanner, prefetch_batches: usize) -> std::io::Result<Self> {
        let channel = Arc::new(PrefetchChannel::default());
        let producer_channel = Arc::clone(&channel);
        let thread = std::thread::Builder::new()
            .name("earth2bufr-prefetch".into())
            .spawn(move || {
                let _done = ProducerDone(Arc::clone(&producer_channel));
                run_prefetch(planner, &producer_channel, prefetch_batches);
            })?;
        Ok(Self {
            channel,
            thread: Some(thread),
        })
    }

    /// The next decoded group in source order, or `None` once the producer is done.
    fn next_group(&self) -> Option<DecodedGroup> {
        let mut queue = self.channel.lock();
        loop {
            if let Some((group, bytes)) = queue.groups.pop_front() {
                queue.queued_bytes -= bytes;
                self.channel.changed.notify_all();
                return Some(group);
            }
            if queue.producer_done {
                return None;
            }
            queue = self.channel.wait(queue);
        }
    }

    fn stop(&mut self) -> std::thread::Result<()> {
        self.channel.lock().consumer_gone = true;
        self.channel.changed.notify_all();
        self.thread.take().map_or(Ok(()), JoinHandle::join)
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        if self.thread.is_some() {
            // The producer finishes its current decode before it sees the stop.
            Python::with_gil(|py| {
                py.allow_threads(|| {
                    let _ = self.stop();
                })
            });
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Native ordered iterator over bounded, compatible BUFR message groups,
/// decoded ahead of the consumer on a prefetch thread, or during each
/// `__next__` call when the decode pool has one thread.
#[pyclass]
struct BufrBatchIterator {
    planner: Option<BatchPlanner>,
    prefetch_batches: usize,
    prefetch: Option<Prefetch>,
    ready_batches: VecDeque<RecordBatch>,
    decode_failures: Vec<DecodeFailure>,
}

/// Bytes the batch's values occupy, excluding unused buffer capacity.
fn decoded_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| {
            column
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| column.get_array_memory_size())
        })
        .sum()
}

/// Decode `messages`, read from the file at `indices`. Callers group
/// messages of one data category, compression, template and table epoch, so
/// the first index describes the whole group.
fn decode_bounded_indices(
    state: &BufrFileState,
    indices: &[usize],
    messages: &[BufrMessage],
    options: BoundedDecodeOptions<'_>,
) -> Result<RecordBatch, String> {
    let group = MessageGroup {
        messages: messages.iter().collect(),
        indices,
        template: state.indexed(indices[0])?,
        subset_counts: indices
            .iter()
            .map(|&index| state.indexed(index).map(|indexed| indexed.num_subsets))
            .collect::<Result<_, _>>()?,
        tables: state.tables_for(indices[0])?,
    };
    decode_bounded_group(&group, options)
}

/// Data message indices of each `read_bufr` group: per table epoch in stream
/// order, data categories in `category_filter` order or first appearance, then
/// each run of one descriptor template within a category.
fn read_bufr_groups(
    state: &BufrFileState,
    category_filter: Option<&[u8]>,
) -> Result<Vec<Vec<usize>>, String> {
    let mut groups = Vec::new();
    for epoch in 0..state.epochs.len() {
        let mut messages: Vec<(usize, &IndexedDataMessage)> = Vec::new();
        for &index in &state.data_message_indices {
            if state.message_epoch[index] != epoch {
                continue;
            }
            let indexed = state.indexed(index)?;
            if category_filter.is_some_and(|filter| !filter.contains(&indexed.data_category)) {
                continue;
            }
            messages.push((index, indexed));
        }
        let mut categories: Vec<u8> = Vec::new();
        for (_, indexed) in &messages {
            if !categories.contains(&indexed.data_category) {
                categories.push(indexed.data_category);
            }
        }
        if let Some(filter) = category_filter {
            let present = categories;
            categories = Vec::new();
            for category in filter {
                if present.contains(category) && !categories.contains(category) {
                    categories.push(*category);
                }
            }
        }
        for category in categories {
            let mut template: Option<&Vec<u32>> = None;
            let mut group = Vec::new();
            for (index, indexed) in &messages {
                if indexed.data_category != category {
                    continue;
                }
                if template.is_some_and(|descriptors| descriptors != &indexed.descriptors) {
                    groups.push(std::mem::take(&mut group));
                }
                template = Some(&indexed.descriptors);
                group.push(*index);
            }
            if !group.is_empty() {
                groups.push(group);
            }
        }
    }
    Ok(groups)
}

fn without_provenance(batch: RecordBatch) -> Result<RecordBatch, String> {
    let keep: Vec<usize> = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| !PROVENANCE_COLUMNS.contains(&field.name().as_str()))
        .map(|(index, _)| index)
        .collect();
    batch
        .project(&keep)
        .map_err(|error| format!("Arrow projection error: {error}"))
}

/// Ends every error raised for a message whose declared content extends past its
/// Section 4 payload, so `skip_truncated_messages` can tell truncation from other
/// malformed input.
const SECTION4_OVERRUN_SUFFIX: &str = "-bit Section 4 payload";

fn is_section4_overrun(error: &str) -> bool {
    error.ends_with(SECTION4_OVERRUN_SUFFIX)
}

fn decode_bounded_indices_resilient(
    state: &BufrFileState,
    indices: &[usize],
    options: BoundedDecodeOptions<'_>,
    skip_truncated_messages: bool,
) -> Result<(Vec<RecordBatch>, Vec<DecodeFailure>), String> {
    let messages = state.read_messages(indices)?;
    decode_messages_resilient(state, indices, &messages, options, skip_truncated_messages)
}

fn decode_messages_resilient(
    state: &BufrFileState,
    indices: &[usize],
    messages: &[BufrMessage],
    options: BoundedDecodeOptions<'_>,
    skip_truncated_messages: bool,
) -> Result<(Vec<RecordBatch>, Vec<DecodeFailure>), String> {
    match decode_bounded_indices(state, indices, messages, options) {
        Ok(batch) => Ok((vec![batch], Vec::new())),
        Err(error)
            if skip_truncated_messages && is_section4_overrun(&error) && indices.len() > 1 =>
        {
            let middle = indices.len() / 2;
            let (mut left_batches, mut left_failures) = decode_messages_resilient(
                state,
                &indices[..middle],
                &messages[..middle],
                options,
                true,
            )?;
            let (right_batches, right_failures) = decode_messages_resilient(
                state,
                &indices[middle..],
                &messages[middle..],
                options,
                true,
            )?;
            left_batches.extend(right_batches);
            left_failures.extend(right_failures);
            Ok((left_batches, left_failures))
        }
        Err(error)
            if skip_truncated_messages && is_section4_overrun(&error) && indices.len() == 1 =>
        {
            Ok((
                Vec::new(),
                vec![DecodeFailure {
                    source_message_index: indices[0],
                    reason: error,
                }],
            ))
        }
        Err(error) if indices.len() == 1 => {
            Err(format!("Source BUFR message {}: {error}", indices[0]))
        }
        Err(error) => Err(format!(
            "Source BUFR message group {}..{}: {error}",
            indices[0],
            indices[indices.len() - 1]
        )),
    }
}

#[pymethods]
impl BufrBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyRecordBatch>> {
        // A group whose messages were all quarantined yields nothing; take the next one.
        loop {
            if let Some(batch) = self.ready_batches.pop_front() {
                return Ok(Some(PyRecordBatch::new(batch)));
            }
            if self.prefetch.is_none() {
                let pool = decode_pool().map_err(PyRuntimeError::new_err)?;
                if pool.current_num_threads() == 1 {
                    // A prefetch thread would only alternate with the one decode
                    // thread, so each group is decoded during this call, on
                    // this thread.
                    let Some(planner) = self.planner.as_mut() else {
                        return Ok(None);
                    };
                    let decoded = py.allow_threads(|| {
                        CALLER_POOL
                            .with(|caller| planner.decode_next(1, caller.as_ref().unwrap_or(&pool)))
                    });
                    if decoded.is_empty() {
                        self.planner = None;
                        return Ok(None);
                    }
                    for group in decoded {
                        let (batches, failures) = group.map_err(PyValueError::new_err)?;
                        self.ready_batches.extend(batches);
                        self.decode_failures.extend(failures);
                    }
                    continue;
                }
                let Some(planner) = self.planner.take() else {
                    return Ok(None);
                };
                let prefetch =
                    Prefetch::start(planner, self.prefetch_batches).map_err(|error| {
                        PyRuntimeError::new_err(format!(
                            "Cannot start the BUFR decode thread: {error}"
                        ))
                    })?;
                self.prefetch = Some(prefetch);
            }
            let Some(prefetch) = self.prefetch.as_ref() else {
                return Ok(None);
            };
            match py.allow_threads(|| prefetch.next_group()) {
                Some(Ok((batches, failures))) => {
                    self.ready_batches.extend(batches);
                    self.decode_failures.extend(failures);
                }
                Some(Err(error)) => return Err(PyValueError::new_err(error)),
                None => {
                    let Some(mut finished) = self.prefetch.take() else {
                        return Ok(None);
                    };
                    return py
                        .allow_threads(|| finished.stop())
                        .map(|()| None)
                        .map_err(|panic| {
                            PyRuntimeError::new_err(format!(
                                "BUFR decode thread panicked: {}",
                                panic_message(panic.as_ref())
                            ))
                        });
                }
            }
        }
    }

    fn decode_failures_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.decode_failures)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pymethods]
impl BufrFileHandle {
    /// Message headers as Arrow and DX table epochs as JSON, from the cached index.
    fn file_info(&self) -> PyResult<(PyRecordBatch, String)> {
        let state = &self.state;
        file_info(
            &state.headers,
            &state.spans,
            &state.message_epoch,
            &state.epochs,
        )
        .map(|(batch, epochs)| (PyRecordBatch::new(batch), epochs))
        .map_err(PyValueError::new_err)
    }

    /// Describe the active table/template/operator state for one data message.
    fn source_structure_json(&self, source_message_index: usize) -> PyResult<String> {
        let descriptors = self
            .state
            .indexed(source_message_index)
            .map_err(PyValueError::new_err)?
            .descriptors
            .clone();
        let tables = self
            .state
            .tables_for(source_message_index)
            .map_err(PyValueError::new_err)?;
        let mut active_sequences = FxHashMap::default();
        let mut active_elements = FxHashMap::default();
        collect_active_descriptors(
            &descriptors,
            tables,
            &mut active_sequences,
            &mut active_elements,
        )
        .map_err(PyValueError::new_err)?;

        let expanded = expand_descriptors(
            &descriptors,
            tables,
            self.state.strict,
            FixedReplicationLayout::Flat,
        )
        .map_err(PyValueError::new_err)?;
        let mut effective_occurrences = Vec::new();
        let mut mnemonic_counts = HashMap::new();
        collect_effective_occurrences(
            &expanded,
            None,
            &mut effective_occurrences,
            &mut mnemonic_counts,
        );
        let decoded_schema = derive_schema(&expanded)
            .iter()
            .enumerate()
            .map(|(position, item)| {
                let mut payload = schema_item_payload(item);
                payload["position"] = json!(position);
                payload
            })
            .collect::<Vec<_>>();
        let mut sequences = active_sequences.into_iter().collect::<Vec<_>>();
        sequences.sort_by_key(|(descriptor, _)| *descriptor);
        let mut elements = active_elements.into_iter().collect::<Vec<_>>();
        elements.sort_by_key(|(descriptor, _)| *descriptor);
        // The hash of the table epoch that decodes this message, as in
        // FileInfo.table_epochs.
        let epoch_table_sha256 = self
            .state
            .message_epoch
            .get(source_message_index)
            .and_then(|index| self.state.epochs.get(*index))
            .and_then(|epoch| epoch.embedded_table_sha256.clone());
        let payload = json!({
            "format_version": 1,
            "embedded_table_sha256": epoch_table_sha256,
            "observation_descriptors": descriptors.iter().map(|value| descriptor_to_string(*value)).collect::<Vec<_>>(),
            "active_sequences": sequences.into_iter().map(|(_, value)| value).collect::<Vec<_>>(),
            "active_elements": elements.into_iter().map(|(_, value)| value).collect::<Vec<_>>(),
            "effective_occurrences": effective_occurrences,
            "decoded_physical_schema": decoded_schema,
        });
        serde_json::to_string(&payload).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Return a native iterator over ordered, bounded Arrow record batches.
    #[allow(clippy::too_many_arguments)]
    fn iter_batches(
        &self,
        messages_per_batch: usize,
        projection: Option<Vec<String>>,
        raw_code_paths: Option<Vec<String>>,
        data_category_filter: Option<Vec<u8>>,
        list_value_filters: Option<Vec<(String, String, Vec<f64>)>>,
        prefetch_batches: usize,
        target_decoded_bytes: Option<usize>,
        fixed_replication: &str,
        skip_truncated_messages: bool,
        projection_missing: &str,
    ) -> PyResult<BufrBatchIterator> {
        let options = ParsedDecodeOptions::parse(
            projection,
            raw_code_paths,
            list_value_filters,
            fixed_replication,
            skip_truncated_messages,
            projection_missing,
        )
        .map_err(PyValueError::new_err)?;
        if messages_per_batch == 0 {
            return Err(PyValueError::new_err(
                "messages_per_batch must be greater than zero",
            ));
        }
        if prefetch_batches == 0 {
            return Err(PyValueError::new_err(
                "prefetch_batches must be greater than zero",
            ));
        }
        if target_decoded_bytes == Some(0) {
            return Err(PyValueError::new_err(
                "target_decoded_bytes must be greater than zero",
            ));
        }

        let mut message_indices = Vec::new();
        for index in &self.state.data_message_indices {
            let indexed = self.state.indexed(*index).map_err(PyValueError::new_err)?;
            if data_category_filter
                .as_ref()
                .is_some_and(|categories| !categories.contains(&indexed.data_category))
            {
                continue;
            }
            message_indices.push(*index);
        }

        let planner = BatchPlanner {
            state: Arc::clone(&self.state),
            message_indices,
            cursor: 0,
            messages_per_batch,
            target_decoded_bytes,
            estimated_template_index: None,
            estimated_bytes_per_subset: None,
            options,
        };
        Ok(BufrBatchIterator {
            planner: Some(planner),
            prefetch_batches,
            prefetch: None,
            ready_batches: VecDeque::new(),
            decode_failures: Vec::new(),
        })
    }

    /// Decode the whole file as `read_bufr` groups, one batch per group,
    /// without provenance columns.
    ///
    /// Each group is split into runs of one compression mode, and all runs
    /// decode in parallel. Messages skipped by `skip_truncated_messages` are
    /// omitted, and a group with every message skipped yields no batch.
    #[allow(clippy::too_many_arguments)]
    fn read_groups(
        &self,
        py: Python<'_>,
        data_category_filter: Option<Vec<u8>>,
        projection: Option<Vec<String>>,
        raw_code_paths: Option<Vec<String>>,
        list_value_filters: Option<Vec<(String, String, Vec<f64>)>>,
        fixed_replication: &str,
        skip_truncated_messages: bool,
        projection_missing: &str,
    ) -> PyResult<Vec<PyRecordBatch>> {
        let parsed = ParsedDecodeOptions::parse(
            projection,
            raw_code_paths,
            list_value_filters,
            fixed_replication,
            skip_truncated_messages,
            projection_missing,
        )
        .map_err(PyValueError::new_err)?;
        let state = Arc::clone(&self.state);
        let groups = read_bufr_groups(&state, data_category_filter.as_deref())
            .map_err(PyValueError::new_err)?;
        if groups.is_empty() {
            return Err(PyValueError::new_err("No matching data messages found"));
        }
        let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
        for (group_index, group) in groups.iter().enumerate() {
            let mut run: Vec<usize> = Vec::new();
            for &index in group {
                let compressed = |message: usize| {
                    state
                        .indexed(message)
                        .is_ok_and(|indexed| indexed.is_compressed)
                };
                if run
                    .last()
                    .is_some_and(|&last| compressed(last) != compressed(index))
                {
                    runs.push((group_index, std::mem::take(&mut run)));
                }
                run.push(index);
            }
            runs.push((group_index, run));
        }
        let options = parsed.bounded(state.strict);
        let decode_and_join = || -> Result<Vec<RecordBatch>, String> {
            let decoded: Vec<Result<Vec<RecordBatch>, String>> = runs
                .par_iter()
                .map(|(_, run)| {
                    let messages = state.read_messages(run)?;
                    let (batches, _) = decode_messages_resilient(
                        &state,
                        run,
                        &messages,
                        options,
                        parsed.skip_truncated_messages,
                    )?;
                    batches.into_iter().map(without_provenance).collect()
                })
                .collect();
            let mut grouped: Vec<Vec<RecordBatch>> = vec![Vec::new(); groups.len()];
            for ((group_index, _), batches) in runs.iter().zip(decoded) {
                grouped[*group_index].extend(batches?);
            }
            grouped
                .into_par_iter()
                .filter(|batches| !batches.is_empty())
                .map(|batches| {
                    if batches.len() == 1 {
                        return Ok(batches.into_iter().next().unwrap());
                    }
                    parallel_concat_batches(batches)
                        .map_err(|error| format!("Arrow concat error: {error}"))
                })
                .collect()
        };
        let pool = decode_pool().map_err(PyRuntimeError::new_err)?;
        let joined = py
            .allow_threads(|| pool.install(decode_and_join))
            .map_err(PyValueError::new_err)?;
        Ok(joined.into_iter().map(PyRecordBatch::new).collect())
    }
}

/// Open and index a BUFR file once for bounded reads, without holding the GIL.
#[pyfunction]
fn open_bufr_rust(
    py: Python<'_>,
    file_path: &str,
    table_b_json: &str,
    table_d_json: &str,
    local_tables_json: Option<&str>,
    strict: bool,
) -> PyResult<BufrFileHandle> {
    let tables = TableSet::from_json(table_b_json, table_d_json).map_err(PyValueError::new_err)?;
    let file_path = file_path.to_owned();
    let local_tables_json = local_tables_json.map(str::to_owned);
    py.allow_threads(|| open_indexed(&file_path, tables, local_tables_json.as_deref(), strict))
        .map(|state| BufrFileHandle {
            state: Arc::new(state),
        })
        .map_err(PyValueError::new_err)
}

/// Read the message headers of a file and index its data messages.
fn open_indexed(
    file_path: &str,
    tables: TableSet,
    local_tables_json: Option<&str>,
    strict: bool,
) -> Result<BufrFileState, String> {
    let read_error = |error| format!("Failed to read file: {error}");
    let mut source = FileSource::open(file_path).map_err(read_error)?;
    let (messages, spans) = read_message_headers(&mut source).map_err(read_error)?;
    if messages.is_empty() {
        return Err("No BUFR messages found in file".into());
    }

    let mut data_message_indices = Vec::new();
    let mut epochs = extract_dx_table_epochs(&messages, &tables)
        .map_err(|error| format!("Failed to extract DX table epochs: {error}"))?;
    // A file with no decodable data messages has no epochs; it opens and
    // yields no batches.
    if let Some(local_json) = local_tables_json {
        for epoch in &mut epochs {
            epoch.tables.merge_local(local_json)?;
        }
    }
    let message_epoch = message_epochs(messages.len(), &epochs);

    let mut message_index: Vec<Option<IndexedDataMessage>> = std::iter::repeat_with(|| None)
        .take(messages.len())
        .collect();
    for (index, message) in messages.iter().enumerate() {
        if quick_data_category(message) == Some(DX_TABLE_DATA_CATEGORY) {
            continue;
        }
        let parsed = parse_message_header(message, spans[index].len())
            .map_err(|error| format!("Failed to index BUFR message {index}: {error}"))?;
        if parsed.num_subsets == 0 {
            continue;
        }
        data_message_indices.push(index);
        message_index[index] = Some(IndexedDataMessage {
            data_category: parsed.data_category,
            num_subsets: parsed.num_subsets,
            is_compressed: parsed.is_compressed,
            descriptors: parsed.descriptors,
        });
    }

    if strict {
        let mut validated_templates = HashSet::new();
        for (index, message) in message_index.iter().enumerate() {
            let Some(message) = message else { continue };
            let epoch = message_epoch[index];
            if validated_templates.insert((epoch, message.descriptors.clone())) {
                let tables = epochs
                    .get(epoch)
                    .map(|epoch| &epoch.tables)
                    .ok_or_else(|| format!("BUFR message {index} has no DX table epoch"))?;
                expand_descriptors(
                    &message.descriptors,
                    tables,
                    true,
                    FixedReplicationLayout::Flat,
                )
                .map_err(|error| format!("Strict BUFR template validation failed: {error}"))?;
            }
        }
    }

    Ok(BufrFileState {
        file: source.file,
        headers: messages,
        spans,
        data_message_indices,
        message_index,
        epochs,
        message_epoch,
        strict,
    })
}

/// Message index -> index into `epochs`. `usize::MAX` for DX/empty messages.
fn message_epochs(message_count: usize, epochs: &[TableEpoch]) -> Vec<usize> {
    let mut message_epoch = vec![usize::MAX; message_count];
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        for message in &epoch.data_message_indices {
            message_epoch[*message] = epoch_index;
        }
    }
    message_epoch
}

/// Message headers, one Arrow row per message with the columns of
/// `MessageInfo`, and the table epochs as JSON.
///
/// `messages[i]` holds at least Sections 0-3 and the Section 4 header of the
/// message at `spans[i]` in the file, and DX messages whole.
fn file_info(
    messages: &[BufrMessage],
    spans: &[std::ops::Range<usize>],
    message_epoch: &[usize],
    epochs: &[TableEpoch],
) -> Result<(RecordBatch, String), String> {
    let headers = messages
        .iter()
        .zip(spans)
        .enumerate()
        .map(|(index, (message, span))| {
            parse_message_header(message, span.len())
                .map_err(|error| format!("Failed to read BUFR message {index}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut descriptors = ListBuilder::new(StringBuilder::new());
    for header in &headers {
        for descriptor in &header.descriptors {
            descriptors
                .values()
                .append_value(descriptor_to_string(*descriptor));
        }
        descriptors.append(true);
    }
    let u8s = |value: fn(&ParsedMessage<'_>) -> u8| -> ArrayRef {
        Arc::new(UInt8Array::from_iter_values(headers.iter().map(value)))
    };
    let u16s = |value: fn(&ParsedMessage<'_>) -> u16| -> ArrayRef {
        Arc::new(UInt16Array::from_iter_values(headers.iter().map(value)))
    };
    let bools = |value: fn(&ParsedMessage<'_>) -> bool| -> ArrayRef {
        Arc::new(BooleanArray::from_iter(
            headers.iter().map(|header| Some(value(header))),
        ))
    };
    let u64s = |values: Vec<u64>| -> ArrayRef { Arc::new(UInt64Array::from(values)) };
    let columns: Vec<(&str, ArrayRef)> = vec![
        (
            "source_message_index",
            u64s((0..headers.len() as u64).collect()),
        ),
        (
            "offset",
            u64s(spans.iter().map(|span| span.start as u64).collect()),
        ),
        (
            "size_bytes",
            u64s(spans.iter().map(|span| span.len() as u64).collect()),
        ),
        ("edition", u8s(|header| header.edition)),
        ("master_table", u8s(|header| header.master_table)),
        ("center", u16s(|header| header.center)),
        ("sub_center", u16s(|header| header.sub_center)),
        ("update_sequence", u8s(|header| header.update_sequence)),
        (
            "optional_section_present",
            bools(|header| header.optional_section_present),
        ),
        ("data_category", u8s(|header| header.data_category)),
        (
            "international_sub_category",
            u8s(|header| header.intl_sub_category),
        ),
        (
            "local_sub_category",
            u8s(|header| header.local_sub_category),
        ),
        (
            "master_table_version",
            u8s(|header| header.master_table_version),
        ),
        (
            "local_table_version",
            u8s(|header| header.local_table_version),
        ),
        ("year", u16s(|header| header.year)),
        ("month", u8s(|header| header.month)),
        ("day", u8s(|header| header.day)),
        ("hour", u8s(|header| header.hour)),
        ("minute", u8s(|header| header.minute)),
        ("second", u8s(|header| header.second)),
        ("subsets", u16s(|header| header.num_subsets)),
        ("compressed", bools(|header| header.is_compressed)),
        ("section3_flags", u8s(|header| header.section3_flags)),
        ("descriptors", Arc::new(descriptors.finish())),
        (
            "table_epoch",
            Arc::new(UInt64Array::from_iter(
                message_epoch
                    .iter()
                    .map(|epoch| (*epoch != usize::MAX).then_some(*epoch as u64)),
            )),
        ),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), *name == "table_epoch"))
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .map_err(|error| format!("Message header table error: {error}"))?;
    let table_epochs = serde_json::to_string(&table_epoch_report(messages, epochs)?)
        .map_err(|error| error.to_string())?;
    Ok((batch, table_epochs))
}

/// Parse a message read up to at least its Section 4 header, out of
/// `message_len` bytes in all.
fn parse_message_header(
    message: &BufrMessage,
    message_len: usize,
) -> Result<ParsedMessage<'_>, String> {
    match parse_header(message.bytes(), message_len) {
        Ok(parsed) => Ok(parsed),
        Err(HeaderError::Invalid(error)) => Err(error),
        Err(HeaderError::NeedBytes(_)) => unreachable!("the message holds its whole header"),
    }
}

/// Read each message of a file up to its Section 4 header, and DX table
/// messages whole, at the byte ranges `MessageScanner` finds.
fn read_message_headers(
    source: &mut FileSource,
) -> std::io::Result<(Vec<BufrMessage>, Vec<std::ops::Range<usize>>)> {
    let mut scanner = MessageScanner::default();
    let mut messages = Vec::new();
    let mut spans = Vec::new();
    while let Some(span) = scanner.next_span(source)? {
        let mut prefix = Vec::new();
        while let Err(HeaderError::NeedBytes(needed)) = parse_header(&prefix, span.len()) {
            let have = prefix.len();
            prefix.resize(needed, 0);
            source.read_at(span.start + have, &mut prefix[have..])?;
        }
        let mut message = BufrMessage {
            end: prefix.len(),
            buffer: prefix.into(),
            start: 0,
        };
        // Table epochs hash and decode the DX messages themselves.
        if message.end < span.len() && quick_data_category(&message) == Some(DX_TABLE_DATA_CATEGORY)
        {
            let mut whole = vec![0; span.len()];
            source.read_at(span.start, &mut whole)?;
            message = BufrMessage {
                end: whole.len(),
                buffer: whole.into(),
                start: 0,
            };
        }
        messages.push(message);
        spans.push(span);
    }
    Ok((messages, spans))
}

/// Describe a BUFR file's messages and table epochs from their headers.
///
/// Reads Sections 0-3 of each message and DX table messages whole. The JSON
/// equals `BufrFileHandle.file_info` for the same file. Table epochs
/// depend only on DX blocks, so base tables are not loaded. Runs without
/// holding the GIL.
#[pyfunction]
fn file_info_rust(py: Python<'_>, file_path: &str) -> PyResult<(PyRecordBatch, String)> {
    let file_path = file_path.to_owned();
    py.allow_threads(|| read_file_info(&file_path))
        .map(|(batch, epochs)| (PyRecordBatch::new(batch), epochs))
        .map_err(PyValueError::new_err)
}

fn read_file_info(file_path: &str) -> Result<(RecordBatch, String), String> {
    let (messages, spans) = FileSource::open(file_path)
        .and_then(|mut source| read_message_headers(&mut source))
        .map_err(|error| format!("Failed to read file: {error}"))?;
    if messages.is_empty() {
        return Err("No BUFR messages found in file".into());
    }
    let epochs = extract_dx_table_epochs(&messages, &TableSet::new())
        .map_err(|error| format!("Failed to extract DX table epochs: {error}"))?;
    let message_epoch = message_epochs(messages.len(), &epochs);
    file_info(&messages, &spans, &message_epoch, &epochs)
}

/// Report each table epoch's embedded DX hash and the descriptors its DX block
/// defines differently from earlier blocks.
///
/// Blocks are compared on what each one itself declared, not on merged epoch
/// tables, since merged tables also hold base entries a block never mentions.
fn table_epoch_report(
    messages: &[BufrMessage],
    epochs: &[TableEpoch],
) -> Result<Vec<serde_json::Value>, String> {
    // What each epoch's own DX block declared, isolated from the base tables.
    let mut declared: Vec<TableSet> = Vec::with_capacity(epochs.len());
    for epoch in epochs {
        if epoch.dx_message_indices.is_empty() {
            declared.push(TableSet::new());
        } else {
            declared.push(extract_dx_tables(messages, &epoch.dx_message_indices)?);
        }
    }

    // Declarations from earlier blocks, so a redefinition is judged against
    // everything actually in force before it, not only the adjacent epoch.
    let mut seen_b: HashMap<u32, Arc<TableBEntry>> = HashMap::new();
    let mut seen_d: HashMap<u32, Arc<TableDEntry>> = HashMap::new();
    let mut report = Vec::new();

    for (index, epoch) in epochs.iter().enumerate() {
        let block = &declared[index];
        let mut redefines = Vec::new();
        for (descriptor, entry) in &block.table_b {
            let Some(previous) = seen_b.get(descriptor) else {
                continue;
            };
            // Units and name carry no bits but are part of what a block declares.
            let unchanged = previous.bit_width == entry.bit_width
                && previous.scale == entry.scale
                && previous.reference_value == entry.reference_value
                && previous.mnemonic == entry.mnemonic
                && previous.units == entry.units
                && previous.name == entry.name;
            if unchanged {
                continue;
            }
            let definition = |entry: &TableBEntry| {
                json!({"mnemonic": entry.mnemonic, "name": entry.name, "units": entry.units,
                       "scale": entry.scale, "reference_value": entry.reference_value,
                       "bit_width": entry.bit_width})
            };
            redefines.push(json!({
                "descriptor": descriptor_to_string(*descriptor),
                "table": "B",
                "before": definition(previous),
                "after": definition(entry),
            }));
        }
        for (descriptor, entry) in &block.table_d {
            let Some(previous) = seen_d.get(descriptor) else {
                continue;
            };
            if previous.descriptors == entry.descriptors && previous.mnemonic == entry.mnemonic {
                continue;
            }
            let definition = |entry: &TableDEntry| {
                let descriptors: Vec<String> = entry
                    .descriptors
                    .iter()
                    .map(|value| descriptor_to_string(*value))
                    .collect();
                json!({"mnemonic": entry.mnemonic, "descriptors": descriptors})
            };
            redefines.push(json!({
                "descriptor": descriptor_to_string(*descriptor),
                "table": "D",
                "before": definition(previous),
                "after": definition(entry),
            }));
        }
        for (descriptor, entry) in &block.table_b {
            seen_b.insert(*descriptor, Arc::clone(entry));
        }
        for (descriptor, entry) in &block.table_d {
            seen_d.insert(*descriptor, Arc::clone(entry));
        }

        redefines.sort_by(|left, right| {
            left["descriptor"]
                .as_str()
                .cmp(&right["descriptor"].as_str())
        });
        report.push(json!({
            "embedded_table_sha256": epoch.embedded_table_sha256,
            "redefines": redefines,
        }));
    }

    Ok(report)
}

/// Replace the decode pool with one of `threads` threads.
#[pyfunction]
fn set_num_threads_rust(threads: usize) -> PyResult<()> {
    if threads == 0 {
        return Err(PyValueError::new_err(
            "The number of threads must be at least 1",
        ));
    }
    let pool = build_decode_pool(threads).map_err(PyRuntimeError::new_err)?;
    *DECODE_POOL.write().unwrap_or_else(PoisonError::into_inner) = Some(pool);
    Ok(())
}

/// Thread count of the current decode pool.
#[pyfunction]
fn get_num_threads_rust() -> PyResult<usize> {
    Ok(decode_pool()
        .map_err(PyRuntimeError::new_err)?
        .current_num_threads())
}

/// One message table value as a Python object: `None` for null, `int`,
/// `bool`, or a tuple of `str` for a list of strings.
fn message_table_value(py: Python<'_>, column: &ArrayRef, row: usize) -> PyResult<PyObject> {
    if column.is_null(row) {
        return Ok(py.None());
    }
    match column.data_type() {
        DataType::Boolean => column.as_boolean().value(row).into_py_any(py),
        DataType::UInt8 => column
            .as_primitive::<UInt8Type>()
            .value(row)
            .into_py_any(py),
        DataType::UInt16 => column
            .as_primitive::<UInt16Type>()
            .value(row)
            .into_py_any(py),
        DataType::UInt64 => column
            .as_primitive::<UInt64Type>()
            .value(row)
            .into_py_any(py),
        DataType::List(_) => {
            let values = column.as_list::<i32>().value(row);
            let strings = values.as_string::<i32>();
            PyTuple::new(py, strings.iter().map(Option::unwrap_or_default))?.into_py_any(py)
        }
        other => Err(PyValueError::new_err(format!(
            "Unsupported message table column type {other}"
        ))),
    }
}

/// Rows of a message table as tuples of Python values in column order.
#[pyfunction]
fn message_rows_rust(py: Python<'_>, batch: PyRecordBatch) -> PyResult<Vec<Py<PyTuple>>> {
    let batch = batch.into_inner();
    (0..batch.num_rows())
        .map(|row| {
            let values = batch
                .columns()
                .iter()
                .map(|column| message_table_value(py, column, row))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, values)?.unbind())
        })
        .collect()
}

/// Flatten one decoded batch with the same rules as `read_bufr(flatten=True)`.
#[pyfunction]
fn flatten_batch_rust(batch: PyRecordBatch) -> PyResult<PyRecordBatch> {
    flatten_record_batch(batch.into_inner())
        .map(PyRecordBatch::new)
        .map_err(PyValueError::new_err)
}

/// Python module definition.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flatten_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(open_bufr_rust, m)?)?;
    m.add_function(wrap_pyfunction!(file_info_rust, m)?)?;
    m.add_function(wrap_pyfunction!(message_rows_rust, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads_rust, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads_rust, m)?)?;
    m.add_class::<BufrFileHandle>()?;
    m.add_class::<BufrBatchIterator>()?;
    Ok(())
}
