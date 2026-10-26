<!-- markdownlint-disable MD013 -->

# BUFR Format Reference

This page provides reference information for the BUFR (Binary Universal Form
for the Representation of meteorological data) format as relevant to
earth2bufr.

## BUFR Editions

earth2bufr supports BUFR Edition 3 and Edition 4. The key differences:

| Feature | Edition 3 | Edition 4 |
|---------|-----------|-----------|
| Year field | 2-digit (YY) | 4-digit (YYYY) |
| Sub-category | Local only | International + local |
| Seconds | Not present | Present |
| Section 2 | Optional | Optional |

## BUFR Message Structure

Every BUFR message contains five (optionally six) sections:

| Section | Name | Contents |
|---------|------|----------|
| 0 | Indicator | "BUFR" marker, total message length, edition number |
| 1 | Identification | Originating centre, data category, date/time, table versions |
| 2 | Optional | Local-use data (centre-specific) |
| 3 | Data Description | Unexpanded descriptor list, number of subsets, compression flag |
| 4 | Data | Encoded observation values |
| 5 | End | "7777" marker |

## Descriptor Types

BUFR descriptors are 16-bit values encoded as F-XX-YYY where F (2 bits),
X (6 bits), and Y (8 bits) determine the type:

| F | Type | Purpose |
|---|------|---------|
| 0 | Table B (Element) | Single data element (e.g., temperature, pressure) |
| 1 | Replication | Repeat a group of descriptors N times |
| 2 | Operator | Modify encoding rules (change width, scale, etc.) |
| 3 | Table D (Sequence) | Macro expanding to a list of descriptors |

### Table B Elements

Table B defines individual data elements. Each entry specifies:

- **Mnemonic** -- Short name, which names the decoded column (e.g., `B012101`, or `TOB` in NCEP DX tables)
- **Units** -- Physical units or `CCITT IA5` for strings, `Code table` /
  `Flag table` for categorical data
- **Scale** -- Decimal scaling factor (value = (raw + reference) / 10^scale)
- **Reference value** -- Offset added to the raw value before scaling
- **Bit width** -- Number of bits used to encode the value

The full WMO Table B is defined in the [WMO Manual on Codes, Vol. I.2](https://library.wmo.int/records/item/35713-manual-on-codes-volume-i-1-international-codes),
also available on [Github](https://github.com/wmo-im/BUFR4).

### Table D Sequences

Table D defines reusable sequences of descriptors. For example, a common
sequence might expand latitude + longitude + station height into a single
descriptor for convenience.

The full WMO Table D is maintained alongside Table B in the WMO Manual on
Codes, available on
[Github](https://github.com/wmo-im/BUFR4).

### Replication (F=1)

Replication descriptors repeat a group of descriptors:

- **Fixed replication** (Y > 0): Repeat the next X descriptors Y times
- **Delayed replication** (Y = 0): The repeat count is encoded in the data
  stream itself (read from the next descriptor)

### Operator Descriptors (F=2)

Operator descriptors modify how subsequent elements are encoded:

| Operator | Name | Effect |
|----------|------|--------|
| 2-01-YYY | Change data width | Add (YYY - 128) bits to subsequent numeric widths; CCITT IA5, code, and flag tables are exempt |
| 2-02-YYY | Change scale | Add (YYY - 128) to subsequent non-code numeric scales; code and flag tables are not rescaled |
| 2-03-YYY | Change reference value | Following values define new reference values |
| 2-04-YYY | Associated field | Add YYY bits of associated data before each element |
| 2-05-YYY | Text string | Read YYY*8 bits as character data |
| 2-06-YYY | Signify local descriptor | Next descriptor uses YYY bits |
| 2-07-YYY | Increase scale and width | For ordinary numeric data, scale += YYY, reference *= 10^YYY, and width += floor((10*YYY + 2)/3); CCITT IA5, code, and flag tables are exempt |
| 2-08-YYY | Change string width | Override CCITT IA5 string width to YYY*8 bits |

## WMO Data Categories

The `data_category` field in Section 1 identifies the observation type.
Standard WMO categories (defined in
[WMO Common Code Table C-13](https://github.com/wmo-im/CCT/blob/master/C13.csv)
and [WMO Manual on Codes, Vol. I.2](https://library.wmo.int/records/item/35713-manual-on-codes-volume-i-1-international-codes)):

| Code | Category |
|------|----------|
| 0 | Surface data -- land |
| 1 | Surface data -- sea |
| 2 | Vertical soundings (other than satellite) |
| 3 | Vertical soundings (satellite) |
| 4 | Single-level upper-air (other than satellite) |
| 5 | Single-level upper-air (satellite) |
| 6 | Radar data |
| 7 | Synoptic features |
| 8 | Physical/chemical constituents |
| 9 | Dispersal and transport |
| 10 | Radiological data |
| 11 | BUFR tables (including DX-tables) |
| 12 | Surface data -- sea (waves) |
| 13 | Forecasts |
| 14 | National/bilateral arrangements (non-standard) |
| 15-19 | Reserved |
| 20 | Status information |
| 21-30 | Reserved |
| 31 | Oceanographic data |

## PrepBUFR Format

PrepBUFR is an NCEP-specific extension of BUFR used for quality-controlled
observational data in the NCEP Global Data Assimilation System (GDAS).

### Key Differences from Standard BUFR

1. **Embedded DX-tables** -- Local Table B and Table D definitions are stored
   as BUFR messages with `dataCategory=11` before the data messages they
   govern; a file can hold several DX blocks
2. **NCEP-local descriptors** -- Uses mnemonics like `SID`, `XOB`, `YOB`,
   `TOB`, `POB` instead of standard WMO names
3. **Operator 2-07** -- Enhanced precision fields (e.g., longitude/latitude
   stored with extra scale and width)

### PrepBUFR Data Categories

PrepBUFR files use NCEP-specific data category codes. The table below shows
the **message-level codes** (values that appear in the BUFR Section 1 header
and in the `_data_category` column) alongside the NCEP logical type codes.
Use the message-level codes with `data_category_filter`; `get_metadata(path).data_messages` lists each message's category.

#### Message-Level Codes (Section 1 `dataCategory`)

These are the values you will see in the file and pass to
`data_category_filter`:

| Code | Mnemonic | Description |
|------|----------|-------------|
| 102 | ADPUPA | Upper-air (radiosondes, pilot balloons, dropsondes) |
| 103 | PROFLR | Wind profiler / acoustic sounder |
| 104 | AIRCFT | Conventional aircraft (AIREP, PIREP, AMDAR, TAMDAR) |
| 105 | SATWND | Satellite-derived winds |
| 107 | VADWND | VAD (NEXRAD) winds |
| 109 | ADPSFC | Surface land (synoptic, METAR, ASOS) |
| 110 | SFCSHP | Surface marine (ship, buoy, C-MAN, tide gauge) |
| 112 | GPSIPW | GPS-IPW (integrated precipitable water) |
| 113 | SYNDAT | Synthetic tropical cyclone bogus |
| 119 | RASSDA | RASS virtual temperature profiles |
| 121 | ASCATW | ASCAT scatterometer winds |

#### NCEP Logical Type Codes

The NCEP documentation also defines a separate "logical" numbering (0-19)
used internally by the PrepBUFR processing system. These do **not** appear in
the BUFR message headers but are documented here for cross-reference with
NCEP literature:

| Code | Mnemonic | Description |
|------|----------|-------------|
| 0 | ADPUPA | Upper-air (radiosondes, pilot balloons, dropsondes) |
| 1 | AIRCAR | MDCRS ACARS aircraft |
| 2 | AIRCFT | Conventional aircraft (AMDAR, etc.) |
| 3 | SATWND | Satellite-derived winds |
| 4 | PROFLR | Wind profilers (NOAA, JMA) |
| 5 | VADWND | VAD winds (NEXRAD) |
| 6 | SATEMP | TOVS/ATOVS sounding / satellite radiance |
| 7 | ADPSFC | Surface land (METAR, SYNOP, CMAN) |
| 8 | SFCSHP | Surface marine (ships, buoys) |
| 9 | SFCBOG | Surface bogus |
| 10 | SPSSMI | SSM/I retrievals |
| 11 | SYNDAT | Synthetic tropical cyclone bogus |
| 12 | ERS1DA | ERS scatterometer |
| 13 | GOESND | GOES sounder/radiance |
| 14 | QKSWND | QuikSCAT |
| 15 | MSONET | Mesonets |
| 16 | GPSIPW | GPS-derived integrated precipitable water |
| 17 | RASSDA | RASS virtual temperature |
| 18 | WDSATR | WindSat |
| 19 | ASCATW | ASCAT scatterometer winds |

:::{tip}
Use `earth2bufr.get_metadata` to discover which categories are present in a
specific file without decoding any data:

```python
info = earth2bufr.get_metadata("gdas.prepbufr")
print(sorted({message.data_category for message in info.data_messages}))
```

:::

Sources for PrepBUFR category definitions:

- [NCEPLIBS-bufr PrepBUFR table](https://github.com/NOAA-EMC/NCEPLIBS-bufr/tree/develop/tables)
- [PrepBUFR Processing at NCEP (PDF), Table 1](https://dtcenter.org/sites/default/files/community-code/gsi/downloads/BUFR/BUFR_PrepBUFR_User_Guide_v1.pdf)
- [GSI User's Guide, Chapter 3](https://dtcenter.org/sites/default/files/community-code/gsi/docs/users-guide/GSIUserGuide_v3.7.pdf)

### DX-Table Message Format

The Section 4 data of a DX-table message (`dataCategory=11`) holds local
Table A, B and D definitions as fixed-width CCITT IA5 records:

```text
n_a (1 byte), n_a x 67-byte Table A records,
n_b (1 byte), n_b x 112-byte Table B records,
n_d (1 byte), Table D records (descriptor, mnemonic, description, member count, FXXYYY members)
```

- **Table A**: category, mnemonic and description per record -- usually empty
- **Table B**: descriptor, mnemonic, name, units, scale, reference and width
- **Table D**: a sequence descriptor and its member descriptors

earth2bufr extracts these tables entirely in Rust -- no external Python
dependencies are needed.

### Common PrepBUFR Mnemonics

| Mnemonic | Description | Units |
|----------|-------------|-------|
| SID | Station identifier | CCITT IA5 |
| XOB | Longitude | Degrees East |
| YOB | Latitude | Degrees North |
| DHR | Observation time minus cycle time | Hours |
| ELV | Station elevation | Meters |
| TYP | Report type | Code table |
| POB | Pressure observation | mb (hPa) |
| TOB | Temperature observation | Degrees C |
| QOB | Specific humidity | mg/kg |
| ZOB | Height observation | Meters |
| UOB | U-component of wind | m/s |
| VOB | V-component of wind | m/s |
| PWO | Precipitable water observation | mm |
| CAT | Data category | Code table |

## External References

### WMO Standards

- [WMO Manual on Codes (WMO-306)](https://library.wmo.int/records/item/35713-manual-on-codes-volume-i-1-international-codes)
  -- Authoritative source for BUFR Table B, Table D, and code/flag tables
- [WMO BUFR/CREX Tables (CSV)](https://github.com/wmo-im/BUFR4/tree/master)
  -- Machine-readable Tables from WMO GitHub
- [WMO Common Code Tables](https://github.com/wmo-im/CCT)
  -- Data categories, originating centres, and other common code tables

### NCEP/NOAA PrepBUFR

- [NCEPLIBS-bufr Tables](https://github.com/NOAA-EMC/NCEPLIBS-bufr/tree/develop/tables)
  -- NCEP-maintained local BUFR tables used in PrepBUFR files
- [NCEPLIBS-bufr Documentation](https://noaa-emc.github.io/NCEPLIBS-bufr/)
  -- The library that reads/writes PrepBUFR (successor to legacy BUFRLIB)
- [PrepBUFR Processing at NCEP (PDF)](https://dtcenter.org/sites/default/files/community-code/gsi/downloads/BUFR/BUFR_PrepBUFR_User_Guide_v1.pdf)
  -- Overview of the PrepBUFR observation processing system
- [NNJA (NOAA-NASA Joint Archive)](https://psl.noaa.gov/data/nnja_obs/)
  -- Archive for GDAS PrepBUFR files
- [NOMADS PrepBUFR](https://nomads.ncep.noaa.gov/)
  -- Real-time access to PrepBUFR from NCEP operational runs

### EUMETSAT EPS-SG / MTG

- [EPS-SG Test Data in BUFR Format](https://user.eumetsat.int/news-events/news/eps-sg-test-data-in-bufr-format-1)
  -- EUMETSAT page with links to EPS-SG and MTG test BUFR data files
- [README: BUFR Data Description (Aug 2025, PDF)](https://user.eumetsat.int/s3/ope-eup-strapi-media/Readme_BUFR_data_description_Aug_2025_db27f55057.pdf)
  -- EUMETSAT readme describing the BUFR data format for EPS-SG products
- [EUMETSAT BUFR Vocabulary Manager](https://vocabulary-manager.eumetsat.int/vocabularies/BUFR)
  -- EUMETSAT vocabulary/table browser for BUFR descriptors and code tables

### ECMWF BUFR

- [ECMWF BUFR Software (ecCodes)](https://confluence.ecmwf.int/display/ECC)
  -- ECMWF's BUFR/GRIB encoding/decoding library
- [ECMWF BUFR Users Guide](https://www.ecmwf.int/sites/default/files/elibrary/2008/80533-bufr-users-guide_0.pdf)
  -- ECMWF's BUFR guide
- [ECCodes Github BUFR Examples](https://github.com/ecmwf/eccodes/tree/develop/examples/python)
