# Earth2bufr Examples

This directory contains examples demonstrating how to use earth2bufr to read
different BUFR file formats.

## Examples

### 01_gdas_prepbufr.py - NOAA GDAS PrepBUFR Files

Demonstrates reading PrepBUFR files from NOAA's Global Data Assimilation System
(GDAS). PrepBUFR is an NCEP extension that embeds local descriptor tables in
the file itself.

```bash
uv run python examples/01_gdas_prepbufr.py
```

**Key functions:**

- `read_bufr()` - Reads the embedded DX-tables and decodes data
- `get_metadata()` - Message headers and table epochs without decoding data

**Data categories in GDAS PrepBUFR:**

| Code | Name   | Description                           |
|------|--------|---------------------------------------|
| 102  | ADPUPA | Upper air (radiosondes, pilot ballon) |
| 104  | AIRCFT | Aircraft reports                      |
| 105  | SATWND | Satellite-derived winds               |
| 107  | VADWND | VAD/NEXRAD radar winds                |
| 109  | ADPSFC | Surface land observations             |
| 110  | SFCSHP | Surface marine/ship observations      |
| 112  | GPSIPW | GPS precipitable water                |

### 02_noaa_nucaps.py - NOAA NUCAPS Atmospheric Soundings

Demonstrates reading NUCAPS (NOAA Unique Combined Atmospheric Processing
System) BUFR files. NUCAPS produces retrieved temperature and moisture profiles
from CrIS and ATMS measurements on the Suomi NPP and NOAA-20/21 satellites.

```bash
uv run python examples/02_noaa_nucaps.py
```

**Key functions:**

- `read_bufr()` - Read standard WMO BUFR files
- `get_metadata()` - Inspect message structure without full decode
- `pyarrow.compute.list_parent_indices` - One row per vertical level, in Arrow

**NUCAPS data characteristics:**

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Vertical levels   | 2211 per sounding                        |
| Format            | WMO BUFR Edition 4, compressed           |
| Satellites        | Suomi NPP, NOAA-20, NOAA-21             |
| S3 Bucket         | `noaa-nesdis-snpp-pds`                   |
| Path              | `NUCAPS_C2211_BUFR/{YYYY}/{MM}/{DD}/*.bufr`|

### 03_noaa_jpss_atms.py - NOAA JPSS ATMS Satellite Radiances

Demonstrates reading Advanced Technology Microwave Sounder (ATMS) BUFR files
from NOAA's Joint Polar Satellite System. ATMS is a 22-channel microwave
radiometer used for atmospheric temperature and moisture profiling.

```bash
uv run python examples/03_noaa_jpss_atms.py
```

**Key functions:**

- `read_bufr()` - Read standard WMO BUFR files (ATMS uses compressed BUFR)
- `get_metadata()` - Inspect message structure without full decode
- `pyarrow.compute.list_parent_indices` - One row per channel, in Arrow

**ATMS data characteristics:**

| Property          | Value                              |
|-------------------|------------------------------------|
| Channels          | 22 (50-183 GHz)                    |
| Format            | WMO BUFR Edition 4, compressed     |
| Satellites        | NOAA-20, NOAA-21, Suomi NPP       |
| S3 Bucket         | `noaa-nesdis-n20-pds`              |
| Path              | `ATMS_BUFR/{YYYY}/{MM}/{DD}/*.bufr`|

## Data Sources

### NOAA PrepBUFR

- **AWS S3**: `s3://noaa-reanalyses-pds/observations/reanalysis/conv/prepbufr/`
- **NOMADS**: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/`

### NOAA Satellite (JPSS / Suomi NPP)

- **ATMS BUFR**: `s3://noaa-nesdis-n20-pds/ATMS_BUFR/{YYYY}/{MM}/{DD}/`
- **NUCAPS BUFR**: `s3://noaa-nesdis-snpp-pds/NUCAPS_C2211_BUFR/{YYYY}/{MM}/{DD}/`
- All NOAA S3 buckets are public (anonymous access)
- Satellites: `noaa-nesdis-n20-pds` (NOAA-20), `noaa-nesdis-n21-pds` (NOAA-21), `noaa-nesdis-snpp-pds` (Suomi NPP)

## PrepBUFR and Standard BUFR

`read_bufr` reads both. Embedded DX-tables (`dataCategory=11` messages) are
applied to the data messages that follow them; other descriptors resolve
against the WMO tables:

```python
import earth2bufr

batches = earth2bufr.read_bufr("gdas.prepbufr")
batches = earth2bufr.read_bufr("myfile.bufr")
```

| Feature              | Standard BUFR      | PrepBUFR (NCEP)         |
|----------------------|--------------------|-------------------------|
| Tables               | WMO standard       | Embedded in file (DX)   |
| Local descriptors    | Separate files     | In dataCategory=11 msgs |
| Data categories      | WMO codes (0-6)    | NCEP codes (102-121)    |
