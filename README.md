# ubx-mga-gnss-rinex-ephemeris-converter

Converts RINEX navigation files to u-blox 8 / M8 UBX-MGA ephemeris messages (GPS, QZSS, GLONASS) and provides a serial tool for sending assistance data to u-blox receivers.

Provides two tools:

- **convert_eph.py** — Converts RINEX 2/3 navigation files to UBX binary files containing MGA-GPS-EPH, MGA-QZSS-EPH, MGA-GLO-EPH, and supplementary messages (health, ionosphere, UTC, time offset)
- **ubx_tool.py** — Serial tool for interacting with u-blox receivers: send MGA messages, reset, measure TTFF, poll status, dump navigation database

## Background

u-blox 8 / M8 receivers support Multiple GNSS Assistance (MGA) for injecting ephemeris, almanac, time, and position data to reduce Time To First Fix. This tool provides an offline method for generating MGA ephemeris messages from publicly available RINEX broadcast navigation data.

By injecting current ephemeris data, the receiver can skip downloading navigation data from satellites (which takes 12.5+ minutes for a full almanac cycle) and achieve a fix within seconds — particularly useful for receivers with limited sky visibility.

### Related project

[ublox8-gps-qzss-yuma-almanac-converter](https://github.com/jkivilin/ublox8-gps-qzss-yuma-almanac-converter) — Converts YUMA almanac files to UBX-MGA-GPS-ALM and UBX-MGA-QZSS-ALM messages.

## Requirements

Python 3.10+

```
pip install -r requirements.txt
```

### convert_eph.py dependencies

- [georinex](https://github.com/geospace-code/georinex) — RINEX file parser
- numpy

### ubx_tool.py dependencies

- [pyubx2](https://github.com/semuconsulting/pyubx2) — UBX protocol library
- [pyserial](https://github.com/pyserial/pyserial) — Serial port access

## convert_eph.py

Converts RINEX navigation files to UBX-MGA messages.

### Supported input formats

- RINEX 2 GPS navigation (`.XXn`) — tested with QZSS site `brdc*.26n`
- RINEX 2 QZSS navigation (`.XXq`) — tested with QZSS site `brdc*.26q`
- RINEX 3 mixed multi-GNSS navigation (`.rnx`) — tested with IGS/BKG `BRDC00WRD_R_*.rnx`

The tool includes its own RINEX 2 and RINEX 3 parser as a fallback, since georinex cannot decode some RINEX files (e.g. certain QZSS RINEX 2 files and RINEX 3 files with mixed GNSS systems). The built-in parser is used automatically when georinex fails or when the file contains systems that georinex does not handle correctly.

### Generated UBX messages

| Message | Class/ID | Description |
|---------|----------|-------------|
| MGA-GPS-EPH | 0x13 0x00 | GPS broadcast ephemeris (per satellite) |
| MGA-QZSS-EPH | 0x13 0x05 | QZSS broadcast ephemeris (per satellite) |
| MGA-GLO-EPH | 0x13 0x06 | GLONASS broadcast ephemeris (per satellite) |
| MGA-GPS-HEALTH | 0x13 0x00 | GPS satellite health status |
| MGA-QZSS-HEALTH | 0x13 0x05 | QZSS satellite health status |
| MGA-GPS-IONO | 0x13 0x00 | GPS ionosphere model parameters (Klobuchar) |
| MGA-GPS-UTC | 0x13 0x00 | GPS-UTC time correction |
| MGA-GLO-TIMEOFFSET | 0x13 0x06 | GLONASS-UTC/GPS time correction |

### Usage

```bash
# GPS only
convert_eph.py brdc0380.26n -o eph.ubx

# GPS + QZSS from separate files
convert_eph.py brdc0380.26n brdc0380.26q -o eph.ubx

# Mixed RINEX 3 (GPS + QZSS + GLONASS)
convert_eph.py BRDC00WRD_R_20260390000_01D_MN.rnx -o eph.ubx

# Select specific target time (picks closest epoch per satellite)
convert_eph.py brdc0380.26n -o eph.ubx --time 2026-02-07T16:00

# GPS and GLONASS only (skip QZSS)
convert_eph.py brdc0380.26n -o eph.ubx --systems GPS,GLO

# Verbose output with per-satellite details
convert_eph.py brdc0380.26n brdc0380.26q -o eph.ubx -v
```

### Epoch selection

By default, the tool selects the most recent ephemeris epoch per satellite (within 4 hours for GPS/QZSS, 1 hour for GLONASS). Use `--time` to select epochs closest to a specific time, or `--all-epochs` to include all available epochs.

### RINEX data sources

- **IGS combined broadcast navigation (BKG)**: [igs.bkg.bund.de/root_ftp/IGS/BRDC/](https://igs.bkg.bund.de/root_ftp/IGS/BRDC/) — RINEX 3 mixed navigation files (`BRDC00WRD_R_*.rnx.gz`) containing GPS, GLONASS, Galileo, BeiDou, QZSS, and SBAS, updated multiple times daily
- **QZSS official**: [sys.qzss.go.jp](https://sys.qzss.go.jp/dod/en/archives/pnt.html) — GPS+QZSS RINEX 2 navigation data (`brdc*.26n` + `brdc*.26q`), updated once per day

## ubx_tool.py

Serial/TCP tool for u-blox 8 / M8 receivers.

Supports direct serial connection and TCP connection through a serial-mux server that bridges TCP clients to the receiver's UART.

### Commands

```bash
# Poll receiver status
ubx_tool.py /dev/ttyS2 poll NAV-SAT      # Satellite tracking status
ubx_tool.py /dev/ttyS2 poll NAV-PVT      # Position/velocity/time
ubx_tool.py /dev/ttyS2 poll NAV-ORB      # Orbit database status

# Send assistance data to receiver
ubx_tool.py /dev/ttyS2 send eph.ubx -v              # Send UBX messages
ubx_tool.py /dev/ttyS2 send eph.ubx --assist --lat 51.50 --lon -0.13 -v  # Prepend time + position

# Reset receiver
ubx_tool.py /dev/ttyS2 reset cold    # Clear all nav data (BBR)
ubx_tool.py /dev/ttyS2 reset warm    # Clear ephemeris only
ubx_tool.py /dev/ttyS2 reset hot     # Keep everything

# Measure Time To First Fix
ubx_tool.py /dev/ttyS2 ttff                # Wait for any fix
ubx_tool.py /dev/ttyS2 ttff --3d           # Wait for 3D fix
ubx_tool.py /dev/ttyS2 ttff --timeout 300  # Custom timeout

# Dump navigation database
ubx_tool.py /dev/ttyS2 dump-dbd            # Dump to stderr (hex)
ubx_tool.py /dev/ttyS2 dump-dbd nav.ubx    # Dump to file

# Dump ephemeris from receiver via subframe polling
ubx_tool.py /dev/ttyS2 dump-eph            # Poll and decode GPS subframes

# Monitor receiver output
ubx_tool.py /dev/ttyS2 monitor             # Continuous output
ubx_tool.py /dev/ttyS2 monitor -d 60       # Monitor for 60 seconds

# Enable UBX protocol output
ubx_tool.py /dev/ttyS2 enable-ubx

# Parse UBX binary file (offline, no serial port needed)
ubx_tool.py - parse input.ubx
ubx_tool.py - decode-eph input.ubx         # Decode MGA-GPS-EPH messages
```

### TCP mode

Use `--tcp` to connect through a TCP serial-mux server instead of a local serial port. The mux server bridges multiple TCP clients to the receiver's UART — data from the receiver is broadcast to all clients, and data from clients is written to the serial port.

```bash
# Same commands, just add --tcp and use host:port instead of serial device
ubx_tool.py --tcp localhost:4000 poll NAV-SAT
ubx_tool.py --tcp localhost:4000 send eph.ubx --assist --lat 51.50 --lon -0.13 -v
ubx_tool.py --tcp localhost:4000 dump-dbd backup.ubx
ubx_tool.py --tcp localhost:4000 ttff --3d
ubx_tool.py --tcp 192.168.1.10:4000 monitor --duration 60
ubx_tool.py --tcp localhost:4000 enable-ubx
```

Since the mux server is not UBX-aware and manages the physical UART, the `reset` command is blocked in TCP mode (it changes UART baud rate and triggers hardware reset). All other commands work normally. The CFG-PRT message (used by `enable-ubx` and sent automatically before most commands) always uses 115200 baud in TCP mode regardless of `--baud`.

### TTFF test workflow

```bash
# 1. Cold reset
ubx_tool.py /dev/ttyS2 reset cold

# 2. Inject assistance data (prepends current time + position)
ubx_tool.py /dev/ttyS2 send eph.ubx --assist --lat 51.50 --lon -0.13 -v

# 3. Measure time to 3D fix
ubx_tool.py /dev/ttyS2 ttff --3d
```

### Serial configuration

Default baud rate is 115200. Use `--baud` to change:

```bash
ubx_tool.py /dev/ttyS2 poll NAV-SAT --baud 9600
```

The tool automatically configures the receiver for UBX+NMEA output on the serial port, and handles baud rate recovery after hardware resets.

## Example output

### convert_eph.py

```
$ python3 convert_eph.py BRDC00WRD_R_20260390000_01D_MN.rnx -o eph.ubx -v
  G01 2026-02-08T04:00:00 IODC=   47 Toe= 14400 health=  0 sqrtA=5153.658 URA=2.0m(idx=1)
  G02 2026-02-08T04:00:00 IODC=   44 Toe= 14400 health=  0 sqrtA=5153.618 URA=2.0m(idx=1)
  ...
  J01 2026-02-08T03:55:44 IODC=    8 Toe= 14144 health=  0 sqrtA=6493.428 URA=2.8m(idx=3)
  ...
  R01 2026-02-08T05:15:00    X=  -3497.734  Y= -22805.977  Z=  12175.361  -TauN= 1.3783574e-04
  ...

Converted 69 messages for 64 satellites (4472 bytes) to eph.ubx
  Including: GPS-HEALTH, QZSS-HEALTH, GPS-IONO, GPS-UTC, GLO-TIMEOFFSET
```

### ubx_tool.py TTFF test

```
$ python3 ubx_tool.py /dev/ttyS2 ttff --3d
Waiting for 3D fix (timeout 120s)...
  27.3s: 3D fix,  5 SVs, hAcc=28.3m
  28.3s: 3D fix, 10 SVs, hAcc=12.1m
  30.0s: 3D fix, 14 SVs, hAcc=7.1m
TTFF: 27.3s (3D fix, 14 SVs, hAcc=7.1m)
```

## Known issues and notes

- **pyubx2 `X` type handling**: pyubx2 requires `X2`/`X4` bitfield attributes to be passed as `bytes` objects rather than integers in SET messages. For messages using bitfield types (e.g. CFG-RST `navBbrMask`), `ubx_tool.py` constructs the message manually with `struct.pack` to avoid this limitation.
- **u-blox documentation errors**: Some MGA message field descriptions in the u-blox 8 receiver description contain errors that were identified and corrected during development by cross-validating against pyubx2 source code and the receiver's actual behavior.
- **georinex limitations**: georinex cannot decode some RINEX files, including certain QZSS RINEX 2 navigation files and RINEX 3 mixed navigation files with GLONASS records. The tool includes built-in RINEX 2 and RINEX 3 parsers that are used automatically as a fallback when georinex fails.
- **RINEX field naming**: georinex uses slightly different field names than standard RINEX documentation. The parser handles both RINEX 2 and 3 naming conventions.
- **GLONASS ephemeris validity**: GLONASS ephemeris has a shorter validity period (~30 min) than GPS (~4 hours). The epoch selection defaults to 1 hour max age for GLONASS.

## Testing

The project includes a comprehensive test suite with 253 tests covering both tools.

### Setup

Install test dependencies alongside the main requirements:

```bash
pip install -r requirements.txt
pip install pytest
```

### Running tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests for a single tool
pytest test_convert_eph.py
pytest test_ubx_tool.py

# Run a specific test class
pytest test_ubx_tool.py::TestTcpStream
pytest test_convert_eph.py::TestRinex2Parser
```

No hardware or network access is required — all tests run offline using constructed test data, real RINEX records embedded in the test files, and loopback TCP sockets.

### Test coverage

**test_convert_eph.py** (150 tests):

- RINEX 2 GPS and QZSS record parsing (real satellite data from `brdc` files)
- RINEX 3 multi-GNSS record parsing
- RINEX header parsing (ionosphere, UTC, time system corrections)
- GPS time utilities (week number, day-of-year, epoch calculations)
- UBX framing and checksum computation
- URA index lookup table
- Signed/unsigned integer and angular scaling functions
- MGA-GPS-EPH and MGA-QZSS-EPH message builders
- MGA-GLO-EPH message builder (ECEF state vector encoding, tau negation)
- MGA-GPS-HEALTH, MGA-GPS-IONO, MGA-GPS-UTC message builders
- GLONASS day number computation
- Epoch selection logic (freshest, target time, max age filtering)
- Integration tests with real RINEX 2 and RINEX 3 zip/gzip archives

**test_ubx_tool.py** (103 tests):

- IS-GPS-200 24-bit word bit extraction
- Two's complement signed integer conversion (8/14/16/22/32-bit)
- Full GPS subframe 1/2/3 decoding (clock, orbit, health parameters)
- URA meters-to-index conversion
- MGA-GPS-EPH payload construction and field encoding
- Subframe decode to MGA-GPS-EPH round-trip validation
- MGA-INI-POS_LLH and MGA-INI-TIME_UTC builders
- Ephemeris table formatting
- Cross-validation between convert_eph.py and ubx_tool.py (scaling round-trips, payload layout)
- NAV-PVT formatting
- TcpStream class (connection, read/write, timeout, baudrate guard, buffer drain)
- ensure_ubx_output TCP/serial behaviour
- TCP blocked command verification

## Development

This project was developed with the assistance of Claude Opus 4.6 extended thinking (Anthropic).

## License

[MIT](LICENSE)
