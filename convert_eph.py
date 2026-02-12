#!/usr/bin/env python3
"""
convert_eph.py -- Convert RINEX navigation files to u-blox MGA assistance messages.

Reads RINEX 2/3 GPS, QZSS, and GLONASS navigation files and produces UBX binary
files containing MGA ephemeris (EPH), health, ionosphere, UTC, and time offset
messages for assisted GNSS startup.

Dependencies: georinex, numpy

Usage:
  convert_eph.py brdc0380.26n -o eph.ubx
  convert_eph.py brdc0380.26n -o eph.ubx --time 2026-02-07T16:00
  convert_eph.py brdc0380.26n -o eph.ubx --systems GPS,QZSS
"""

import argparse
import contextlib
import gzip
import os
import shutil
import sys
import struct
import math
import tempfile
import warnings
import numpy as np
import xarray as xr
import georinex as gr


@contextlib.contextmanager
def _open_rinex(filepath):
    """Yield a path to a plain RINEX file, decompressing .gz if needed."""
    if filepath.endswith('.gz'):
        with tempfile.NamedTemporaryFile(suffix='.rnx', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with gzip.open(filepath, 'rb') as f_in, open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            yield tmp_path
        finally:
            os.unlink(tmp_path)
    else:
        yield filepath

# ---- RINEX 2 manual parser (fallback for georinex limitations) ----

# Field order for RINEX 2 GPS/QZSS navigation records (after header line)
# 7 data lines x 4 values = 28 parameters
RINEX2_NAV_FIELDS = [
    # Line 2
    'IODE', 'Crs', 'DeltaN', 'M0',
    # Line 3
    'Cuc', 'Eccentricity', 'Cus', 'sqrtA',
    # Line 4
    'Toe', 'Cic', 'Omega0', 'Cis',
    # Line 5
    'Io', 'Crc', 'omega', 'OmegaDot',
    # Line 6
    'IDOT', 'CodesL2', 'GPSWeek', 'L2Pflag',
    # Line 7
    'SVacc', 'health', 'TGD', 'IODC',
    # Line 8
    'TransTime', 'FitIntvl',
]

# Field order for RINEX GLONASS navigation records (after header line)
# 3 data lines x 4 values = 12 parameters
RINEX_GLO_FIELDS = [
    # Line 2
    'X', 'dX', 'dX2', 'health',
    # Line 3
    'Y', 'dY', 'dY2', 'FreqNum',
    # Line 4
    'Z', 'dZ', 'dZ2', 'AgeOpInfo',
]


def _parse_rinex2_float(s):
    """Parse RINEX 2 FORTRAN-style float (D exponent -> E)."""
    s = s.strip()
    if not s:
        return float('nan')
    s = s.replace('D', 'E').replace('d', 'e')
    return float(s)


def _parse_rinex2_nav_record(lines):
    """Parse one RINEX 2 navigation record (8 lines).

    Handles both GPS (numeric SV) and QZSS (J-prefix) RINEX 2 formats.
    QZSS records have a 1-character offset due to 'J' system identifier.

    Returns: (sv_label, epoch_dt64, field_dict) or None on error.
    """
    header = lines[0].rstrip('\n')

    # Detect system and SV number
    sv_char = header[0]
    if sv_char.isalpha():
        # QZSS (J), Galileo (E), etc: A1 + I2 format -> +1 offset
        prefix = sv_char
        try:
            sv_num = int(header[1:3])
        except ValueError:
            return None
        offset = 1  # everything shifted by 1 char
    else:
        # GPS: I2 format (space-padded number)
        prefix = 'G'
        try:
            sv_num = int(header[0:2])
        except ValueError:
            return None
        offset = 0

    sv_label = f'{prefix}{sv_num:02d}'

    # Parse epoch: YY(I3) MM(I3) DD(I3) HH(I3) MM(I3) SS.S(F5.1)
    # GPS:  positions 2-4, 5-7, 8-10, 11-13, 14-16, 17-21
    # QZSS: positions 3-5, 6-8, 9-11, 12-14, 15-17, 18-22 (+1 offset)
    base = 2 + offset
    try:
        yy = int(header[base:base+3])
        mm = int(header[base+3:base+6])
        dd = int(header[base+6:base+9])
        hh = int(header[base+9:base+12])
        mi = int(header[base+12:base+15])
        ss = float(header[base+15:base+20])

        year = yy + 2000 if yy < 80 else yy + 1900
        sec_int = int(ss)
        usec = int(round((ss - sec_int) * 1e6))
        epoch = np.datetime64(f'{year:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{sec_int:02d}', 'ns')
        if usec:
            epoch += np.timedelta64(usec, 'us')
    except (ValueError, IndexError):
        return None

    # Parse clock parameters: 3 x D19.12
    # GPS:  positions 22-40, 41-59, 60-78
    # QZSS: positions 23-41, 42-60, 61-79 (+1 offset)
    clk_base = 22 + offset
    fields = {}
    try:
        fields['SVclockBias'] = _parse_rinex2_float(header[clk_base:clk_base+19])
        fields['SVclockDrift'] = _parse_rinex2_float(header[clk_base+19:clk_base+38])
        fields['SVclockDriftRate'] = _parse_rinex2_float(header[clk_base+38:clk_base+57])
    except (ValueError, IndexError):
        return None

    # Parse 7 data lines (4 values each, 19 chars per value)
    # GPS: 3-char indent, QZSS: 4-char indent (same +1 offset)
    data_indent = 3 + offset
    field_idx = 0
    for line in lines[1:8]:
        line = line.rstrip('\n').ljust(data_indent + 4 * 19)
        for col in range(4):
            start = data_indent + col * 19
            end = start + 19
            if field_idx < len(RINEX2_NAV_FIELDS):
                val_str = line[start:end] if end <= len(line) else ''
                fields[RINEX2_NAV_FIELDS[field_idx]] = _parse_rinex2_float(val_str)
                field_idx += 1

    return sv_label, epoch, fields


def parse_rinex2_nav(filepath):
    """Parse RINEX 2 navigation file manually.

    Returns xarray Dataset compatible with georinex output.
    """
    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    # Skip header
    header_end = 0
    for i, line in enumerate(all_lines):
        if 'END OF HEADER' in line:
            header_end = i + 1
            break

    data_lines = all_lines[header_end:]

    # Parse all records (8 lines each)
    records = []
    i = 0
    while i + 7 < len(data_lines):
        # Check if this looks like a record start
        line = data_lines[i]
        if len(line.strip()) == 0:
            i += 1
            continue

        result = _parse_rinex2_nav_record(data_lines[i:i+8])
        if result:
            records.append(result)
            i += 8
        else:
            i += 1

    if not records:
        return None

    # Build xarray Dataset
    all_fields = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate'] + RINEX2_NAV_FIELDS

    # Get unique SVs and times
    sv_set = sorted(set(r[0] for r in records))
    time_set = sorted(set(r[1] for r in records))

    sv_arr = np.array(sv_set)
    time_arr = np.array(time_set)

    # Create NaN-filled arrays
    data_vars = {}
    for field in all_fields:
        data_vars[field] = (['time', 'sv'],
                            np.full((len(time_arr), len(sv_arr)), np.nan))

    # Fill in data
    sv_idx = {sv: i for i, sv in enumerate(sv_set)}
    time_idx = {t: i for i, t in enumerate(time_set)}

    for sv_label, epoch, fields in records:
        si = sv_idx[sv_label]
        ti = time_idx[epoch]
        for field_name, value in fields.items():
            if field_name in data_vars:
                data_vars[field_name][1][ti, si] = value

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'time': time_arr, 'sv': sv_arr},
    )

    return ds


# ---- RINEX 3 manual parser (for mixed nav files) ----

# Field order is identical to RINEX 2 for GPS/QZSS
RINEX3_NAV_FIELDS = RINEX2_NAV_FIELDS


def _parse_rinex3_float(s):
    """Parse RINEX 3 float value (lowercase or uppercase exponent)."""
    s = s.strip()
    if not s:
        return float('nan')
    s = s.replace('D', 'E').replace('d', 'e')
    return float(s)


def _parse_rinex3_nav_record(lines, allowed_systems='GJ'):
    """Parse one RINEX 3 GPS/QZSS navigation record (8 lines).

    RINEX 3 format:
      SV line: X## YYYY MM DD HH MM SS af0(D19.12) af1(D19.12) af2(D19.12)
      Data lines: 4-char indent, 4 values x 19 chars each

    Returns: (sv_label, epoch_dt64, field_dict) or None on error/skip.
    """
    header = lines[0].rstrip('\n')
    if len(header) < 23:
        return None

    # System identifier + PRN: X## (e.g. G01, J02, E25, C01)
    prefix = header[0]
    if prefix not in allowed_systems:
        return None
    try:
        sv_num = int(header[1:3])
    except ValueError:
        return None
    sv_label = f'{prefix}{sv_num:02d}'

    # Parse epoch: YYYY MM DD HH MM SS (fixed columns)
    try:
        year = int(header[4:8])
        mm = int(header[9:11])
        dd = int(header[12:14])
        hh = int(header[15:17])
        mi = int(header[18:20])
        ss = int(header[21:23])
        epoch = np.datetime64(f'{year:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{ss:02d}', 'ns')
    except (ValueError, IndexError):
        return None

    # Parse clock parameters: 3 x D19.12 starting at col 23
    fields = {}
    try:
        fields['SVclockBias'] = _parse_rinex3_float(header[23:42])
        fields['SVclockDrift'] = _parse_rinex3_float(header[42:61])
        fields['SVclockDriftRate'] = _parse_rinex3_float(header[61:80])
    except (ValueError, IndexError):
        return None

    # Parse 7 data lines (4 values each, 19 chars per value, 4-char indent)
    field_idx = 0
    for line in lines[1:8]:
        line = line.rstrip('\n').ljust(4 + 4 * 19)
        for col in range(4):
            start = 4 + col * 19
            end = start + 19
            if field_idx < len(RINEX3_NAV_FIELDS):
                val_str = line[start:end] if end <= len(line) else ''
                fields[RINEX3_NAV_FIELDS[field_idx]] = _parse_rinex3_float(val_str)
                field_idx += 1

    return sv_label, epoch, fields


def _parse_rinex3_glo_record(lines):
    """Parse one RINEX 3 GLONASS navigation record (4 lines).

    RINEX 3 GLONASS format:
      SV line: R## YYYY MM DD HH MM SS -TauN(D19.12) GammaN(D19.12) MsgFrameTime(D19.12)
      Data lines (3): 4-char indent, 4 values x 19 chars each

    Note: georinex multiplies GLONASS X/Y/Z by 1000 (km->m).
    We follow that convention here for compatibility.

    Returns: (sv_label, epoch_dt64, field_dict) or None on error.
    """
    header = lines[0].rstrip('\n')
    if len(header) < 23 or header[0] != 'R':
        return None

    try:
        sv_num = int(header[1:3])
    except ValueError:
        return None
    sv_label = f'R{sv_num:02d}'

    # Parse epoch (UTC for GLONASS)
    try:
        year = int(header[4:8])
        mm = int(header[9:11])
        dd = int(header[12:14])
        hh = int(header[15:17])
        mi = int(header[18:20])
        ss = int(header[21:23])
        epoch = np.datetime64(f'{year:04d}-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{ss:02d}', 'ns')
    except (ValueError, IndexError):
        return None

    # Clock parameters: -TauN, GammaN, MessageFrameTime
    fields = {}
    try:
        fields['SVclockBias'] = _parse_rinex3_float(header[23:42])
        fields['SVrelFreqBias'] = _parse_rinex3_float(header[42:61])
        fields['MessageFrameTime'] = _parse_rinex3_float(header[61:80])
    except (ValueError, IndexError):
        return None

    # Parse 3 data lines (4 values each)
    field_idx = 0
    for line in lines[1:4]:
        line = line.rstrip('\n').ljust(4 + 4 * 19)
        for col in range(4):
            start = 4 + col * 19
            end = start + 19
            if field_idx < len(RINEX_GLO_FIELDS):
                val_str = line[start:end] if end <= len(line) else ''
                val = _parse_rinex3_float(val_str)
                fname = RINEX_GLO_FIELDS[field_idx]
                # Multiply position/velocity/acceleration by 1000 (km->m)
                # to match georinex convention
                if fname in ('X', 'Y', 'Z', 'dX', 'dY', 'dZ',
                             'dX2', 'dY2', 'dZ2'):
                    val = val * 1000.0
                fields[fname] = val
                field_idx += 1

    return sv_label, epoch, fields


def _detect_rinex_version(filepath):
    """Detect RINEX version and type from first header line.

    Returns (major_version, file_type) e.g. (3, 'MIXED') or (2, 'GPS').
    """
    with open(filepath) as f:
        line = f.readline()
    version_str = line[0:9].strip()
    try:
        version = float(version_str)
    except ValueError:
        return 0, 'UNKNOWN'
    major = int(version)

    # File type indicator
    type_field = line[40:60].strip().upper()
    if 'MIXED' in type_field or 'M' == type_field:
        file_type = 'MIXED'
    elif 'J' in line[40:41]:
        file_type = 'QZSS'
    else:
        file_type = 'GPS'

    return major, file_type


def parse_rinex3_nav(filepath, systems='GJ'):
    """Parse RINEX 3 navigation file manually, extracting only specified systems.

    Supports GPS (G), QZSS (J), and GLONASS (R) records.
    GPS/QZSS records are 8 lines; GLONASS records are 4 lines.

    Returns xarray Dataset compatible with georinex output, or None.
    """
    # Record line counts per GNSS system
    RECORD_LINES = {'G': 8, 'J': 8, 'R': 4,
                    'E': 8, 'C': 8, 'I': 8, 'S': 4}

    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    # Skip header
    header_end = 0
    for i, line in enumerate(all_lines):
        if 'END OF HEADER' in line:
            header_end = i + 1
            break

    data_lines = all_lines[header_end:]

    # Separate GPS/QZSS and GLONASS records (they have different field structures)
    gj_records = []
    glo_records = []
    i = 0
    while i < len(data_lines):
        line = data_lines[i]
        if len(line) < 4 or not line[0].isalpha() or not line[1:3].strip().isdigit():
            i += 1
            continue

        sys_char = line[0]
        n_lines = RECORD_LINES.get(sys_char, 8)

        if sys_char not in systems:
            # Skip this record
            i += n_lines
            continue

        if i + n_lines - 1 < len(data_lines):
            if sys_char == 'R':
                result = _parse_rinex3_glo_record(data_lines[i:i+n_lines])
                if result:
                    glo_records.append(result)
            elif sys_char in ('G', 'J'):
                result = _parse_rinex3_nav_record(data_lines[i:i+n_lines], systems)
                if result:
                    gj_records.append(result)
            i += n_lines
        else:
            break

    datasets = []

    # Build GPS/QZSS dataset
    if gj_records:
        all_fields = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate'] + RINEX3_NAV_FIELDS
        sv_set = sorted(set(r[0] for r in gj_records))
        time_set = sorted(set(r[1] for r in gj_records))
        sv_arr = np.array(sv_set)
        time_arr = np.array(time_set)
        data_vars = {}
        for field in all_fields:
            data_vars[field] = (['time', 'sv'],
                                np.full((len(time_arr), len(sv_arr)), np.nan))
        sv_idx = {sv: i for i, sv in enumerate(sv_set)}
        time_idx = {t: i for i, t in enumerate(time_set)}
        for sv_label, epoch, fields in gj_records:
            si = sv_idx[sv_label]
            ti = time_idx[epoch]
            for field_name, value in fields.items():
                if field_name in data_vars:
                    data_vars[field_name][1][ti, si] = value
        datasets.append(xr.Dataset(
            data_vars=data_vars,
            coords={'time': time_arr, 'sv': sv_arr},
        ))

    # Build GLONASS dataset
    if glo_records:
        all_glo_fields = ['SVclockBias', 'SVrelFreqBias',
                          'MessageFrameTime'] + RINEX_GLO_FIELDS
        sv_set = sorted(set(r[0] for r in glo_records))
        time_set = sorted(set(r[1] for r in glo_records))
        sv_arr = np.array(sv_set)
        time_arr = np.array(time_set)
        data_vars = {}
        for field in all_glo_fields:
            data_vars[field] = (['time', 'sv'],
                                np.full((len(time_arr), len(sv_arr)), np.nan))
        sv_idx = {sv: i for i, sv in enumerate(sv_set)}
        time_idx = {t: i for i, t in enumerate(time_set)}
        for sv_label, epoch, fields in glo_records:
            si = sv_idx[sv_label]
            ti = time_idx[epoch]
            for field_name, value in fields.items():
                if field_name in data_vars:
                    data_vars[field_name][1][ti, si] = value
        datasets.append(xr.Dataset(
            data_vars=data_vars,
            coords={'time': time_arr, 'sv': sv_arr},
        ))

    if not datasets:
        return None

    if len(datasets) == 1:
        return datasets[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return xr.merge(datasets, join='outer')


def load_rinex_nav(filepath):
    """Load RINEX navigation file, with fallback for formats georinex can't handle.

    Supports:
      - RINEX 2.x GPS-only files (e.g. brdc0380.26n from QZSS Japan)
      - RINEX 2.x QZSS-only files (e.g. brdc0380.26q from QZSS Japan)
      - RINEX 3.x mixed multi-GNSS files (e.g. BRDC00WRD from IGS)
      - GLONASS (R##) from RINEX 3.x mixed files

    Returns xarray Dataset with GPS (G##), QZSS (J##), and/or GLONASS (R##) satellites.
    """
    major_ver, _ = _detect_rinex_version(filepath)

    if major_ver >= 3:
        # RINEX 3: try georinex with system filter first
        parts = []

        # Try GPS+QZSS combined (fastest if it works)
        nav_gj = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nav_gj = gr.load(filepath, use='GJ')
            if len(nav_gj.coords.get('sv', [])) == 0:
                nav_gj = None
        except Exception:
            nav_gj = None

        if nav_gj is None:
            # Combined failed -- try G and J separately
            for sys_char in ('G', 'J'):
                nav_sys = None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        nav_sys = gr.load(filepath, use=sys_char)
                    if len(nav_sys.coords.get('sv', [])) == 0:
                        nav_sys = None
                except Exception:
                    nav_sys = None
                if nav_sys is None:
                    nav_sys = parse_rinex3_nav(filepath, systems=sys_char)
                if nav_sys is not None:
                    parts.append(nav_sys)
        else:
            parts.append(nav_gj)

        # GLONASS: try georinex first, fall back to manual parser
        nav_glo = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nav_glo = gr.load(filepath, use='R')
            if len(nav_glo.coords.get('sv', [])) == 0:
                nav_glo = None
        except Exception:
            nav_glo = None
        if nav_glo is None:
            nav_glo = parse_rinex3_nav(filepath, systems='R')
        if nav_glo is not None:
            parts.append(nav_glo)

        if parts:
            if len(parts) == 1:
                nav = parts[0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    nav = xr.merge(parts, join='outer')
            if len(nav.coords.get('sv', [])) > 0:
                return nav

        # Everything failed -- try full manual parser for all systems
        nav = parse_rinex3_nav(filepath, systems='GJR')
        if nav is not None and len(nav.coords.get('sv', [])) > 0:
            return nav

    else:
        # RINEX 2: try georinex first
        try:
            nav = gr.load(filepath)
            svs = nav.coords.get('sv', [])
            times = nav.coords.get('time', [])
            if len(svs) > 0 and len(times) > 0:
                return nav
        except Exception:
            pass

        # Fallback: manual RINEX 2 parser
        nav = parse_rinex2_nav(filepath)
        if nav is not None:
            return nav

    raise RuntimeError(f"Could not parse {filepath} (unsupported RINEX format)")


# ---- GPS time utilities ----

# GPS epoch: January 6, 1980 00:00:00 UTC
GPS_EPOCH = np.datetime64('1980-01-06T00:00:00', 'ns')
SECONDS_PER_WEEK = 604800


def datetime64_to_gps_seconds(dt64):
    """Convert numpy datetime64 to total GPS seconds since epoch."""
    delta = (dt64 - GPS_EPOCH) / np.timedelta64(1, 's')
    return float(delta)


def datetime64_to_gps_week_toc(dt64):
    """Convert datetime64 to (gps_week, toc_seconds_of_week)."""
    total_secs = datetime64_to_gps_seconds(dt64)
    week = int(total_secs // SECONDS_PER_WEEK)
    toc = total_secs - week * SECONDS_PER_WEEK
    return week, toc


# ---- UBX message construction ----

def ubx_checksum(data):
    """Compute UBX Fletcher-8 checksum over class+id+length+payload."""
    ck_a, ck_b = 0, 0
    for b in data:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return bytes([ck_a, ck_b])


def create_ubx_message(msg_class, msg_id, payload):
    """Build a complete UBX message with sync, header, payload, and checksum."""
    header = struct.pack('<BBBBH', 0xB5, 0x62, msg_class, msg_id, len(payload))
    ck = ubx_checksum(header[2:] + payload)
    return header + payload + ck


# ---- URA lookup table (IS-GPS-200) ----

URA_TABLE = [2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24, 48,
             96, 192, 384, 768, 1536, 3072, 6144, 6145]


def ura_meters_to_index(ura_m):
    """Convert URA accuracy in meters to URA index (0-15)."""
    if np.isnan(ura_m):
        return 0
    for i, threshold in enumerate(URA_TABLE):
        if ura_m <= threshold:
            return i
    return 15


# ---- Scaling: RINEX SI -> UBX raw integers ----

def scale_signed(value, lsb, bits):
    """Scale a RINEX float to a signed integer with given LSB and bit width.

    Args:
        value: RINEX value in SI units
        lsb: least significant bit scaling factor
        bits: target bit width (for clamping)
    Returns:
        signed integer clamped to [-2^(bits-1), 2^(bits-1)-1]
    """
    raw = round(value / lsb)
    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    return max(min_val, min(max_val, raw))


def scale_unsigned(value, lsb, bits):
    """Scale a RINEX float to an unsigned integer with given LSB and bit width."""
    raw = round(value / lsb)
    max_val = (1 << bits) - 1
    return max(0, min(max_val, raw))


def scale_angular_signed(value_rad, lsb, bits):
    """Scale a RINEX angular value (radians) to semi-circles, then to raw integer.

    RINEX stores angles in radians. UBX uses semi-circles (value/pi).
    """
    semicircles = value_rad / math.pi
    return scale_signed(semicircles, lsb, bits)


def scale_angular_unsigned(value_rad, lsb, bits):
    """Scale a RINEX angular value (radians) to unsigned semi-circle raw integer."""
    semicircles = value_rad / math.pi
    return scale_unsigned(semicircles, lsb, bits)


# ---- Build MGA-GPS-EPH / MGA-QZSS-EPH payload ----

def rinex_epoch_to_mga_eph(sv_id, epoch_data, toc_seconds):
    """Convert one RINEX ephemeris epoch to MGA-GPS/QZSS-EPH raw integers.

    Args:
        sv_id: satellite PRN number (1-32 for GPS, 1-10 for QZSS)
        epoch_data: dict-like with RINEX field values (floats)
        toc_seconds: Toc in seconds of GPS week

    Returns:
        dict of raw integer values ready for build_mga_eph_payload()
    """
    # Integer fields (no scaling needed beyond rounding)
    iodc = int(epoch_data['IODC'])
    sv_health = int(epoch_data['health'])
    ura_index = ura_meters_to_index(epoch_data['SVacc'])

    # FitInterval: 0 = 4h, 1 = >4h
    fit_intvl = epoch_data.get('FitIntvl', 4.0)
    fit_interval = 0 if (np.isnan(fit_intvl) or fit_intvl <= 4.0) else 1

    # Time fields: seconds -> /16
    toc_raw = scale_unsigned(toc_seconds, 16, 16)
    toe_raw = scale_unsigned(epoch_data['Toe'], 16, 16)

    # Clock parameters
    tgd_raw = scale_signed(epoch_data['TGD'], 2**-31, 8)
    af2_raw = scale_signed(epoch_data['SVclockDriftRate'], 2**-55, 8)
    af1_raw = scale_signed(epoch_data['SVclockDrift'], 2**-43, 16)
    af0_raw = scale_signed(epoch_data['SVclockBias'], 2**-31, 32)

    # Orbital: non-angular
    crs_raw = scale_signed(epoch_data['Crs'], 2**-5, 16)
    crc_raw = scale_signed(epoch_data['Crc'], 2**-5, 16)
    cuc_raw = scale_signed(epoch_data['Cuc'], 2**-29, 16)
    cus_raw = scale_signed(epoch_data['Cus'], 2**-29, 16)
    cic_raw = scale_signed(epoch_data['Cic'], 2**-29, 16)
    cis_raw = scale_signed(epoch_data['Cis'], 2**-29, 16)
    e_raw = scale_unsigned(epoch_data['Eccentricity'], 2**-33, 32)
    sqrtA_raw = scale_unsigned(epoch_data['sqrtA'], 2**-19, 32)

    # Orbital: angular (radians -> semi-circles -> raw)
    deltaN_raw = scale_angular_signed(epoch_data['DeltaN'], 2**-43, 16)
    m0_raw = scale_angular_signed(epoch_data['M0'], 2**-31, 32)
    omega0_raw = scale_angular_signed(epoch_data['Omega0'], 2**-31, 32)
    i0_raw = scale_angular_signed(epoch_data['Io'], 2**-31, 32)
    omega_raw = scale_angular_signed(epoch_data['omega'], 2**-31, 32)
    omegaDot_raw = scale_angular_signed(epoch_data['OmegaDot'], 2**-43, 32)
    idot_raw = scale_angular_signed(epoch_data['IDOT'], 2**-43, 16)

    return {
        'sv_id': sv_id,
        'fit_interval': fit_interval,
        'ura_index': ura_index,
        'sv_health': sv_health,
        'tgd': tgd_raw,
        'iodc': iodc,
        'toc': toc_raw,
        'af2': af2_raw,
        'af1': af1_raw,
        'af0': af0_raw,
        'crs': crs_raw,
        'deltaN': deltaN_raw,
        'm0': m0_raw,
        'cuc': cuc_raw,
        'cus': cus_raw,
        'e': e_raw,
        'sqrtA': sqrtA_raw,
        'toe': toe_raw,
        'cic': cic_raw,
        'omega0': omega0_raw,
        'cis': cis_raw,
        'crc': crc_raw,
        'i0': i0_raw,
        'omega': omega_raw,
        'omegaDot': omegaDot_raw,
        'idot': idot_raw,
    }


def build_mga_eph_payload(raw):
    """Build 68-byte MGA-GPS/QZSS-EPH payload from raw integer dict.

    Field order verified against pyubx2 v1.2.59 (not u-blox PDF).
    """
    payload = struct.pack('<BBBBBBBb',
        0x01,                   # type
        0x00,                   # version
        raw['sv_id'],           # svId
        0x00,                   # reserved0
        raw['fit_interval'],    # fitInterval
        raw['ura_index'],       # uraIndex
        raw['sv_health'],       # svHealth
        raw['tgd'],             # tgd (I1)
    )
    payload += struct.pack('<HH',
        raw['iodc'],            # iodc (U2)
        raw['toc'],             # toc (U2)
    )
    payload += struct.pack('<Bbh',
        0x00,                   # reserved1 (U1)
        raw['af2'],             # af2 (I1)
        raw['af1'],             # af1 (I2)
    )
    payload += struct.pack('<i',
        raw['af0'],             # af0 (I4)
    )
    payload += struct.pack('<hh',
        raw['crs'],             # crs (I2)
        raw['deltaN'],          # deltaN (I2)
    )
    payload += struct.pack('<i',
        raw['m0'],              # m0 (I4)
    )
    payload += struct.pack('<hh',
        raw['cuc'],             # cuc (I2)
        raw['cus'],             # cus (I2)
    )
    payload += struct.pack('<II',
        raw['e'] & 0xFFFFFFFF,  # e (U4)
        raw['sqrtA'] & 0xFFFFFFFF,  # sqrtA (U4)
    )
    payload += struct.pack('<Hh',
        raw['toe'],             # toe (U2)
        raw['cic'],             # cic (I2)
    )
    payload += struct.pack('<i',
        raw['omega0'],          # omega0 (I4)
    )
    payload += struct.pack('<hh',
        raw['cis'],             # cis (I2)
        raw['crc'],             # crc (I2)
    )
    payload += struct.pack('<i',
        raw['i0'],              # i0 (I4)
    )
    payload += struct.pack('<i',
        raw['omega'],           # omega (I4)
    )
    payload += struct.pack('<i',
        raw['omegaDot'],        # omegaDot (I4)
    )
    payload += struct.pack('<hH',
        raw['idot'],            # idot (I2)
        0x0000,                 # reserved2 (U2)
    )

    assert len(payload) == 68, f"Payload must be 68 bytes, got {len(payload)}"
    return payload


def convert_epoch(sv_id, epoch_data, toc_seconds, msg_id):
    """Full conversion: RINEX epoch -> complete UBX message bytes.

    Args:
        sv_id: PRN number
        epoch_data: RINEX field values
        toc_seconds: Toc in seconds of GPS week
        msg_id: 0x00 for GPS, 0x05 for QZSS
    Returns:
        bytes: complete UBX message
    """
    raw = rinex_epoch_to_mga_eph(sv_id, epoch_data, toc_seconds)
    payload = build_mga_eph_payload(raw)
    return create_ubx_message(0x13, msg_id, payload)


# ---- Build MGA-*-HEALTH messages ----

def build_mga_gps_health(health_map):
    """Build MGA-GPS-HEALTH message (class=0x13, id=0x00, 40-byte payload).

    Args:
        health_map: dict {sv_id(1-32): health_byte(U1)} from RINEX SVhealth field.
                    6 LSBs used. 0 = healthy. Missing SVs default to 0.
    Returns:
        bytes: complete UBX message
    """
    payload = struct.pack('<BB2s',
        0x04,               # type
        0x00,               # version
        b'\x00\x00',        # reserved1
    )
    # healthCode[32]: one byte per GPS SV 1-32
    for sv in range(1, 33):
        payload += struct.pack('<B', health_map.get(sv, 0) & 0x3F)
    payload += b'\x00\x00\x00\x00'  # reserved2
    assert len(payload) == 40
    return create_ubx_message(0x13, 0x00, payload)


def build_mga_qzss_health(health_map):
    """Build MGA-QZSS-HEALTH message (class=0x13, id=0x05, 12-byte payload).

    Args:
        health_map: dict {sv_id(1-5): health_byte(U1)} from RINEX SVhealth field.
                    6 LSBs used. 0 = healthy. Missing SVs default to 0.
    Returns:
        bytes: complete UBX message
    """
    payload = struct.pack('<BB2s',
        0x04,               # type
        0x00,               # version
        b'\x00\x00',        # reserved1
    )
    # healthCode[5]: one byte per QZSS SV 1-5
    for sv in range(1, 6):
        payload += struct.pack('<B', health_map.get(sv, 0) & 0x3F)
    payload += b'\x00\x00\x00'  # reserved2
    assert len(payload) == 12
    return create_ubx_message(0x13, 0x05, payload)


# ---- Build MGA-GPS-IONO / MGA-GPS-UTC from RINEX header ----

def parse_rinex_header(path):
    """Extract ionospheric, UTC, and leap second parameters from RINEX nav header.

    Returns dict with keys like 'GPSA', 'GPSB', 'GPUT', 'QZSA', 'QZSB', 'QZUT', 'LEAP'.
    """
    params = {}
    with open(path) as f:
        for line in f:
            label = line[60:].strip()
            if label == 'END OF HEADER':
                break
            if label == 'IONOSPHERIC CORR':
                key = line[:4].strip()  # GPSA, GPSB, QZSA, QZSB
                vals = []
                for i in range(4):
                    s = line[5 + i*12 : 5 + (i+1)*12].strip().replace('D', 'E')
                    vals.append(float(s))
                params[key] = vals
            elif label == 'TIME SYSTEM CORR':
                key = line[:4].strip()  # GPUT, QZUT
                a0 = float(line[5:22].strip().replace('D', 'E'))
                a1 = float(line[22:38].strip().replace('D', 'E'))
                t = int(line[38:45].strip())
                w = int(line[45:50].strip())
                params[key] = {'a0': a0, 'a1': a1, 'tot': t, 'wnt': w}
            elif label == 'LEAP SECONDS':
                params['LEAP'] = int(line[:6])
    return params


def build_mga_gps_iono(gpsa, gpsb):
    """Build MGA-GPS-IONO message (class=0x13, id=0x00, 16-byte payload).

    Args:
        gpsa: list of 4 Klobuchar alpha parameters (floats, SI units)
        gpsb: list of 4 Klobuchar beta parameters (floats, SI units)
    Returns:
        bytes: complete UBX message
    """
    # Scaling from u-blox spec (same as GPS ICD):
    # alpha: 2^-30, 2^-27, 2^-24, 2^-24
    # beta:  2^11, 2^14, 2^16, 2^16
    a_scales = [2**-30, 2**-27, 2**-24, 2**-24]
    b_scales = [2**11, 2**14, 2**16, 2**16]

    a_raw = [max(-128, min(127, round(gpsa[i] / a_scales[i]))) for i in range(4)]
    b_raw = [max(-128, min(127, round(gpsb[i] / b_scales[i]))) for i in range(4)]

    payload = struct.pack('<BB2s4b4b4s',
        0x06,               # type
        0x00,               # version
        b'\x00\x00',        # reserved1
        *a_raw,             # alpha0-3 (I1)
        *b_raw,             # beta0-3 (I1)
        b'\x00\x00\x00\x00',  # reserved2
    )
    assert len(payload) == 16
    return create_ubx_message(0x13, 0x00, payload)


def build_mga_gps_utc(gput, leap_seconds):
    """Build MGA-GPS-UTC message (class=0x13, id=0x00, 20-byte payload).

    Args:
        gput: dict with 'a0', 'a1', 'tot' (seconds), 'wnt' (full GPS week)
        leap_seconds: current UTC-GPS leap seconds (int)
    Returns:
        bytes: complete UBX message
    """
    # utcA0: I4, scaled 2^-30 s
    # utcA1: I4, scaled 2^-50 s/s
    # utcDtLS: I1, current leap seconds
    # utcTot: U1, scaled 2^12 s (tot / 4096)
    # utcWNt: U1, 8-bit truncated GPS week
    # utcWNlsf, utcDN, utcDtLSF: future leap second info (not in RINEX)
    a0_raw = round(gput['a0'] / 2**-30)
    a0_raw = max(-2**31, min(2**31 - 1, a0_raw))
    a1_raw = round(gput['a1'] / 2**-50)
    a1_raw = max(-2**31, min(2**31 - 1, a1_raw))
    tot_raw = round(gput['tot'] / 4096) & 0xFF
    wnt_raw = gput['wnt'] & 0xFF

    # Future leap second fields: not available in RINEX header.
    # Set dtLSF = dtLS (no pending change), wnlsf/dn = 0.
    payload = struct.pack('<BB2siibBBBBb2s',
        0x05,               # type
        0x00,               # version
        b'\x00\x00',        # reserved1
        a0_raw,             # utcA0 (I4)
        a1_raw,             # utcA1 (I4)
        leap_seconds,       # utcDtLS (I1)
        tot_raw,            # utcTot (U1)
        wnt_raw,            # utcWNt (U1)
        0,                  # utcWNlsf (U1) -- not in RINEX
        0,                  # utcDN (U1) -- not in RINEX
        leap_seconds,       # utcDtLSF (I1) -- same as dtLS (no pending change)
        b'\x00\x00',        # reserved2
    )
    assert len(payload) == 20, f"Expected 20, got {len(payload)}"
    return create_ubx_message(0x13, 0x00, payload)


# ---- Build MGA-GLO-EPH / MGA-GLO-TIMEOFFSET ----

def build_mga_glo_eph(sv_id, epoch_vals, epoch_time):
    """Build MGA-GLO-EPH message (class=0x13, id=0x06, 48-byte payload).

    Args:
        sv_id: GLONASS slot number (1-24)
        epoch_vals: dict of GLONASS ephemeris values (georinex convention)
        epoch_time: numpy datetime64 of the epoch (UTC)
    Returns:
        bytes: complete UBX message
    """
    # Position: georinex meters -> km, then scale by 2^11
    x_km = epoch_vals.get('X', 0.0) / 1000.0
    y_km = epoch_vals.get('Y', 0.0) / 1000.0
    z_km = epoch_vals.get('Z', 0.0) / 1000.0
    x_raw = max(-2**31, min(2**31 - 1, round(x_km / 2**-11)))
    y_raw = max(-2**31, min(2**31 - 1, round(y_km / 2**-11)))
    z_raw = max(-2**31, min(2**31 - 1, round(z_km / 2**-11)))

    # Velocity: georinex m/s -> km/s, then scale by 2^20
    dx_kms = epoch_vals.get('dX', 0.0) / 1000.0
    dy_kms = epoch_vals.get('dY', 0.0) / 1000.0
    dz_kms = epoch_vals.get('dZ', 0.0) / 1000.0
    dx_raw = max(-2**31, min(2**31 - 1, round(dx_kms / 2**-20)))
    dy_raw = max(-2**31, min(2**31 - 1, round(dy_kms / 2**-20)))
    dz_raw = max(-2**31, min(2**31 - 1, round(dz_kms / 2**-20)))

    # Acceleration: georinex m/s^2 -> km/s^2, then scale by 2^30
    ddx_kms2 = epoch_vals.get('dX2', 0.0) / 1000.0
    ddy_kms2 = epoch_vals.get('dY2', 0.0) / 1000.0
    ddz_kms2 = epoch_vals.get('dZ2', 0.0) / 1000.0
    ddx_raw = max(-128, min(127, round(ddx_kms2 / 2**-30)))
    ddy_raw = max(-128, min(127, round(ddy_kms2 / 2**-30)))
    ddz_raw = max(-128, min(127, round(ddz_kms2 / 2**-30)))

    # tb: epoch (UTC) -> Moscow time (UTC+3) -> 15-minute interval index
    ts = (epoch_time - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's')
    total_seconds = int(ts)
    utc_hour = (total_seconds % 86400) // 3600
    utc_min = (total_seconds % 3600) // 60
    moscow_min = (utc_hour + 3) * 60 + utc_min  # Moscow = UTC + 3h
    if moscow_min >= 1440:
        moscow_min -= 1440  # Day wrap
    tb_raw = moscow_min // 15
    tb_raw = min(255, max(0, tb_raw))

    # gamma: SVrelFreqBias (dimensionless) -> I2 scaled 2^-40
    gamma = epoch_vals.get('SVrelFreqBias', 0.0)
    gamma_raw = max(-32768, min(32767, round(gamma / 2**-40)))

    # tau: RINEX stores -TauN in SVclockBias; MGA stores TauN
    # So: MGA_tau = -SVclockBias
    sv_clock_bias = epoch_vals.get('SVclockBias', 0.0)
    tau = -sv_clock_bias
    tau_raw = max(-2**31, min(2**31 - 1, round(tau / 2**-30)))

    # E: age of operation info (days)
    e_raw = min(255, max(0, int(epoch_vals.get('AgeOpInfo', 0.0))))

    # B: health (Bn), 3 bits
    bn = min(7, max(0, int(epoch_vals.get('health', 0.0))))

    # H: frequency number (-7 to +13)
    freq_num = int(epoch_vals.get('FreqNum', 0.0))
    freq_num = max(-128, min(127, freq_num))

    # FT: accuracy index -- not available in RINEX, default to 0
    ft = 0

    # M: GLONASS-M flag -- default to 1 (most active sats are GLONASS-M)
    m_type = 1

    # deltaTau: L1-L2 time difference -- not in RINEX, default to 0
    delta_tau_raw = 0

    payload = struct.pack('<BBBBBBBbiiiiiibbbbhBbiI',
        0x01,           # type
        0x00,           # version
        sv_id,          # svId (U1)
        0x00,           # reserved0
        ft,             # FT (U1)
        bn,             # B (U1)
        m_type,         # M (U1)
        freq_num,       # H (I1)
        x_raw,          # x (I4)
        y_raw,          # y (I4)
        z_raw,          # z (I4)
        dx_raw,         # dx (I4)
        dy_raw,         # dy (I4)
        dz_raw,         # dz (I4)
        ddx_raw,        # ddx (I1)
        ddy_raw,        # ddy (I1)
        ddz_raw,        # ddz (I1)
        tb_raw,         # tb (U1)
        gamma_raw,      # gamma (I2)
        e_raw,          # E (U1)
        delta_tau_raw,  # deltaTau (I1)
        tau_raw,        # tau (I4)
        0,              # reserved1 (U4)
    )
    assert len(payload) == 48, f"Expected 48, got {len(payload)}"
    return create_ubx_message(0x13, 0x06, payload)


def _compute_glonass_day_number(dt64):
    """Compute GLONASS day number N within 4-year cycle.

    The GLONASS 4-year cycle starts on Jan 1 of each leap year:
    ..., 2020, 2024, 2028, ...

    N = 1 on Jan 1 of the leap year.
    """
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's')
    from datetime import datetime as dt_cls, timezone as tz
    utc_dt = dt_cls.fromtimestamp(int(ts), tz=tz.utc)

    # Find the start of current 4-year cycle (nearest leap year <= current year)
    year = utc_dt.year
    cycle_start_year = year - (year % 4)
    cycle_start = dt_cls(cycle_start_year, 1, 1, tzinfo=tz.utc)

    # Day number (1-indexed)
    day_num = (utc_dt - cycle_start).days + 1
    return day_num


def build_mga_glo_timeoffset(glut=None, glgp=None, epoch_time=None):
    """Build MGA-GLO-TIMEOFFSET message (class=0x13, id=0x06, 20-byte payload).

    Args:
        glut: dict with 'a0' (tauC, GLONASS-UTC offset in seconds), or None
        glgp: dict with 'a0' (tauGps, GLONASS-GPS offset in seconds), or None
        epoch_time: numpy datetime64 for computing day number N
    Returns:
        bytes: complete UBX message
    """
    # N: day number within 4-year cycle
    if epoch_time is not None:
        n_day = _compute_glonass_day_number(epoch_time)
    else:
        n_day = 0

    # tauC: GLONASS-UTC correction, I4 scaled 2^-27 ns ... actually scaled in seconds
    # pyubx2 scale: 7.450580596923828e-09 = 2^-27 (interpreted as seconds)
    tau_c = glut['a0'] if glut else 0.0
    tau_c_raw = max(-2**31, min(2**31 - 1, round(tau_c / 2**-27)))

    # tauGps: GLONASS-GPS correction, I4 scaled 2^-31 (seconds)
    tau_gps = glgp['a0'] if glgp else 0.0
    tau_gps_raw = max(-2**31, min(2**31 - 1, round(tau_gps / 2**-31)))

    # B1, B2: UT1-UTC correction parameters -- not available from RINEX header
    b1_raw = 0
    b2_raw = 0

    payload = struct.pack('<BBHiihhI',
        0x03,           # type
        0x00,           # version
        n_day & 0xFFFF, # N (U2)
        tau_c_raw,      # tauC (I4)
        tau_gps_raw,    # tauGps (I4)
        b1_raw,         # B1 (I2)
        b2_raw,         # B2 (I2)
        0,              # reserved0 (U4)
    )
    assert len(payload) == 20, f"Expected 20, got {len(payload)}"
    return create_ubx_message(0x13, 0x06, payload)


# ---- Epoch selection ----

def select_best_epoch(sv_data, target_time=None, max_age_hours=4.0, sentinel=None):
    """Select the best ephemeris epoch for a satellite.

    Args:
        sv_data: xarray Dataset for one SV (all epochs)
        target_time: numpy datetime64 target time (default: latest available)
        max_age_hours: maximum age in hours (default: 4.0, standard ephemeris validity)
        sentinel: field name to check for valid data (default: auto-detect)

    Returns:
        (time_index, epoch_time) or None if no valid epoch found
    """
    # Determine which field indicates valid data
    if sentinel is None:
        if 'sqrtA' in sv_data.data_vars:
            sentinel = 'sqrtA'
        elif 'X' in sv_data.data_vars:
            sentinel = 'X'
        else:
            return None
    valid_mask = ~np.isnan(sv_data[sentinel].values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return None

    times = sv_data.coords['time'].values

    if target_time is None:
        # Use the latest epoch
        idx = valid_indices[-1]
        return idx, times[idx]

    # Find closest epoch to target_time, within max_age_hours
    valid_times = times[valid_indices]
    deltas = np.abs(valid_times - target_time) / np.timedelta64(1, 'h')
    best_offset = np.argmin(deltas)

    if deltas[best_offset] > max_age_hours:
        return None

    idx = valid_indices[best_offset]
    return idx, times[idx]


# ---- Main conversion logic ----

def convert_rinex(nav, target_time=None, max_age_hours=4.0, systems=None):
    """Convert RINEX navigation data to list of UBX MGA-EPH messages.

    Args:
        nav: xarray Dataset from georinex
        target_time: target time for epoch selection (datetime64)
        max_age_hours: maximum ephemeris age
        systems: set of systems to include, e.g. {'GPS', 'QZSS', 'GLO'} (default: all)

    Returns:
        list of (sv_label, ubx_bytes, epoch_time, epoch_vals) tuples
    """
    if systems is None:
        systems = {'GPS', 'QZSS', 'GLO'}

    svs = [str(s) for s in nav.coords['sv'].values]
    messages = []

    for sv_label in sorted(svs):
        prefix = sv_label[0]
        sv_num = int(sv_label[1:])

        # Determine GNSS system and conversion path
        if prefix == 'G' and 'GPS' in systems:
            gnss = 'GPS'
            msg_id = 0x00  # MGA-GPS-EPH
            sv_id = sv_num  # 1-32
        elif prefix == 'J' and 'QZSS' in systems:
            gnss = 'QZSS'
            msg_id = 0x05  # MGA-QZSS-EPH
            sv_id = sv_num  # 1-10
        elif prefix == 'R' and 'GLO' in systems:
            gnss = 'GLO'
            sv_id = sv_num  # 1-24
        else:
            continue

        sv_data = nav.sel(sv=sv_label)

        # GLONASS ephemeris has shorter validity (~30 min updates)
        if gnss == 'GLO':
            age = min(max_age_hours, 1.0)
            sentinel = 'X'
        else:
            age = max_age_hours
            sentinel = 'sqrtA'
        result = select_best_epoch(sv_data, target_time, age, sentinel=sentinel)
        if result is None:
            continue

        idx, epoch_time = result

        # Extract field values for this epoch
        epoch_vals = {}
        for var in nav.data_vars:
            val = float(sv_data[var].values[idx])
            if np.isnan(val):
                epoch_vals[var] = 0.0  # Default NaN fields to 0
            else:
                epoch_vals[var] = val

        try:
            if gnss == 'GLO':
                ubx_msg = build_mga_glo_eph(sv_id, epoch_vals, epoch_time)
            else:
                # GPS / QZSS -- Keplerian ephemeris
                _, toc_seconds = datetime64_to_gps_week_toc(epoch_time)
                ubx_msg = convert_epoch(sv_id, epoch_vals, toc_seconds, msg_id)
            messages.append((sv_label, ubx_msg, epoch_time, epoch_vals))
        except (ValueError, struct.error) as e:
            print(f"  Warning: failed to convert {sv_label}: {e}", file=sys.stderr)

    return messages


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(
        description='Convert RINEX navigation files to u-blox MGA ephemeris messages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s brdc0380.26n -o eph.ubx
  %(prog)s brdc0370.26n brdc0370.26q -o eph.ubx
  %(prog)s BRDC00WRD_R_20260390000_01D_MN.rnx -o eph.ubx
  %(prog)s brdc0380.26n -o eph.ubx --time 2026-02-07T16:00
  %(prog)s brdc0380.26n -o eph.ubx --systems GPS
  %(prog)s brdc0380.26n -o eph.ubx --systems GPS,GLO
  %(prog)s brdc0380.26n -o eph.ubx --all-epochs
        """,
    )

    parser.add_argument('input', nargs='+', help='RINEX navigation file(s)')
    parser.add_argument('-o', '--output', required=True, help='Output UBX binary file')
    parser.add_argument('--time', metavar='DATETIME',
        help='Target time for epoch selection (ISO format, e.g. 2026-02-07T16:00). '
             'Default: latest available epoch.')
    parser.add_argument('--max-age', type=float, default=4.0, metavar='HOURS',
        help='Maximum ephemeris age in hours (default: 4.0, GLONASS capped at 1.0)')
    parser.add_argument('--systems', default='GPS,QZSS,GLO',
        help='Comma-separated GNSS systems to include (default: GPS,QZSS,GLO)')
    parser.add_argument('--all-epochs', action='store_true',
        help='Output all epochs, not just the best per satellite')
    parser.add_argument('--verbose', '-v', action='store_true',
        help='Show per-satellite conversion details')

    args = parser.parse_args()

    # Parse systems
    systems = set(s.strip().upper() for s in args.systems.split(','))

    # Parse target time
    target_time = None
    if args.time:
        try:
            target_time = np.datetime64(args.time)
        except ValueError:
            print(f"Error: invalid time format: {args.time}", file=sys.stderr)
            sys.exit(1)

    # Load and convert each input file
    all_messages = []
    all_headers = {}  # Merged RINEX header params from all input files

    for input_file in args.input:
        with _open_rinex(input_file) as rinex_path:
            print(f"Loading {input_file}...", file=sys.stderr)

            # Parse header for iono/UTC parameters
            try:
                hdr = parse_rinex_header(rinex_path)
                all_headers.update(hdr)
            except Exception as e:
                print(f"  Warning: failed to parse header: {e}", file=sys.stderr)

            try:
                nav = load_rinex_nav(rinex_path)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                continue

            svs = [str(s) for s in nav.coords['sv'].values]
            times = nav.coords['time'].values
            print(f"  Satellites: {len(svs)}, epochs: {len(times)}", file=sys.stderr)
            if len(times) > 0:
                print(f"  Time range: {times[0]} to {times[-1]}", file=sys.stderr)
            if target_time is not None:
                print(f"  Target time: {target_time}", file=sys.stderr)

            if len(svs) == 0 or len(times) == 0:
                print("  No satellite data found, skipping.", file=sys.stderr)
                continue

            if args.all_epochs:
                for sv_label in sorted(svs):
                    prefix = sv_label[0]
                    sv_num = int(sv_label[1:])
                    is_glo = False
                    if prefix == 'G' and 'GPS' in systems:
                        msg_id = 0x00
                        sv_id = sv_num
                    elif prefix == 'J' and 'QZSS' in systems:
                        msg_id = 0x05
                        sv_id = sv_num
                    elif prefix == 'R' and 'GLO' in systems:
                        is_glo = True
                        sv_id = sv_num
                    else:
                        continue

                    sv_data = nav.sel(sv=sv_label)
                    sentinel = 'X' if is_glo else 'sqrtA'
                    if sentinel not in sv_data.data_vars:
                        continue
                    valid_mask = ~np.isnan(sv_data[sentinel].values)
                    valid_indices = np.where(valid_mask)[0]

                    for idx in valid_indices:
                        epoch_time = times[idx]
                        epoch_vals = {}
                        for var in nav.data_vars:
                            val = float(sv_data[var].values[idx])
                            epoch_vals[var] = 0.0 if np.isnan(val) else val
                        try:
                            if is_glo:
                                ubx_msg = build_mga_glo_eph(sv_id, epoch_vals,
                                                            epoch_time)
                            else:
                                _, toc_seconds = datetime64_to_gps_week_toc(
                                    epoch_time)
                                ubx_msg = convert_epoch(sv_id, epoch_vals,
                                                        toc_seconds, msg_id)
                            all_messages.append((sv_label, ubx_msg, epoch_time,
                                                epoch_vals))
                        except (ValueError, struct.error) as e:
                            if args.verbose:
                                print(f"  Warning: {sv_label} @ {epoch_time}: {e}",
                                      file=sys.stderr)
            else:
                msgs = convert_rinex(nav, target_time, args.max_age, systems)
                all_messages.extend(msgs)

    if not all_messages:
        print("No ephemeris data converted.", file=sys.stderr)
        sys.exit(1)

    # Collect health values per system from converted EPH messages
    gps_health = {}
    qzss_health = {}
    for sv_label, ubx_msg, epoch_time, epoch_vals in all_messages:
        prefix = sv_label[0]
        sv_num = int(sv_label[1:])
        health_val = int(epoch_vals.get('health', 0))
        if prefix == 'G':
            gps_health[sv_num] = health_val
        elif prefix == 'J':
            qzss_health[sv_num] = health_val

    # Build supplementary messages (HEALTH)
    extra_messages = []
    if gps_health:
        extra_messages.append(('GPS-HEALTH', build_mga_gps_health(gps_health)))
        unhealthy = {sv: h for sv, h in gps_health.items() if h != 0}
        if args.verbose:
            if unhealthy:
                print(f"  GPS health: {len(unhealthy)} unhealthy: "
                      f"{', '.join(f'G{sv:02d}={h}' for sv, h in sorted(unhealthy.items()))}",
                      file=sys.stderr)
            else:
                print(f"  GPS health: all {len(gps_health)} healthy", file=sys.stderr)
    if qzss_health:
        extra_messages.append(('QZSS-HEALTH', build_mga_qzss_health(qzss_health)))
        # QZSS L1C/A and L1C/B are mutually exclusive (IS-QZSS-PNT-004 4.1.2.7);
        # one of bits 0x01 (L1C/B) or 0x10 (L1C/A) is normally set for the
        # non-transmitting signal.  Only flag as unhealthy if other bits are set:
        # 0x20=L1, 0x08=L2, 0x04=L5, 0x02=L1C.
        unhealthy = {sv: h for sv, h in qzss_health.items() if h & 0x2E}
        if args.verbose:
            if unhealthy:
                print(f"  QZSS health: {len(unhealthy)} unhealthy: "
                      f"{', '.join(f'J{sv:02d}={h}' for sv, h in sorted(unhealthy.items()))}",
                      file=sys.stderr)
            else:
                print(f"  QZSS health: all {len(qzss_health)} healthy", file=sys.stderr)

    # Build IONO message from RINEX header (Klobuchar parameters)
    if 'GPSA' in all_headers and 'GPSB' in all_headers and 'GPS' in systems:
        extra_messages.append(('GPS-IONO', build_mga_gps_iono(
            all_headers['GPSA'], all_headers['GPSB'])))
        if args.verbose:
            print(f"  GPS iono: alpha={all_headers['GPSA']}", file=sys.stderr)

    # Build UTC message from RINEX header (GPS-UTC time correction)
    if 'GPUT' in all_headers and 'LEAP' in all_headers and 'GPS' in systems:
        extra_messages.append(('GPS-UTC', build_mga_gps_utc(
            all_headers['GPUT'], all_headers['LEAP'])))
        if args.verbose:
            gput = all_headers['GPUT']
            print(f"  GPS UTC: a0={gput['a0']:.4e} tot={gput['tot']} "
                  f"wnt={gput['wnt']} leap={all_headers['LEAP']}",
                  file=sys.stderr)

    # Build GLONASS time offset message
    if 'GLO' in systems and any(m[0][0] == 'R' for m in all_messages):
        glut = all_headers.get('GLUT')
        glgp = all_headers.get('GLGP')
        if glut or glgp:
            # Use latest GLONASS epoch for day number
            glo_epochs = [t for s, _, t, _ in all_messages if s[0] == 'R']
            latest_glo = max(glo_epochs) if glo_epochs else None
            extra_messages.append(('GLO-TIMEOFFSET',
                build_mga_glo_timeoffset(glut, glgp, latest_glo)))
            if args.verbose:
                parts = []
                if glut:
                    parts.append(f"tauC={glut['a0']:.4e}")
                if glgp:
                    parts.append(f"tauGps={glgp['a0']:.4e}")
                print(f"  GLO timeoffset: {' '.join(parts)}", file=sys.stderr)

    # Write output: EPH messages first, then supplementary (HEALTH, IONO, UTC)
    with open(args.output, 'wb') as f:
        for sv_label, ubx_msg, epoch_time, epoch_vals in all_messages:
            f.write(ubx_msg)
        for name, ubx_msg in extra_messages:
            f.write(ubx_msg)

    total_bytes = (sum(len(m[1]) for m in all_messages)
                   + sum(len(m[1]) for m in extra_messages))
    unique_svs = sorted(set(m[0] for m in all_messages))

    msg_count = len(all_messages) + len(extra_messages)
    print(f"\nConverted {msg_count} messages for {len(unique_svs)} satellites "
          f"({total_bytes} bytes) to {args.output}", file=sys.stderr)
    if extra_messages:
        names = [name for name, _ in extra_messages]
        print(f"  Including: {', '.join(names)}", file=sys.stderr)

    if args.verbose:
        # Separate GPS/QZSS and GLONASS for different column formats
        gps_msgs = [(s, m, t, v) for s, m, t, v in all_messages if s[0] in 'GJ']
        glo_msgs = [(s, m, t, v) for s, m, t, v in all_messages if s[0] == 'R']

        if gps_msgs:
            print(f"\n{'SV':>4s} {'Epoch':>24s} {'IODC':>5s} {'Toe':>7s} "
                  f"{'sqrtA':>11s} {'e':>12s}", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            for sv_label, ubx_msg, epoch_time, epoch_vals in gps_msgs:
                print(f"{sv_label:>4s} {str(epoch_time):>24s} "
                      f"{int(epoch_vals.get('IODC', 0)):5d} "
                      f"{epoch_vals.get('Toe', 0):7.0f} "
                      f"{epoch_vals.get('sqrtA', 0):11.4f} "
                      f"{epoch_vals.get('Eccentricity', 0):12.10f}",
                      file=sys.stderr)

        if glo_msgs:
            print(f"\n{'SV':>4s} {'Epoch':>24s} {'X km':>12s} {'Y km':>12s} "
                  f"{'Z km':>12s} {'tau s':>13s}", file=sys.stderr)
            print("-" * 82, file=sys.stderr)
            for sv_label, ubx_msg, epoch_time, epoch_vals in glo_msgs:
                print(f"{sv_label:>4s} {str(epoch_time):>24s} "
                      f"{epoch_vals.get('X', 0)/1000:12.3f} "
                      f"{epoch_vals.get('Y', 0)/1000:12.3f} "
                      f"{epoch_vals.get('Z', 0)/1000:12.3f} "
                      f"{-epoch_vals.get('SVclockBias', 0):13.7e}",
                      file=sys.stderr)

    # Print output filename to stdout (for scripting)
    print(args.output)


if __name__ == '__main__':
    main()
