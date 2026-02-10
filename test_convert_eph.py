"""
Comprehensive unit tests for convert_eph.py

Tests cover:
  - RINEX 2/3 manual parsers
  - GPS time utilities
  - UBX message construction and checksum
  - URA index lookup
  - Scaling functions (signed, unsigned, angular)
  - MGA-GPS/QZSS-EPH message building
  - MGA-GPS-HEALTH / MGA-QZSS-HEALTH message building
  - MGA-GPS-IONO / MGA-GPS-UTC message building
  - MGA-GLO-EPH / MGA-GLO-TIMEOFFSET message building
  - GLONASS day number computation
  - Epoch selection logic
  - RINEX header parsing
  - End-to-end integration with real RINEX data files
"""

import math
import struct
import sys
import os
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr
from pyubx2 import UBXMessage, SET

# Add repo root to path so we can import the module
sys.path.insert(0, os.path.dirname(__file__))

# Import the module under test
import importlib
convert_eph = importlib.import_module("convert_eph")


# ============================================================
# Helpers
# ============================================================

def parse_ubx_frame(raw):
    """Parse a raw UBX message into (cls, id, payload) and verify checksum."""
    assert raw[0:2] == b'\xb5\x62', "Bad sync bytes"
    cls_id = raw[2]
    msg_id = raw[3]
    length = struct.unpack('<H', raw[4:6])[0]
    payload = raw[6:6+length]
    ck_a_expected, ck_b_expected = raw[6+length], raw[7+length]
    # Verify checksum
    ck_a, ck_b = 0, 0
    for b in raw[2:6+length]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    assert ck_a == ck_a_expected, f"CK_A mismatch: {ck_a} != {ck_a_expected}"
    assert ck_b == ck_b_expected, f"CK_B mismatch: {ck_b} != {ck_b_expected}"
    return cls_id, msg_id, payload


# ============================================================
# RINEX 2 parser tests
# ============================================================

class TestParseRinex2Float:
    def test_normal_float(self):
        assert convert_eph._parse_rinex2_float("  1.234E+05") == pytest.approx(1.234e5)

    def test_fortran_d_exponent(self):
        assert convert_eph._parse_rinex2_float(" 1.234D+05") == pytest.approx(1.234e5)

    def test_lowercase_d(self):
        assert convert_eph._parse_rinex2_float(" 1.234d+05") == pytest.approx(1.234e5)

    def test_negative(self):
        assert convert_eph._parse_rinex2_float("-3.14D-08") == pytest.approx(-3.14e-8)

    def test_empty_string(self):
        assert np.isnan(convert_eph._parse_rinex2_float(""))

    def test_whitespace_only(self):
        assert np.isnan(convert_eph._parse_rinex2_float("     "))


class TestParseRinex2NavRecord:
    """Test parsing of individual RINEX 2 navigation records.

    Record data from actual RINEX 2 files (sys.qzss.go.jp BRDC format).
    """

    # Real GPS RINEX 2 record (G01, 2026-02-09 00:00:00)
    GPS_RECORD = [
        " 1 26  2  9  0  0  0.0 3.345320001245E-04-5.002220859751E-12 0.000000000000E+00\n",
        "    2.900000000000E+01-2.831250000000E+01 4.651622330170E-09-2.685675915079E+00\n",
        "   -1.467764377594E-06 1.488578156568E-03 5.397945642471E-06 5.153630035400E+03\n",
        "    8.640000000000E+04-5.401670932770E-08-2.769294893459E+00 1.490116119385E-08\n",
        "    9.584031772936E-01 2.753437500000E+02 3.752528970751E-02-8.224628303067E-09\n",
        "   -1.357199389945E-10 1.000000000000E+00 2.405000000000E+03 0.000000000000E+00\n",
        "    2.000000000000E+00 0.000000000000E+00-8.847564458847E-09 5.410000000000E+02\n",
        "    7.920600000000E+04 4.000000000000E+00                                      \n",
    ]

    def test_gps_record_sv_label(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        assert result is not None
        sv_label, _, _ = result
        assert sv_label == "G01"

    def test_gps_record_epoch(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        _, epoch, _ = result
        expected = np.datetime64('2026-02-09T00:00:00', 'ns')
        assert epoch == expected

    def test_gps_record_clock_bias(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        _, _, fields = result
        assert fields['SVclockBias'] == pytest.approx(3.345320001245e-04)

    def test_gps_record_sqrtA(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        _, _, fields = result
        assert fields['sqrtA'] == pytest.approx(5153.630035400)

    def test_gps_record_eccentricity(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        _, _, fields = result
        assert fields['Eccentricity'] == pytest.approx(1.488578156568e-03)

    def test_gps_record_health(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        _, _, fields = result
        assert fields['health'] == pytest.approx(0.0)

    def test_gps_record_iodc(self):
        result = convert_eph._parse_rinex2_nav_record(self.GPS_RECORD)
        _, _, fields = result
        assert fields['IODC'] == pytest.approx(541.0)

    # Real QZSS RINEX 2 record (J02, 2026-02-09 00:00:00)
    QZSS_RECORD = [
        "J 2 26  2  9  0  0  0.0-1.121312379837E-06-2.273736754432E-13 0.000000000000E+00\n",
        "     9.300000000000E+01 6.307500000000E+02 2.857261873569E-09 2.204186683897E+00\n",
        "     1.919828355312E-05 7.489704748150E-02 5.487352609634E-06 6.493527595500E+03\n",
        "     8.640000000000E+04 1.812353730202E-06 1.867598885744E+00-1.631677150726E-06\n",
        "     6.909222204152E-01-7.156250000000E+01-1.570852628660E+00-3.267993267894E-09\n",
        "     3.325138505366E-10 2.000000000000E+00 2.405000000000E+03 0.000000000000E+00\n",
        "     2.800000000000E+00 1.000000000000E+00 1.396983861923E-09 8.610000000000E+02\n",
        "     8.280600000000E+04 2.000000000000E+00                                      \n",
    ]

    def test_qzss_j_prefix(self):
        """Test QZSS record with 'J' prefix (offset format)."""
        result = convert_eph._parse_rinex2_nav_record(self.QZSS_RECORD)
        assert result is not None
        sv_label, _, fields = result
        assert sv_label == "J02"
        assert fields['sqrtA'] == pytest.approx(6493.527595500)
        assert fields['Eccentricity'] == pytest.approx(7.489704748150e-02)

    def test_invalid_record(self):
        bad_lines = ["not a valid RINEX record\n"] * 8
        result = convert_eph._parse_rinex2_nav_record(bad_lines)
        assert result is None


class TestParseRinex2Nav:
    """Test full RINEX 2 file parsing."""

    def _make_rinex2_file(self, records_text, header_extra=""):
        """Create a temporary RINEX 2 file."""
        header = textwrap.dedent(f"""\
             2.12           N                                       RINEX VERSION / TYPE
            SPO                 QSS                 20260210 001751 UTC PGM / RUN BY / DATE
            {header_extra}                                                            COMMENT
                                                                        END OF HEADER
        """)
        content = header + records_text
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.26n', delete=False)
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_empty_file(self):
        path = self._make_rinex2_file("")
        try:
            result = convert_eph.parse_rinex2_nav(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_single_record(self):
        record_text = (
            " 1 26  2  9  0  0  0.0 3.345320001245E-04-5.002220859751E-12 0.000000000000E+00\n"
            "    2.900000000000E+01-2.831250000000E+01 4.651622330170E-09-2.685675915079E+00\n"
            "   -1.467764377594E-06 1.488578156568E-03 5.397945642471E-06 5.153630035400E+03\n"
            "    8.640000000000E+04-5.401670932770E-08-2.769294893459E+00 1.490116119385E-08\n"
            "    9.584031772936E-01 2.753437500000E+02 3.752528970751E-02-8.224628303067E-09\n"
            "   -1.357199389945E-10 1.000000000000E+00 2.405000000000E+03 0.000000000000E+00\n"
            "    2.000000000000E+00 0.000000000000E+00-8.847564458847E-09 5.410000000000E+02\n"
            "    7.920600000000E+04 4.000000000000E+00                                      \n"
        )
        path = self._make_rinex2_file(record_text)
        try:
            ds = convert_eph.parse_rinex2_nav(path)
            assert ds is not None
            assert 'G01' in ds.coords['sv'].values
            assert 'sqrtA' in ds.data_vars
            # sqrtA for G01 should be populated
            val = float(ds['sqrtA'].sel(sv='G01').values[0])
            assert val == pytest.approx(5153.630035400)
        finally:
            os.unlink(path)


# ============================================================
# GPS time utilities
# ============================================================

class TestGpsTimeUtilities:
    def test_gps_epoch_itself(self):
        """GPS epoch should be 0 seconds."""
        gps_epoch = np.datetime64('1980-01-06T00:00:00', 'ns')
        assert convert_eph.datetime64_to_gps_seconds(gps_epoch) == pytest.approx(0.0)

    def test_known_gps_time(self):
        """Test a known GPS time conversion."""
        # 2026-01-01T00:00:00 UTC
        dt = np.datetime64('2026-01-01T00:00:00', 'ns')
        total = convert_eph.datetime64_to_gps_seconds(dt)
        # Should be a large positive number (46 years worth of seconds)
        assert total > 1.4e9

    def test_gps_week_toc(self):
        """Test GPS week and TOC computation."""
        dt = np.datetime64('1980-01-06T00:00:00', 'ns')
        week, toc = convert_eph.datetime64_to_gps_week_toc(dt)
        assert week == 0
        assert toc == pytest.approx(0.0)

    def test_gps_week_toc_one_week(self):
        """One week after GPS epoch."""
        dt = np.datetime64('1980-01-13T00:00:00', 'ns')
        week, toc = convert_eph.datetime64_to_gps_week_toc(dt)
        assert week == 1
        assert toc == pytest.approx(0.0)

    def test_gps_week_toc_midweek(self):
        """Mid-week should give non-zero toc."""
        dt = np.datetime64('1980-01-09T12:00:00', 'ns')
        week, toc = convert_eph.datetime64_to_gps_week_toc(dt)
        assert week == 0
        assert toc == pytest.approx(3.5 * 86400)


# ============================================================
# UBX message construction
# ============================================================

class TestUbxChecksum:
    def test_known_checksum(self):
        """Test checksum against known UBX ACK-ACK for CFG-MSG."""
        # ACK-ACK: class=0x05, id=0x01, len=2, payload=0x06 0x01
        data = bytes([0x05, 0x01, 0x02, 0x00, 0x06, 0x01])
        ck = convert_eph.ubx_checksum(data)
        assert ck == bytes([0x0F, 0x38])

    def test_empty_data(self):
        ck = convert_eph.ubx_checksum(b'')
        assert ck == bytes([0x00, 0x00])

    def test_single_byte(self):
        ck = convert_eph.ubx_checksum(bytes([0x42]))
        assert ck == bytes([0x42, 0x42])


class TestCreateUbxMessage:
    def test_sync_bytes(self):
        msg = convert_eph.create_ubx_message(0x13, 0x00, b'\x00' * 4)
        assert msg[0:2] == b'\xb5\x62'

    def test_class_id(self):
        msg = convert_eph.create_ubx_message(0x13, 0x05, b'\x00' * 4)
        assert msg[2] == 0x13
        assert msg[3] == 0x05

    def test_length_field(self):
        payload = b'\x01\x02\x03'
        msg = convert_eph.create_ubx_message(0x13, 0x00, payload)
        length = struct.unpack('<H', msg[4:6])[0]
        assert length == 3

    def test_payload_present(self):
        payload = b'\xDE\xAD\xBE\xEF'
        msg = convert_eph.create_ubx_message(0x13, 0x00, payload)
        assert msg[6:10] == payload

    def test_valid_checksum(self):
        """Verify the message can be parsed back with valid checksum."""
        payload = b'\x01\x02\x03\x04\x05'
        msg = convert_eph.create_ubx_message(0x13, 0x00, payload)
        cls_id, msg_id, parsed_payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00
        assert parsed_payload == payload

    def test_total_length(self):
        payload = b'\x00' * 10
        msg = convert_eph.create_ubx_message(0x13, 0x00, payload)
        # sync(2) + class(1) + id(1) + length(2) + payload(10) + cksum(2) = 18
        assert len(msg) == 18


# ============================================================
# URA lookup
# ============================================================

class TestUraMetersToIndex:
    def test_best_accuracy(self):
        assert convert_eph.ura_meters_to_index(2.0) == 0

    def test_typical_value(self):
        # 2.0m -> URA index 0 (threshold 2.4)
        assert convert_eph.ura_meters_to_index(2.0) == 0

    def test_exact_threshold(self):
        assert convert_eph.ura_meters_to_index(2.4) == 0

    def test_above_threshold(self):
        assert convert_eph.ura_meters_to_index(2.5) == 1

    def test_large_value(self):
        assert convert_eph.ura_meters_to_index(10000) == 15

    def test_nan_returns_zero(self):
        assert convert_eph.ura_meters_to_index(float('nan')) == 0

    def test_index_3(self):
        # threshold[2] = 4.85, threshold[3] = 6.85
        assert convert_eph.ura_meters_to_index(5.0) == 3

    def test_index_boundary(self):
        # Right at threshold[1] = 3.4
        assert convert_eph.ura_meters_to_index(3.4) == 1


# ============================================================
# Scaling functions
# ============================================================

class TestScaleSigned:
    def test_zero(self):
        assert convert_eph.scale_signed(0.0, 2**-31, 32) == 0

    def test_positive(self):
        # 1.0 / 2^-31 = 2^31
        result = convert_eph.scale_signed(1.0, 2**-31, 32)
        # Should be clamped to max signed 32-bit
        assert result == 2**31 - 1

    def test_negative(self):
        result = convert_eph.scale_signed(-1.0, 2**-31, 32)
        assert result == -(2**31)

    def test_small_value(self):
        # 2^-31 / 2^-31 = 1
        result = convert_eph.scale_signed(2**-31, 2**-31, 32)
        assert result == 1

    def test_clamping_positive(self):
        result = convert_eph.scale_signed(1e10, 1.0, 8)
        assert result == 127  # max I8

    def test_clamping_negative(self):
        result = convert_eph.scale_signed(-1e10, 1.0, 8)
        assert result == -128  # min I8

    def test_8bit_range(self):
        result = convert_eph.scale_signed(100.0, 1.0, 8)
        assert result == 100
        result = convert_eph.scale_signed(-100.0, 1.0, 8)
        assert result == -100

    def test_rounding(self):
        # 1.5 / 1.0 = 1.5, should round to 2
        assert convert_eph.scale_signed(1.5, 1.0, 8) == 2


class TestScaleUnsigned:
    def test_zero(self):
        assert convert_eph.scale_unsigned(0.0, 2**-33, 32) == 0

    def test_positive(self):
        result = convert_eph.scale_unsigned(2**-33, 2**-33, 32)
        assert result == 1

    def test_negative_clamped_to_zero(self):
        result = convert_eph.scale_unsigned(-1.0, 1.0, 8)
        assert result == 0

    def test_overflow_clamped(self):
        result = convert_eph.scale_unsigned(1e10, 1.0, 8)
        assert result == 255  # max U8


class TestScaleAngular:
    def test_zero_radians(self):
        assert convert_eph.scale_angular_signed(0.0, 2**-31, 32) == 0

    def test_pi_radians(self):
        # pi radians = 1.0 semi-circles
        result = convert_eph.scale_angular_signed(math.pi, 2**-31, 32)
        expected = round(1.0 / 2**-31)
        assert result == min(expected, 2**31 - 1)

    def test_half_pi(self):
        # pi/2 radians = 0.5 semi-circles
        result = convert_eph.scale_angular_signed(math.pi / 2, 2**-31, 32)
        expected = round(0.5 / 2**-31)
        assert result == expected

    def test_unsigned_angular(self):
        result = convert_eph.scale_angular_unsigned(0.0, 2**-31, 32)
        assert result == 0


# ============================================================
# MGA-GPS/QZSS-EPH message building
# ============================================================

class TestBuildMgaEphPayload:
    """Test the 68-byte MGA-GPS-EPH payload builder."""

    def _make_raw_dict(self):
        return {
            'sv_id': 1,
            'fit_interval': 0,
            'ura_index': 1,
            'sv_health': 0,
            'tgd': -10,
            'iodc': 553,
            'toc': 10350,
            'af2': 0,
            'af1': -1535,
            'af0': -227985,
            'crs': -3970,
            'deltaN': 922,
            'm0': 1352061001,
            'cuc': -346,
            'cus': -378,
            'e': 3205500,
            'sqrtA': 2716110976,
            'toe': 10350,
            'cic': -15,
            'omega0': -1213792850,
            'cis': -16,
            'crc': 6197,
            'i0': 528513960,
            'omega': -490455982,
            'omegaDot': -17260,
            'idot': -157,
        }

    def test_payload_length(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_eph_payload(raw)
        assert len(payload) == 68

    def test_type_byte(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_eph_payload(raw)
        assert payload[0] == 0x01  # type

    def test_version_byte(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_eph_payload(raw)
        assert payload[1] == 0x00  # version

    def test_sv_id(self):
        raw = self._make_raw_dict()
        raw['sv_id'] = 15
        payload = convert_eph.build_mga_eph_payload(raw)
        assert payload[2] == 15

    def test_iodc_encoding(self):
        raw = self._make_raw_dict()
        raw['iodc'] = 553
        payload = convert_eph.build_mga_eph_payload(raw)
        iodc = struct.unpack('<H', payload[8:10])[0]
        assert iodc == 553

    def test_reserved_fields_zero(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_eph_payload(raw)
        assert payload[3] == 0x00   # reserved0
        assert payload[12] == 0x00  # reserved1
        # Last 2 bytes are reserved2
        assert payload[66:68] == b'\x00\x00'


class TestConvertEpoch:
    """Test full RINEX -> UBX message conversion."""

    def _make_epoch_data(self):
        """Create minimal epoch data dict with realistic GPS values."""
        return {
            'IODC': 47.0,
            'health': 0.0,
            'SVacc': 2.0,
            'FitIntvl': 4.0,
            'Toe': 14400.0,
            'TGD': -1.30385e-08,
            'SVclockDriftRate': 0.0,
            'SVclockDrift': -6.594e-12,
            'SVclockBias': -1.062e-04,
            'Crs': -124.0625,
            'DeltaN': 4.245e-09,
            'M0': 2.828,
            'Cuc': -6.488e-06,
            'Eccentricity': 1.493e-03,
            'Cus': -7.084e-06,
            'sqrtA': 5153.630,
            'Cic': -2.794e-08,
            'Omega0': -2.538,
            'Cis': -2.980e-08,
            'Io': 0.986,
            'Crc': 193.656,
            'omega': -1.025,
            'OmegaDot': -7.940e-09,
            'IDOT': -7.210e-11,
        }

    def test_gps_message_structure(self):
        epoch = self._make_epoch_data()
        msg = convert_eph.convert_epoch(1, epoch, 14400.0, 0x00)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00  # GPS
        assert len(payload) == 68

    def test_qzss_message_structure(self):
        epoch = self._make_epoch_data()
        epoch['sqrtA'] = 6493.5
        msg = convert_eph.convert_epoch(1, epoch, 14400.0, 0x05)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x05  # QZSS
        assert len(payload) == 68

    def test_gps_sv_id(self):
        msg = convert_eph.convert_epoch(7, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[2] == 7

    def test_gps_iodc_in_payload(self):
        msg = convert_eph.convert_epoch(1, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        iodc = struct.unpack('<H', payload[8:10])[0]
        assert iodc == 47

    def test_gps_toc_in_payload(self):
        """Toc = 14400s, LSB = 16 -> 900, at offset 10."""
        msg = convert_eph.convert_epoch(1, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        toc = struct.unpack('<H', payload[10:12])[0]
        assert toc == 900

    def test_gps_toe_in_payload(self):
        """Toe = 14400s, LSB = 16 -> 900, at offset 40."""
        msg = convert_eph.convert_epoch(1, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        toe = struct.unpack('<H', payload[40:42])[0]
        assert toe == 900

    def test_gps_health_zero(self):
        msg = convert_eph.convert_epoch(1, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        # svHealth is U1 at offset 6
        assert payload[6] == 0

    def test_gps_sqrtA_encoding(self):
        """sqrtA = 5153.630, LSB = 2^-19."""
        msg = convert_eph.convert_epoch(1, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        # offset: type(1)+ver(1)+svId(1)+res0(1)+fit(1)+ura(1)+health(1)+tgd(1)
        # +iodc(2)+toc(2)+res1(1)+af2(1)+af1(2)+af0(4)+crs(2)+dN(2)+m0(4)+cuc(2)+cus(2)+e(4) = 36
        sqrtA = struct.unpack('<I', payload[36:40])[0]
        expected = round(5153.630 / 2**-19)
        assert sqrtA == expected

    def test_gps_eccentricity_encoding(self):
        """e = 1.493e-03, LSB = 2^-33."""
        msg = convert_eph.convert_epoch(1, self._make_epoch_data(), 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg)
        # e is U4 at offset 32 (after cuc(2)+cus(2) at offset 28)
        e = struct.unpack('<I', payload[32:36])[0]
        expected = round(1.493e-03 / 2**-33)
        assert e == expected


class TestRinexEpochToMgaEph:
    """Test the RINEX epoch -> raw integer dict conversion."""

    def _make_epoch(self):
        return {
            'IODC': 47.0,
            'health': 0.0,
            'SVacc': 2.0,
            'FitIntvl': 4.0,
            'Toe': 14400.0,
            'TGD': -1.30385e-08,
            'SVclockDriftRate': 0.0,
            'SVclockDrift': -6.594e-12,
            'SVclockBias': -1.062e-04,
            'Crs': 0.0,
            'DeltaN': 0.0,
            'M0': 0.0,
            'Cuc': 0.0,
            'Eccentricity': 0.01,
            'Cus': 0.0,
            'sqrtA': 5153.630,
            'Cic': 0.0,
            'Omega0': 0.0,
            'Cis': 0.0,
            'Io': 0.0,
            'Crc': 0.0,
            'omega': 0.0,
            'OmegaDot': 0.0,
            'IDOT': 0.0,
        }

    def test_sv_id_passthrough(self):
        raw = convert_eph.rinex_epoch_to_mga_eph(5, self._make_epoch(), 14400.0)
        assert raw['sv_id'] == 5

    def test_iodc_integer(self):
        raw = convert_eph.rinex_epoch_to_mga_eph(1, self._make_epoch(), 14400.0)
        assert raw['iodc'] == 47

    def test_toe_scaling(self):
        # Toe=14400s, LSB=16 -> 14400/16 = 900
        raw = convert_eph.rinex_epoch_to_mga_eph(1, self._make_epoch(), 14400.0)
        assert raw['toe'] == 900

    def test_toc_scaling(self):
        # toc=14400s, LSB=16 -> 900
        raw = convert_eph.rinex_epoch_to_mga_eph(1, self._make_epoch(), 14400.0)
        assert raw['toc'] == 900

    def test_eccentricity_unsigned(self):
        raw = convert_eph.rinex_epoch_to_mga_eph(1, self._make_epoch(), 14400.0)
        assert raw['e'] > 0

    def test_fit_interval_default(self):
        raw = convert_eph.rinex_epoch_to_mga_eph(1, self._make_epoch(), 14400.0)
        assert raw['fit_interval'] == 0

    def test_fit_interval_extended(self):
        epoch = self._make_epoch()
        epoch['FitIntvl'] = 6.0
        raw = convert_eph.rinex_epoch_to_mga_eph(1, epoch, 14400.0)
        assert raw['fit_interval'] == 1


# ============================================================
# Health message builders
# ============================================================

class TestBuildMgaGpsHealth:
    def test_message_structure(self):
        health = {1: 0, 2: 0, 3: 0}
        msg = convert_eph.build_mga_gps_health(health)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00
        assert len(payload) == 40

    def test_type_byte(self):
        msg = convert_eph.build_mga_gps_health({})
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x04  # type = HEALTH

    def test_healthy_default(self):
        """Missing SVs should default to 0 (healthy)."""
        msg = convert_eph.build_mga_gps_health({})
        _, _, payload = parse_ubx_frame(msg)
        # healthCode starts at offset 4, 32 bytes
        for i in range(32):
            assert payload[4 + i] == 0

    def test_unhealthy_sv(self):
        msg = convert_eph.build_mga_gps_health({5: 0x3F})
        _, _, payload = parse_ubx_frame(msg)
        assert payload[4 + 4] == 0x3F  # SV 5 is at index 4 (0-based)
        assert payload[4 + 0] == 0     # SV 1 should be healthy

    def test_health_mask(self):
        """Health values should be masked to 6 bits."""
        msg = convert_eph.build_mga_gps_health({1: 0xFF})
        _, _, payload = parse_ubx_frame(msg)
        assert payload[4] == 0x3F  # 0xFF & 0x3F


class TestBuildMgaQzssHealth:
    def test_message_structure(self):
        msg = convert_eph.build_mga_qzss_health({})
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x05  # QZSS
        assert len(payload) == 12

    def test_type_byte(self):
        msg = convert_eph.build_mga_qzss_health({})
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x04  # type = HEALTH

    def test_version_and_reserved(self):
        msg = convert_eph.build_mga_qzss_health({})
        _, _, payload = parse_ubx_frame(msg)
        assert payload[1] == 0x00  # version
        assert payload[2:4] == b'\x00\x00'  # reserved
        assert payload[9:12] == b'\x00\x00\x00'  # 3 reserved bytes after 5 health slots

    def test_five_slots(self):
        health = {1: 1, 2: 0, 3: 1, 4: 16, 5: 0}
        msg = convert_eph.build_mga_qzss_health(health)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[4] == 1   # SV 1
        assert payload[5] == 0   # SV 2
        assert payload[6] == 1   # SV 3
        assert payload[7] == 16  # SV 4
        assert payload[8] == 0   # SV 5

    def test_empty_health_all_zero(self):
        msg = convert_eph.build_mga_qzss_health({})
        _, _, payload = parse_ubx_frame(msg)
        for i in range(5):
            assert payload[4 + i] == 0


# ============================================================
# Iono / UTC message builders
# ============================================================

class TestBuildMgaGpsIono:
    def test_message_structure(self):
        gpsa = [1.7695e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07]
        gpsb = [1.2902e+05, -1.1469e+05, 6.5536e+04, -3.2768e+05]
        msg = convert_eph.build_mga_gps_iono(gpsa, gpsb)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00
        assert len(payload) == 16

    def test_type_byte(self):
        gpsa = [0, 0, 0, 0]
        gpsb = [0, 0, 0, 0]
        msg = convert_eph.build_mga_gps_iono(gpsa, gpsb)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x06  # type = IONO

    def test_version_and_reserved(self):
        msg = convert_eph.build_mga_gps_iono([0]*4, [0]*4)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[1] == 0x00  # version
        assert payload[2:4] == b'\x00\x00'  # reserved1
        assert payload[12:16] == b'\x00\x00\x00\x00'  # reserved2

    def test_alpha_encoding(self):
        """Alpha params: scales are 2^-30, 2^-27, 2^-24, 2^-24."""
        # Use exact values that produce known raw integers
        a0 = 19 * 2**-30   # raw = 19
        a1 = -5 * 2**-27   # raw = -5
        a2 = 10 * 2**-24   # raw = 10
        a3 = -20 * 2**-24  # raw = -20
        msg = convert_eph.build_mga_gps_iono([a0, a1, a2, a3], [0]*4)
        _, _, payload = parse_ubx_frame(msg)
        # alpha0-3 are I1 at offsets 4-7
        vals = struct.unpack('<4b', payload[4:8])
        assert vals == (19, -5, 10, -20)

    def test_beta_encoding(self):
        """Beta params: scales are 2^11, 2^14, 2^16, 2^16."""
        b0 = 63 * 2**11    # raw = 63
        b1 = -7 * 2**14    # raw = -7
        b2 = 1 * 2**16     # raw = 1
        b3 = -5 * 2**16    # raw = -5
        msg = convert_eph.build_mga_gps_iono([0]*4, [b0, b1, b2, b3])
        _, _, payload = parse_ubx_frame(msg)
        # beta0-3 are I1 at offsets 8-11
        vals = struct.unpack('<4b', payload[8:12])
        assert vals == (63, -7, 1, -5)

    def test_real_iono_values(self):
        """Test with realistic Klobuchar parameters."""
        gpsa = [1.7695e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07]
        gpsb = [1.2902e+05, -1.1469e+05, 6.5536e+04, -3.2768e+05]
        msg = convert_eph.build_mga_gps_iono(gpsa, gpsb)
        _, _, payload = parse_ubx_frame(msg)
        # Verify alpha values (I1 at offsets 4-7)
        a = struct.unpack('<4b', payload[4:8])
        assert a[0] == round(gpsa[0] / 2**-30)
        assert a[1] == round(gpsa[1] / 2**-27)
        assert a[2] == round(gpsa[2] / 2**-24)
        assert a[3] == round(gpsa[3] / 2**-24)
        # Verify beta values (I1 at offsets 8-11)
        b = struct.unpack('<4b', payload[8:12])
        assert b[0] == round(gpsb[0] / 2**11)
        assert b[1] == round(gpsb[1] / 2**14)
        assert b[2] == round(gpsb[2] / 2**16)
        assert b[3] == round(gpsb[3] / 2**16)


class TestBuildMgaGpsUtc:
    def test_message_structure(self):
        gput = {'a0': -9.3132e-10, 'a1': 0.0, 'tot': 405504, 'wnt': 2405}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00
        assert len(payload) == 20

    def test_type_byte(self):
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x05  # type = UTC

    def test_version_and_reserved(self):
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 0)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[1] == 0x00  # version
        assert payload[2:4] == b'\x00\x00'  # reserved1
        assert payload[18:20] == b'\x00\x00'  # reserved2

    def test_leap_seconds_encoding(self):
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg)
        dtLS = struct.unpack('b', payload[12:13])[0]
        assert dtLS == 18

    def test_a0_encoding(self):
        """utcA0: I4 scaled 2^-30."""
        a0 = -9.3132257462e-10  # = -1 * 2^-30
        gput = {'a0': a0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg)
        a0_raw = struct.unpack('<i', payload[4:8])[0]
        assert a0_raw == -1

    def test_a1_encoding(self):
        """utcA1: I4 scaled 2^-50."""
        a1 = 3 * 2**-50
        gput = {'a0': 0.0, 'a1': a1, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 0)
        _, _, payload = parse_ubx_frame(msg)
        a1_raw = struct.unpack('<i', payload[8:12])[0]
        assert a1_raw == 3

    def test_tot_encoding(self):
        """utcTot: U1 scaled 2^12 (4096s)."""
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 405504, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 0)
        _, _, payload = parse_ubx_frame(msg)
        # 405504 / 4096 = 99
        assert payload[13] == 99

    def test_wnt_encoding(self):
        """utcWNt: U1, 8-bit truncated GPS week."""
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 2405}
        msg = convert_eph.build_mga_gps_utc(gput, 0)
        _, _, payload = parse_ubx_frame(msg)
        # 2405 & 0xFF = 101
        assert payload[14] == 2405 & 0xFF

    def test_dtlsf_mirrors_dtls(self):
        """Future leap second (dtLSF) should equal current (dtLS)."""
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg)
        dtLS = struct.unpack('b', payload[12:13])[0]
        dtLSF = struct.unpack('b', payload[17:18])[0]
        assert dtLS == dtLSF == 18

    def test_wnlsf_and_dn_zero(self):
        """wnlsf and DN not available from RINEX, should be 0."""
        gput = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[15] == 0  # utcWNlsf
        assert payload[16] == 0  # utcDN

    def test_full_payload_layout(self):
        """Validate complete payload with known values."""
        gput = {'a0': -9.3132257462e-10, 'a1': -1.776356839e-15, 'tot': 405504, 'wnt': 2405}
        msg = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x05   # type
        assert payload[1] == 0x00   # version
        a0_raw = struct.unpack('<i', payload[4:8])[0]
        a1_raw = struct.unpack('<i', payload[8:12])[0]
        assert a0_raw == round(-9.3132257462e-10 / 2**-30)
        assert a1_raw == round(-1.776356839e-15 / 2**-50)
        assert struct.unpack('b', payload[12:13])[0] == 18  # dtLS
        assert payload[13] == 99     # tot
        assert payload[14] == 101    # wnt (2405 & 0xFF)


# ============================================================
# GLONASS message builders
# ============================================================

class TestBuildMgaGloEph:
    def _make_epoch_vals(self, **overrides):
        """Create GLO epoch dict with defaults, allowing overrides."""
        vals = {
            'X': -3497734.0, 'Y': -22805977.0, 'Z': 12175361.0,
            'dX': 1234.0, 'dY': -5678.0, 'dZ': 9012.0,
            'dX2': 0.0, 'dY2': 0.0, 'dZ2': 0.0,
            'SVclockBias': -1.3784e-04,
            'SVrelFreqBias': 0.0,
            'health': 0.0,
            'FreqNum': 1.0,
            'AgeOpInfo': 0.0,
        }
        vals.update(overrides)
        return vals

    def _default_epoch_time(self):
        return np.datetime64('2026-02-09T05:15:00', 'ns')

    def test_message_structure(self):
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), self._default_epoch_time())
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x06  # GLO
        assert len(payload) == 48

    def test_type_byte(self):
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x01  # type = EPH

    def test_version_and_reserved(self):
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[1] == 0x00  # version
        assert payload[3] == 0x00  # reserved0
        assert payload[44:48] == b'\x00\x00\x00\x00'  # reserved1

    def test_sv_id(self):
        msg = convert_eph.build_mga_glo_eph(17, self._make_epoch_vals(), self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[2] == 17

    def test_position_encoding(self):
        """Position: meters -> km, scaled by 2^-11."""
        x_m, y_m, z_m = 10000000.0, -20000000.0, 5000000.0
        vals = self._make_epoch_vals(X=x_m, Y=y_m, Z=z_m)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        x_raw, y_raw, z_raw = struct.unpack('<iii', payload[8:20])
        assert x_raw == round((x_m / 1000.0) / 2**-11)
        assert y_raw == round((y_m / 1000.0) / 2**-11)
        assert z_raw == round((z_m / 1000.0) / 2**-11)

    def test_velocity_encoding(self):
        """Velocity: m/s -> km/s, scaled by 2^-20."""
        dx, dy, dz = 1234.0, -5678.0, 9012.0
        vals = self._make_epoch_vals(dX=dx, dY=dy, dZ=dz)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        dx_raw, dy_raw, dz_raw = struct.unpack('<iii', payload[20:32])
        assert dx_raw == round((dx / 1000.0) / 2**-20)
        assert dy_raw == round((dy / 1000.0) / 2**-20)
        assert dz_raw == round((dz / 1000.0) / 2**-20)

    def test_acceleration_encoding(self):
        """Acceleration: m/s^2 -> km/s^2, scaled by 2^-30."""
        ddx, ddy, ddz = 1e-6, -2e-6, 3e-6  # small values in m/s^2
        vals = self._make_epoch_vals(dX2=ddx, dY2=ddy, dZ2=ddz)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        ddx_raw, ddy_raw, ddz_raw = struct.unpack('<bbb', payload[32:35])
        assert ddx_raw == round((ddx / 1000.0) / 2**-30)
        assert ddy_raw == round((ddy / 1000.0) / 2**-30)
        assert ddz_raw == round((ddz / 1000.0) / 2**-30)

    def test_tau_negation(self):
        """RINEX stores -TauN, MGA stores TauN, so tau = -SVclockBias."""
        vals = self._make_epoch_vals(SVclockBias=-1e-4)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        tau_raw = struct.unpack('<i', payload[40:44])[0]
        expected = round(1e-4 / 2**-30)
        assert tau_raw == expected
        assert tau_raw > 0

    def test_tau_positive_clock_bias(self):
        """Positive SVclockBias should yield negative tau_raw."""
        vals = self._make_epoch_vals(SVclockBias=5e-5)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        tau_raw = struct.unpack('<i', payload[40:44])[0]
        expected = round(-5e-5 / 2**-30)
        assert tau_raw == expected
        assert tau_raw < 0

    def test_gamma_encoding(self):
        """SVrelFreqBias -> I2 scaled 2^-40."""
        gamma = 1.818989403546e-12  # = 2 * 2^-40
        vals = self._make_epoch_vals(SVrelFreqBias=gamma)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        gamma_raw = struct.unpack('<h', payload[36:38])[0]
        assert gamma_raw == 2

    def test_tb_encoding(self):
        """tb: UTC epoch -> Moscow time -> 15-min interval index."""
        # 05:15 UTC -> 08:15 Moscow (UTC+3) -> (8*60+15)/15 = 33
        epoch_time = np.datetime64('2026-02-09T05:15:00', 'ns')
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), epoch_time)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[35] == 33

    def test_tb_midnight_wrap(self):
        """23:00 UTC -> 02:00 Moscow next day -> 2*60/15 = 8."""
        epoch_time = np.datetime64('2026-02-09T23:00:00', 'ns')
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), epoch_time)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[35] == 8

    def test_health_encoding(self):
        """B field (health), 3 bits at offset 5."""
        vals = self._make_epoch_vals(health=3.0)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[5] == 3

    def test_freq_num_encoding(self):
        """H field (frequency number), I1 at offset 7."""
        vals = self._make_epoch_vals(FreqNum=-3.0)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        h = struct.unpack('<b', payload[7:8])[0]
        assert h == -3

    def test_ft_default_zero(self):
        """FT (accuracy index) not in RINEX, should default to 0."""
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[4] == 0

    def test_m_type_default_one(self):
        """M (GLONASS-M flag) defaults to 1."""
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[6] == 1

    def test_delta_tau_default_zero(self):
        """deltaTau (L1-L2 difference) not in RINEX, should be 0."""
        msg = convert_eph.build_mga_glo_eph(1, self._make_epoch_vals(), self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert struct.unpack('<b', payload[39:40])[0] == 0

    def test_age_op_info(self):
        """E (age of operation info), U1 at offset 38."""
        vals = self._make_epoch_vals(AgeOpInfo=5.0)
        msg = convert_eph.build_mga_glo_eph(1, vals, self._default_epoch_time())
        _, _, payload = parse_ubx_frame(msg)
        assert payload[38] == 5


class TestComputeGlonassDayNumber:
    def test_leap_year_jan1(self):
        """Jan 1 of a leap year should be day 1."""
        dt = np.datetime64('2024-01-01T00:00:00', 'ns')
        assert convert_eph._compute_glonass_day_number(dt) == 1

    def test_leap_year_jan2(self):
        dt = np.datetime64('2024-01-02T00:00:00', 'ns')
        assert convert_eph._compute_glonass_day_number(dt) == 2

    def test_year_after_leap(self):
        """Jan 1, 2025 should be day 367 (366 days in 2024 + 1)."""
        dt = np.datetime64('2025-01-01T00:00:00', 'ns')
        assert convert_eph._compute_glonass_day_number(dt) == 367

    def test_feb_2026(self):
        """Feb 9, 2026: cycle starts 2024-01-01. Days: 366(2024) + 31(jan25) + 9(feb) + ..."""
        dt = np.datetime64('2026-02-09T12:00:00', 'ns')
        day = convert_eph._compute_glonass_day_number(dt)
        # 2024: 366 days, 2025: 365 days, then jan=31 + feb 9 = 40
        expected = 366 + 365 + 31 + 9
        assert day == expected


class TestBuildMgaGloTimeoffset:
    def test_message_structure(self):
        glut = {'a0': -6.9849e-09}
        glgp = {'a0': 4.6566e-09}
        epoch = np.datetime64('2026-02-09T00:00:00', 'ns')
        msg = convert_eph.build_mga_glo_timeoffset(glut=glut, glgp=glgp, epoch_time=epoch)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x06  # GLO
        assert len(payload) == 20

    def test_type_byte(self):
        msg = convert_eph.build_mga_glo_timeoffset()
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x03  # type = TIMEOFFSET

    def test_version_and_reserved(self):
        msg = convert_eph.build_mga_glo_timeoffset()
        _, _, payload = parse_ubx_frame(msg)
        assert payload[1] == 0x00  # version
        assert payload[16:20] == b'\x00\x00\x00\x00'  # reserved0

    def test_no_params(self):
        """Should work with all None params -- all zeros."""
        msg = convert_eph.build_mga_glo_timeoffset()
        _, _, payload = parse_ubx_frame(msg)
        assert len(payload) == 20
        # N = 0
        assert struct.unpack('<H', payload[2:4])[0] == 0
        # tauC = 0
        assert struct.unpack('<i', payload[4:8])[0] == 0
        # tauGps = 0
        assert struct.unpack('<i', payload[8:12])[0] == 0

    def test_n_day_encoding(self):
        """N: day number within 4-year GLONASS cycle, U2 at offset 2."""
        # 2026-02-09: cycle starts 2024-01-01
        # 366 (2024) + 365 (2025) + 31 (jan) + 9 (feb) = 771
        epoch = np.datetime64('2026-02-09T00:00:00', 'ns')
        msg = convert_eph.build_mga_glo_timeoffset(epoch_time=epoch)
        _, _, payload = parse_ubx_frame(msg)
        n = struct.unpack('<H', payload[2:4])[0]
        assert n == 771

    def test_tau_c_encoding(self):
        """tauC: GLONASS-UTC offset, I4 scaled 2^-27."""
        tau_c = -2 * 2**-27  # exact value -> raw = -2
        msg = convert_eph.build_mga_glo_timeoffset(glut={'a0': tau_c})
        _, _, payload = parse_ubx_frame(msg)
        tau_c_raw = struct.unpack('<i', payload[4:8])[0]
        assert tau_c_raw == -2

    def test_tau_gps_encoding(self):
        """tauGps: GLONASS-GPS offset, I4 scaled 2^-31."""
        tau_gps = 10 * 2**-31  # exact value -> raw = 10
        msg = convert_eph.build_mga_glo_timeoffset(glgp={'a0': tau_gps})
        _, _, payload = parse_ubx_frame(msg)
        tau_gps_raw = struct.unpack('<i', payload[8:12])[0]
        assert tau_gps_raw == 10

    def test_b1_b2_default_zero(self):
        """B1/B2 (UT1-UTC corrections) not in RINEX, should be 0."""
        msg = convert_eph.build_mga_glo_timeoffset()
        _, _, payload = parse_ubx_frame(msg)
        b1 = struct.unpack('<h', payload[12:14])[0]
        b2 = struct.unpack('<h', payload[14:16])[0]
        assert b1 == 0
        assert b2 == 0

    def test_real_values(self):
        """Full payload with realistic RINEX header values."""
        glut = {'a0': -6.9849193096e-09}
        glgp = {'a0': 4.6566128731e-09}
        epoch = np.datetime64('2026-02-09T00:00:00', 'ns')
        msg = convert_eph.build_mga_glo_timeoffset(glut=glut, glgp=glgp, epoch_time=epoch)
        _, _, payload = parse_ubx_frame(msg)
        tau_c_raw = struct.unpack('<i', payload[4:8])[0]
        tau_gps_raw = struct.unpack('<i', payload[8:12])[0]
        assert tau_c_raw == round(glut['a0'] / 2**-27)
        assert tau_gps_raw == round(glgp['a0'] / 2**-31)


# ============================================================
# Epoch selection
# ============================================================

class TestSelectBestEpoch:
    def _make_sv_dataset(self, times, sqrtA_values):
        """Create a minimal xarray Dataset for one satellite."""
        time_arr = np.array(times)
        ds = xr.Dataset(
            data_vars={'sqrtA': (['time'], sqrtA_values)},
            coords={'time': time_arr},
        )
        return ds

    def test_latest_epoch_no_target(self):
        times = [
            np.datetime64('2026-02-09T00:00:00', 'ns'),
            np.datetime64('2026-02-09T02:00:00', 'ns'),
            np.datetime64('2026-02-09T04:00:00', 'ns'),
        ]
        ds = self._make_sv_dataset(times, [5153.0, 5153.0, 5153.0])
        result = convert_eph.select_best_epoch(ds)
        assert result is not None
        idx, epoch = result
        assert idx == 2
        assert epoch == times[2]

    def test_closest_to_target(self):
        times = [
            np.datetime64('2026-02-09T00:00:00', 'ns'),
            np.datetime64('2026-02-09T02:00:00', 'ns'),
            np.datetime64('2026-02-09T04:00:00', 'ns'),
        ]
        ds = self._make_sv_dataset(times, [5153.0, 5153.0, 5153.0])
        target = np.datetime64('2026-02-09T01:50:00', 'ns')
        result = convert_eph.select_best_epoch(ds, target_time=target)
        idx, _ = result
        assert idx == 1  # 02:00 is closest to 01:50

    def test_max_age_filter(self):
        times = [
            np.datetime64('2026-02-09T00:00:00', 'ns'),
        ]
        ds = self._make_sv_dataset(times, [5153.0])
        # Target 8 hours later, max_age=4h -> should return None
        target = np.datetime64('2026-02-09T08:00:00', 'ns')
        result = convert_eph.select_best_epoch(ds, target_time=target, max_age_hours=4.0)
        assert result is None

    def test_nan_values_skipped(self):
        times = [
            np.datetime64('2026-02-09T00:00:00', 'ns'),
            np.datetime64('2026-02-09T02:00:00', 'ns'),
            np.datetime64('2026-02-09T04:00:00', 'ns'),
        ]
        ds = self._make_sv_dataset(times, [np.nan, 5153.0, np.nan])
        result = convert_eph.select_best_epoch(ds)
        idx, _ = result
        assert idx == 1  # Only valid epoch

    def test_all_nan_returns_none(self):
        times = [np.datetime64('2026-02-09T00:00:00', 'ns')]
        ds = self._make_sv_dataset(times, [np.nan])
        result = convert_eph.select_best_epoch(ds)
        assert result is None

    def test_no_known_sentinel_returns_none(self):
        """Dataset without sqrtA or X should return None."""
        times = [np.datetime64('2026-02-09T00:00:00', 'ns')]
        ds = xr.Dataset(
            data_vars={'unknown': (['time'], [1.0])},
            coords={'time': np.array(times)},
        )
        result = convert_eph.select_best_epoch(ds)
        assert result is None


# ============================================================
# RINEX header parser
# ============================================================

class TestParseRinexHeader:
    def _make_rinex_header_file(self, header_lines):
        content = "\n".join(header_lines) + "\n"
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.26n', delete=False)
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_iono_parameters(self):
        lines = [
            "     2.12           N                                       RINEX VERSION / TYPE",
            "GPSA   1.7695E-08 -7.4506E-09 -5.9605E-08  1.1921E-07       IONOSPHERIC CORR    ",
            "GPSB   1.2902E+05 -1.1469E+05  6.5536E+04 -3.2768E+05       IONOSPHERIC CORR    ",
            "                                                            END OF HEADER       ",
        ]
        path = self._make_rinex_header_file(lines)
        try:
            params = convert_eph.parse_rinex_header(path)
            assert 'GPSA' in params
            assert 'GPSB' in params
            assert params['GPSA'][0] == pytest.approx(1.7695e-08)
            assert params['GPSB'][0] == pytest.approx(1.2902e+05)
        finally:
            os.unlink(path)

    def test_utc_parameters(self):
        lines = [
            "     3.05           N                                       RINEX VERSION / TYPE",
            "GPUT -9.3132257462E-10-1.776356839E-15 405504 2405          TIME SYSTEM CORR    ",
            "                                                            END OF HEADER       ",
        ]
        path = self._make_rinex_header_file(lines)
        try:
            params = convert_eph.parse_rinex_header(path)
            assert 'GPUT' in params
            assert params['GPUT']['a0'] == pytest.approx(-9.3132257462e-10)
            assert params['GPUT']['a1'] == pytest.approx(-1.776356839e-15)
            assert params['GPUT']['tot'] == 405504
            assert params['GPUT']['wnt'] == 2405
        finally:
            os.unlink(path)

    def test_leap_seconds(self):
        lines = [
            "     2.12           N                                       RINEX VERSION / TYPE",
            "    18                                                      LEAP SECONDS        ",
            "                                                            END OF HEADER       ",
        ]
        path = self._make_rinex_header_file(lines)
        try:
            params = convert_eph.parse_rinex_header(path)
            assert params['LEAP'] == 18
        finally:
            os.unlink(path)


# ============================================================
# Integration test with real RINEX data
# ============================================================

_TESTDATA = os.path.join(os.path.dirname(__file__), "testdata")


class TestIntegrationWithRealData:
    """Test with actual RINEX files if available in testdata/."""

    QZSS_GPS = os.path.join(_TESTDATA, "brdc0400.26.zip")
    IGS_MIXED = os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")

    @pytest.fixture(autouse=True)
    def _setup_files(self, tmp_path):
        self.tmp_path = tmp_path
        self.gps_file = None
        self.qzss_file = None
        self.igs_file = None

        # Extract QZSS zip if available
        if os.path.exists(self.QZSS_GPS):
            import zipfile
            with zipfile.ZipFile(self.QZSS_GPS, 'r') as zf:
                zf.extractall(tmp_path)
            gps = tmp_path / "brdc0400.26n"
            qzss = tmp_path / "brdc0400.26q"
            if gps.exists():
                self.gps_file = str(gps)
            if qzss.exists():
                self.qzss_file = str(qzss)

        # Extract IGS gzip if available
        if os.path.exists(self.IGS_MIXED):
            import gzip
            import shutil
            out = tmp_path / "BRDC00WRD_R_20260410000_01D_MN.rnx"
            with gzip.open(self.IGS_MIXED, 'rb') as f_in, open(out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            if out.exists():
                self.igs_file = str(out)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "brdc0400.26.zip")),
        reason="QZSS RINEX test files not available")
    def test_qzss_gps_rinex2_loading(self):
        """Test loading GPS RINEX 2 from QZSS site."""
        ds = convert_eph.load_rinex_nav(self.gps_file)
        assert ds is not None
        svs = [s for s in ds.coords['sv'].values if str(s).startswith('G')]
        assert len(svs) >= 20  # Should have most GPS satellites

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "brdc0400.26.zip")),
        reason="QZSS RINEX test files not available")
    def test_qzss_rinex2_loading(self):
        """Test loading QZSS RINEX 2 from QZSS site."""
        ds = convert_eph.load_rinex_nav(self.qzss_file)
        assert ds is not None
        svs = [s for s in ds.coords['sv'].values if str(s).startswith('J')]
        assert len(svs) >= 3

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "brdc0400.26.zip")),
        reason="QZSS RINEX test files not available")
    def test_qzss_gps_full_conversion(self):
        """Test full GPS+QZSS conversion pipeline."""
        ds_gps = convert_eph.load_rinex_nav(self.gps_file)
        ds_qzss = convert_eph.load_rinex_nav(self.qzss_file)
        # Merge
        ds = xr.merge([ds_gps, ds_qzss], join='outer', compat='override')
        results = convert_eph.convert_rinex(ds)
        assert len(results) > 25  # At least 28 GPS + some QZSS

        # Verify all messages are valid UBX (convert_rinex returns tuples)
        for _, msg_bytes, _, _ in results:
            cls_id, _, _ = parse_ubx_frame(msg_bytes)
            assert cls_id == 0x13

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")),
        reason="IGS RINEX test file not available")
    def test_igs_mixed_rinex3_loading(self):
        """Test loading mixed RINEX 3 from IGS/BKG."""
        ds = convert_eph.load_rinex_nav(self.igs_file)
        assert ds is not None
        svs = list(ds.coords['sv'].values)
        g_count = sum(1 for s in svs if str(s).startswith('G'))
        r_count = sum(1 for s in svs if str(s).startswith('R'))
        assert g_count >= 28
        assert r_count >= 15

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")),
        reason="IGS RINEX test file not available")
    def test_igs_mixed_full_conversion(self):
        """Test full GPS+QZSS+GLONASS conversion from mixed RINEX 3."""
        ds = convert_eph.load_rinex_nav(self.igs_file)
        results = convert_eph.convert_rinex(ds)
        # Should have GPS + QZSS + GLONASS ephemeris messages
        assert len(results) >= 50

        # Check we have all three GNSS types
        msg_ids = set()
        for _, msg_bytes, _, _ in results:
            _, msg_id, _ = parse_ubx_frame(msg_bytes)
            msg_ids.add(msg_id)
        assert 0x00 in msg_ids  # GPS
        assert 0x06 in msg_ids  # GLONASS
        # QZSS (0x05) may or may not be present depending on file

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "brdc0400.26.zip")),
        reason="QZSS RINEX test files not available")
    def test_rinex_header_parsing(self):
        """Test header parsing from real GPS RINEX 2 file."""
        params = convert_eph.parse_rinex_header(self.gps_file)
        assert 'GPSA' in params
        assert 'GPSB' in params
        assert len(params['GPSA']) == 4
        assert len(params['GPSB']) == 4


# ============================================================
# pyubx2 cross-validation
# ============================================================

class TestPyubx2CrossValidation:
    """Cross-validate UBX message construction against pyubx2 independent parser.

    Each test builds a message with project code, parses the payload with
    pyubx2 in SET mode, and verifies that pyubx2's decoded field values match
    expectations.

    pyubx2 applies scale factors to raw integers (e.g. msg.af0 = raw * 2^-31).
    For angular fields (m0, omega0, etc.) pyubx2 returns semi-circles, not
    radians.
    """

    def test_mga_gps_eph(self):
        """Build GPS EPH from RINEX-like values, parse with pyubx2."""
        epoch = {
            'IODC': 47.0, 'health': 0.0, 'SVacc': 2.0, 'FitIntvl': 4.0,
            'Toe': 14400.0, 'TGD': -1.30385e-08,
            'SVclockDriftRate': 0.0, 'SVclockDrift': -6.594e-12,
            'SVclockBias': -1.062e-04,
            'Crs': -124.0625, 'DeltaN': 4.245e-09, 'M0': 2.828,
            'Cuc': -6.488e-06, 'Eccentricity': 1.493e-03, 'Cus': -7.084e-06,
            'sqrtA': 5153.630,
            'Cic': -2.794e-08, 'Omega0': -2.538, 'Cis': -2.980e-08,
            'Io': 0.986, 'Crc': 193.656, 'omega': -1.025,
            'OmegaDot': -7.940e-09, 'IDOT': -7.210e-11,
        }
        msg_bytes = convert_eph.convert_epoch(1, epoch, 14400.0, 0x00)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x00, SET, payload=payload)

        assert parsed.identity == "MGA-GPS-EPH"
        assert parsed.svId == 1
        assert parsed.svHealth == 0
        assert parsed.iodc == 47
        assert parsed.toc == pytest.approx(14400.0)
        assert parsed.toe == pytest.approx(14400.0)
        assert parsed.af0 == pytest.approx(-1.062e-04, rel=1e-4)
        assert parsed.sqrtA == pytest.approx(5153.630, rel=1e-6)
        assert parsed.e == pytest.approx(1.493e-03, rel=1e-4)
        # m0 in semi-circles (pyubx2 does not multiply by pi)
        assert parsed.m0 == pytest.approx(2.828 / math.pi, rel=1e-6)

    def test_mga_qzss_eph(self):
        """Build QZSS EPH, parse with pyubx2."""
        epoch = {
            'IODC': 47.0, 'health': 0.0, 'SVacc': 2.0, 'FitIntvl': 4.0,
            'Toe': 14400.0, 'TGD': -1.30385e-08,
            'SVclockDriftRate': 0.0, 'SVclockDrift': -6.594e-12,
            'SVclockBias': -1.062e-04,
            'Crs': -124.0625, 'DeltaN': 4.245e-09, 'M0': 2.828,
            'Cuc': -6.488e-06, 'Eccentricity': 1.493e-03, 'Cus': -7.084e-06,
            'sqrtA': 6493.5,
            'Cic': -2.794e-08, 'Omega0': -2.538, 'Cis': -2.980e-08,
            'Io': 0.986, 'Crc': 193.656, 'omega': -1.025,
            'OmegaDot': -7.940e-09, 'IDOT': -7.210e-11,
        }
        msg_bytes = convert_eph.convert_epoch(1, epoch, 14400.0, 0x05)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x05, SET, payload=payload)

        assert parsed.identity == "MGA-QZSS-EPH"
        assert parsed.svId == 1
        assert parsed.sqrtA == pytest.approx(6493.5, rel=1e-6)
        assert parsed.e == pytest.approx(1.493e-03, rel=1e-4)
        assert parsed.toc == pytest.approx(14400.0)
        assert parsed.toe == pytest.approx(14400.0)

    def test_mga_glo_eph(self):
        """Build GLONASS EPH, parse with pyubx2."""
        epoch_vals = {
            'X': -3497734.0, 'Y': -22805977.0, 'Z': 12175361.0,
            'dX': 1234.0, 'dY': -5678.0, 'dZ': 9012.0,
            'dX2': 0.0, 'dY2': 0.0, 'dZ2': 0.0,
            'SVclockBias': -1.3784e-04,
            'SVrelFreqBias': 0.0,
            'health': 0.0,
            'FreqNum': 1.0,
            'AgeOpInfo': 0.0,
        }
        epoch_time = np.datetime64('2026-02-09T05:15:00', 'ns')
        msg_bytes = convert_eph.build_mga_glo_eph(1, epoch_vals, epoch_time)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x06, SET, payload=payload)

        assert parsed.identity == "MGA-GLO-EPH"
        assert parsed.svId == 1
        # Position in km (pyubx2 scale: raw * 2^-11)
        assert parsed.x == pytest.approx(-3497734.0 / 1000.0, rel=1e-5)
        assert parsed.y == pytest.approx(-22805977.0 / 1000.0, rel=1e-5)
        assert parsed.z == pytest.approx(12175361.0 / 1000.0, rel=1e-5)
        # Velocity in km/s (pyubx2 scale: raw * 2^-20)
        assert parsed.dx == pytest.approx(1234.0 / 1000.0, rel=1e-5)
        assert parsed.dy == pytest.approx(-5678.0 / 1000.0, rel=1e-5)
        assert parsed.dz == pytest.approx(9012.0 / 1000.0, rel=1e-5)
        # tau: -SVclockBias, scaled by 2^-30
        assert parsed.tau == pytest.approx(1.3784e-04, rel=1e-4)
        # tb: in minutes (raw * 15); 05:15 UTC -> 08:15 Moscow -> index 33
        assert parsed.tb == pytest.approx(33 * 15)
        assert parsed.H == 1

    def test_mga_gps_health(self):
        """Build GPS health with unhealthy SV, parse with pyubx2."""
        msg_bytes = convert_eph.build_mga_gps_health({5: 0x3F})
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x00, SET, payload=payload)

        assert parsed.identity == "MGA-GPS-HEALTH"
        assert parsed.healthCode_05 == 0x3F  # SV 5 unhealthy
        assert parsed.healthCode_01 == 0     # SV 1 healthy

    def test_mga_qzss_health(self):
        """Build QZSS health, parse with pyubx2."""
        health = {1: 1, 2: 0, 3: 1, 4: 16, 5: 0}
        msg_bytes = convert_eph.build_mga_qzss_health(health)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x05, SET, payload=payload)

        assert parsed.identity == "MGA-QZSS-HEALTH"
        assert parsed.healthCode_01 == 1
        assert parsed.healthCode_02 == 0
        assert parsed.healthCode_03 == 1
        assert parsed.healthCode_04 == 16
        assert parsed.healthCode_05 == 0

    def test_mga_gps_iono(self):
        """Build GPS iono, parse with pyubx2, validate Klobuchar parameters."""
        # Use exact values that produce known raw integers
        a0 = 19 * 2**-30
        a1 = -5 * 2**-27
        b0 = 63 * 2**11
        b1 = -7 * 2**14
        gpsa = [a0, a1, 0, 0]
        gpsb = [b0, b1, 0, 0]
        msg_bytes = convert_eph.build_mga_gps_iono(gpsa, gpsb)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x00, SET, payload=payload)

        assert parsed.identity == "MGA-GPS-IONO"
        assert parsed.ionoAlpha0 == pytest.approx(a0)
        assert parsed.ionoAlpha1 == pytest.approx(a1)
        assert parsed.ionoBeta0 == pytest.approx(b0)
        assert parsed.ionoBeta1 == pytest.approx(b1)

    def test_mga_gps_utc(self):
        """Build GPS UTC, parse with pyubx2."""
        gput = {'a0': -9.3132257462e-10, 'a1': 0.0, 'tot': 405504, 'wnt': 2405}
        msg_bytes = convert_eph.build_mga_gps_utc(gput, 18)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x00, SET, payload=payload)

        assert parsed.identity == "MGA-GPS-UTC"
        assert parsed.utcA0 == pytest.approx(-9.3132257462e-10, abs=2**-30)
        assert parsed.utcDtLS == 18
        # utcTot: raw=99, scale=2^12 -> 99*4096=405504
        assert parsed.utcTot == pytest.approx(405504, rel=1e-4)
        assert parsed.utcWNt == 2405 & 0xFF

    def test_mga_glo_timeoffset(self):
        """Build GLO timeoffset, parse with pyubx2."""
        glut = {'a0': -6.9849193096e-09}
        glgp = {'a0': 4.6566128731e-09}
        epoch = np.datetime64('2026-02-09T00:00:00', 'ns')
        msg_bytes = convert_eph.build_mga_glo_timeoffset(
            glut=glut, glgp=glgp, epoch_time=epoch)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x06, SET, payload=payload)

        assert parsed.identity == "MGA-GLO-TIMEOFFSET"
        assert parsed.N == 771
        assert parsed.tauC == pytest.approx(glut['a0'], abs=2**-27)
        assert parsed.tauGps == pytest.approx(glgp['a0'], abs=2**-31)

    def test_mga_gal_eph(self):
        """Build Galileo EPH from RINEX-like values, parse with pyubx2."""
        epoch = {
            'IODnav': 77.0, 'health': 0.0, 'SISA': 3.12,
            'Toe': 14400.0, 'BGDe5b': -4.656612873e-10,
            'SVclockDriftRate': 0.0, 'SVclockDrift': -6.594e-12,
            'SVclockBias': -1.062e-04, 'BGDe5a': 0.0,
            'DataSrc': 516.0, 'GALWeek': 1357.0,
            'Crs': -124.0625, 'DeltaN': 4.245e-09, 'M0': 2.828,
            'Cuc': -6.488e-06, 'Eccentricity': 1.493e-03, 'Cus': -7.084e-06,
            'sqrtA': 5440.600,
            'Cic': -2.794e-08, 'Omega0': -2.538, 'Cis': -2.980e-08,
            'Io': 0.986, 'Crc': 193.656, 'omega': -1.025,
            'OmegaDot': -7.940e-09, 'IDOT': -7.210e-11,
            'spare0': 0.0, 'spare1': 0.0, 'TransTime': 0.0,
        }
        msg_bytes = convert_eph.convert_gal_epoch(1, epoch, 14400.0)
        _, _, payload = parse_ubx_frame(msg_bytes)
        parsed = UBXMessage(0x13, 0x02, SET, payload=payload)

        assert parsed.identity == "MGA-GAL-EPH"
        assert parsed.svId == 1
        assert parsed.iodNav == 77
        # toe: 14400/60 = 240 raw, *60 = 14400 back
        assert parsed.toe == pytest.approx(14400.0)
        # toc: 14400/60 = 240 raw, *60 = 14400 back
        assert parsed.toc == pytest.approx(14400.0)
        # af0: scale 2^-34
        assert parsed.af0 == pytest.approx(-1.062e-04, rel=1e-4)
        # af1: scale 2^-46, I4
        assert parsed.af1 == pytest.approx(-6.594e-12, rel=1e-4)
        assert parsed.sqrtA == pytest.approx(5440.600, rel=1e-6)
        assert parsed.e == pytest.approx(1.493e-03, rel=1e-4)
        # m0 in semi-circles
        assert parsed.m0 == pytest.approx(2.828 / math.pi, rel=1e-6)
        assert parsed.healthE1B == 0
        assert parsed.dataValidityE1B == 0
        assert parsed.healthE5b == 0
        assert parsed.dataValidityE5b == 0


# ============================================================
# Galileo SISA index lookup
# ============================================================

class TestSisaMetersToIndex:
    def test_zero(self):
        assert convert_eph.sisa_meters_to_index(0.0) == 0

    def test_exact_1cm(self):
        assert convert_eph.sisa_meters_to_index(0.01) == 1

    def test_exact_49cm(self):
        assert convert_eph.sisa_meters_to_index(0.49) == 49

    def test_exact_50cm(self):
        """50 cm is first value of 2cm step range."""
        assert convert_eph.sisa_meters_to_index(0.50) == 50

    def test_exact_98cm(self):
        """98 cm = 50 + (74-50)*2 -> index 74."""
        assert convert_eph.sisa_meters_to_index(0.98) == 74

    def test_exact_100cm(self):
        """100 cm = first value of 4cm step range -> index 75."""
        assert convert_eph.sisa_meters_to_index(1.00) == 75

    def test_exact_196cm(self):
        """196 cm = 100 + (99-75)*4 -> index 99."""
        assert convert_eph.sisa_meters_to_index(1.96) == 99

    def test_exact_200cm(self):
        """200 cm = first value of 16cm step range -> index 100."""
        assert convert_eph.sisa_meters_to_index(2.00) == 100

    def test_typical_312cm(self):
        """3.12m = 312 cm -> index 100 + round((312-200)/16) = 107."""
        assert convert_eph.sisa_meters_to_index(3.12) == 107

    def test_exact_600cm(self):
        """600 cm = 200 + (125-100)*16 -> index 125."""
        assert convert_eph.sisa_meters_to_index(6.00) == 125

    def test_above_max(self):
        """Values above 600 cm -> NAPA (255)."""
        assert convert_eph.sisa_meters_to_index(10.0) == 255

    def test_nan_returns_napa(self):
        assert convert_eph.sisa_meters_to_index(float('nan')) == 255

    def test_negative(self):
        assert convert_eph.sisa_meters_to_index(-1.0) == 0


# ============================================================
# Galileo MGA-GAL-EPH message building
# ============================================================

class TestRinexEpochToMgaGalEph:
    """Test the RINEX epoch -> raw integer dict conversion for Galileo."""

    def _make_epoch(self):
        return {
            'IODnav': 77.0, 'health': 0.0, 'SISA': 3.12,
            'Toe': 14400.0, 'BGDe5b': -4.656612873e-10,
            'SVclockDriftRate': 0.0, 'SVclockDrift': -6.594e-12,
            'SVclockBias': -1.062e-04, 'BGDe5a': 0.0,
            'DataSrc': 516.0, 'GALWeek': 1357.0,
            'Crs': 0.0, 'DeltaN': 0.0, 'M0': 0.0,
            'Cuc': 0.0, 'Eccentricity': 0.01, 'Cus': 0.0,
            'sqrtA': 5440.600,
            'Cic': 0.0, 'Omega0': 0.0, 'Cis': 0.0,
            'Io': 0.0, 'Crc': 0.0, 'omega': 0.0,
            'OmegaDot': 0.0, 'IDOT': 0.0,
            'spare0': 0.0, 'spare1': 0.0, 'TransTime': 0.0,
        }

    def test_iod_nav(self):
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        assert raw['iodNav'] == 77

    def test_toe_scaling_60s(self):
        """Toe=14400s, LSB=60 -> 14400/60 = 240."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        assert raw['toe'] == 240

    def test_toc_scaling_60s(self):
        """Toc=14400s, LSB=60 -> 14400/60 = 240."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        assert raw['toc'] == 240

    def test_af0_scaling(self):
        """af0: scale 2^-34 for Galileo (not 2^-31 like GPS)."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        expected = round(-1.062e-04 / 2**-34)
        assert raw['af0'] == expected

    def test_af1_scaling(self):
        """af1: scale 2^-46 for Galileo, I4 (not 2^-43 I2 like GPS)."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        expected = round(-6.594e-12 / 2**-46)
        assert raw['af1'] == expected

    def test_af2_scaling(self):
        """af2: scale 2^-59 for Galileo (not 2^-55 like GPS)."""
        epoch = self._make_epoch()
        epoch['SVclockDriftRate'] = 2**-59  # exact 1 LSB
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, epoch, 14400.0)
        assert raw['af2'] == 1

    def test_bgd_e1e5b_scaling(self):
        """BGDe5b: scale 2^-32, I2."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        expected = round(-4.656612873e-10 / 2**-32)
        assert raw['bgdE1E5b'] == expected

    def test_sisa_index(self):
        """SISA=3.12m -> index 107."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        assert raw['sisaIndex'] == 107

    def test_health_decomposition_healthy(self):
        """Health=0 -> all zero."""
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        assert raw['healthE1B'] == 0
        assert raw['dataValidityE1B'] == 0
        assert raw['healthE5b'] == 0
        assert raw['dataValidityE5b'] == 0

    def test_health_decomposition_e1b(self):
        """Health with E1B status bits set."""
        epoch = self._make_epoch()
        epoch['health'] = 0x07  # bits 0-2: dataValidity=1, healthE1B=3
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, epoch, 14400.0)
        assert raw['dataValidityE1B'] == 1    # bit 0
        assert raw['healthE1B'] == 3           # bits 1-2

    def test_health_decomposition_e5b(self):
        """Health with E5b status bits set."""
        epoch = self._make_epoch()
        epoch['health'] = 0x1C0  # bits 6-8: dataValidityE5b=1, healthE5b=3
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, epoch, 14400.0)
        assert raw['dataValidityE5b'] == 1    # bit 6
        assert raw['healthE5b'] == 3           # bits 7-8

    def test_eccentricity_unsigned(self):
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(1, self._make_epoch(), 14400.0)
        assert raw['e'] > 0

    def test_sv_id_passthrough(self):
        raw = convert_eph.rinex_epoch_to_mga_gal_eph(25, self._make_epoch(), 14400.0)
        assert raw['sv_id'] == 25


class TestBuildMgaGalEphPayload:
    """Test the 76-byte MGA-GAL-EPH payload builder."""

    def _make_raw_dict(self):
        return {
            'sv_id': 1,
            'iodNav': 77,
            'deltaN': 922,
            'm0': 1352061001,
            'e': 3205500,
            'sqrtA': 2856591360,
            'omega0': -1213792850,
            'i0': 528513960,
            'omega': -490455982,
            'omegaDot': -17260,
            'idot': -157,
            'cuc': -346,
            'cus': -378,
            'crc': 6197,
            'crs': -3970,
            'cic': -15,
            'cis': -16,
            'toe': 240,
            'af0': -1826489,
            'af1': -94,
            'af2': 0,
            'sisaIndex': 107,
            'toc': 240,
            'bgdE1E5b': -2,
            'healthE1B': 0,
            'dataValidityE1B': 0,
            'healthE5b': 0,
            'dataValidityE5b': 0,
        }

    def test_payload_length(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert len(payload) == 76

    def test_type_byte(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert payload[0] == 0x01  # type

    def test_version_byte(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert payload[1] == 0x00  # version

    def test_sv_id(self):
        raw = self._make_raw_dict()
        raw['sv_id'] = 25
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert payload[2] == 25

    def test_iodnav_encoding(self):
        raw = self._make_raw_dict()
        raw['iodNav'] = 77
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        iodnav = struct.unpack('<H', payload[4:6])[0]
        assert iodnav == 77

    def test_reserved_fields_zero(self):
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert payload[3] == 0x00   # reserved0
        # reserved1 at offset 66 (U2)
        assert payload[66:68] == b'\x00\x00'
        # reserved2 at offset 72 (U4)
        assert payload[72:76] == b'\x00\x00\x00\x00'

    def test_toe_at_correct_offset(self):
        """toe is U2 at offset 50.

        Layout: type(1)+ver(1)+svId(1)+res0(1)+iodNav(2)+deltaN(2)+m0(4)
        +e(4)+sqrtA(4)+omega0(4)+i0(4)+omega(4)+omegaDot(4)
        +iDot(2)+cuc(2)+cus(2)+crc(2)+crs(2)+cic(2)+cis(2) = 50
        """
        raw = self._make_raw_dict()
        raw['toe'] = 240
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        toe = struct.unpack('<H', payload[50:52])[0]
        assert toe == 240

    def test_af0_at_correct_offset(self):
        """af0 is I4 at offset 52."""
        raw = self._make_raw_dict()
        raw['af0'] = -1826489
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        af0 = struct.unpack('<i', payload[52:56])[0]
        assert af0 == -1826489

    def test_af1_at_correct_offset(self):
        """af1 is I4 at offset 56 (not I2 like GPS!)."""
        raw = self._make_raw_dict()
        raw['af1'] = -94
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        af1 = struct.unpack('<i', payload[56:60])[0]
        assert af1 == -94

    def test_sisa_and_toc_offsets(self):
        """sisaIndex (U1) at 61, toc (U2) at 62."""
        raw = self._make_raw_dict()
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert payload[61] == 107   # sisaIndex
        toc = struct.unpack('<H', payload[62:64])[0]
        assert toc == 240           # toc

    def test_bgd_encoding(self):
        """bgdE1E5b (I2) at offset 64."""
        raw = self._make_raw_dict()
        raw['bgdE1E5b'] = -2
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        bgd = struct.unpack('<h', payload[64:66])[0]
        assert bgd == -2

    def test_health_fields(self):
        """Health fields at offsets 68-71."""
        raw = self._make_raw_dict()
        raw['healthE1B'] = 1
        raw['dataValidityE1B'] = 1
        raw['healthE5b'] = 2
        raw['dataValidityE5b'] = 1
        payload = convert_eph.build_mga_gal_eph_payload(raw)
        assert payload[68] == 1  # healthE1B
        assert payload[69] == 1  # dataValidityE1B
        assert payload[70] == 2  # healthE5b
        assert payload[71] == 1  # dataValidityE5b


# ============================================================
# Galileo supplementary message builders
# ============================================================

class TestBuildMgaGalTimeoffset:
    def test_message_structure(self):
        gagp = {'a0': -3.49e-09, 'a1': 0.0, 'tot': 122400, 'wnt': 2405}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x02  # GAL
        assert len(payload) == 10

    def test_type_byte(self):
        gagp = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x03  # type = TIMEOFFSET

    def test_version_and_reserved(self):
        gagp = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[1] == 0x00  # version
        assert payload[2:4] == b'\x00\x00'  # reserved0

    def test_a0g_encoding(self):
        """a0G: I2 scaled 2^-35."""
        a0 = -3 * 2**-35  # exact -> raw = -3
        gagp = {'a0': a0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        _, _, payload = parse_ubx_frame(msg)
        a0g_raw = struct.unpack('<h', payload[4:6])[0]
        assert a0g_raw == -3

    def test_a1g_encoding(self):
        """a1G: I2 scaled 2^-51."""
        a1 = 5 * 2**-51  # exact -> raw = 5
        gagp = {'a0': 0.0, 'a1': a1, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        _, _, payload = parse_ubx_frame(msg)
        a1g_raw = struct.unpack('<h', payload[6:8])[0]
        assert a1g_raw == 5

    def test_t0g_encoding(self):
        """t0G: U1, in units of 3600 seconds."""
        gagp = {'a0': 0.0, 'a1': 0.0, 'tot': 122400, 'wnt': 0}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        _, _, payload = parse_ubx_frame(msg)
        # 122400 / 3600 = 34
        assert payload[8] == 34

    def test_wn0g_encoding(self):
        """wn0G: U1, 8-bit truncated week."""
        gagp = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 2405}
        msg = convert_eph.build_mga_gal_timeoffset(gagp)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[9] == 2405 & 0xFF


class TestBuildMgaGalUtc:
    def test_message_structure(self):
        gaut = {'a0': -9.3132e-10, 'a1': 0.0, 'tot': 405504, 'wnt': 2405}
        msg = convert_eph.build_mga_gal_utc(gaut, 18)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x02  # GAL
        assert len(payload) == 20

    def test_type_byte(self):
        gaut = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_utc(gaut, 0)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x05  # type = UTC

    def test_a0_encoding(self):
        """utcA0: I4 scaled 2^-30."""
        a0 = -1 * 2**-30  # exact -> raw = -1
        gaut = {'a0': a0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_utc(gaut, 0)
        _, _, payload = parse_ubx_frame(msg)
        a0_raw = struct.unpack('<i', payload[4:8])[0]
        assert a0_raw == -1

    def test_a1_encoding(self):
        """utcA1: I4 scaled 2^-50."""
        a1 = 3 * 2**-50
        gaut = {'a0': 0.0, 'a1': a1, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_utc(gaut, 0)
        _, _, payload = parse_ubx_frame(msg)
        a1_raw = struct.unpack('<i', payload[8:12])[0]
        assert a1_raw == 3

    def test_leap_seconds_encoding(self):
        gaut = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_utc(gaut, 18)
        _, _, payload = parse_ubx_frame(msg)
        dtLS = struct.unpack('b', payload[12:13])[0]
        assert dtLS == 18

    def test_tot_encoding(self):
        """utcTot: U1 scaled 2^12 (4096s)."""
        gaut = {'a0': 0.0, 'a1': 0.0, 'tot': 405504, 'wnt': 0}
        msg = convert_eph.build_mga_gal_utc(gaut, 0)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[13] == 99  # 405504/4096 = 99

    def test_dtlsf_mirrors_dtls(self):
        """Future leap second should equal current."""
        gaut = {'a0': 0.0, 'a1': 0.0, 'tot': 0, 'wnt': 0}
        msg = convert_eph.build_mga_gal_utc(gaut, 18)
        _, _, payload = parse_ubx_frame(msg)
        dtLS = struct.unpack('b', payload[12:13])[0]
        dtLSF = struct.unpack('b', payload[17:18])[0]
        assert dtLS == dtLSF == 18


# ============================================================
# RINEX 3 Galileo parser
# ============================================================

class TestRinex3GalParser:
    """Test RINEX 3 Galileo record parsing with embedded data."""

    GAL_RECORD = [
        "E01 2026 02 10 00 10 00-7.564918976277E-04-6.394884621841E-12 0.000000000000E+00\n",
        "     7.700000000000E+01 1.393750000000E+01 2.649484508262E-09 2.690457236932E+00\n",
        "     7.376074790955E-07 1.646387972869E-04 1.032464206219E-05 5.440588268280E+03\n",
        "     8.760000000000E+04 5.587935447693E-08 2.855773754591E+00-3.166496753693E-08\n",
        "     9.829497879629E-01 1.535625000000E+02-6.766953849610E-01-5.238527953445E-09\n",
        "    -3.428681507379E-10 5.160000000000E+02 1.357000000000E+03 0.000000000000E+00\n",
        "     3.120000000000E+00 0.000000000000E+00-4.656612873077E-10-5.122274160385E-10\n",
        "     8.826400000000E+04 0.000000000000E+00                                      \n",
    ]

    def test_gal_record_parsing(self):
        """Parse a Galileo RINEX 3 record with GAL field names."""
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        assert result is not None
        sv_label, epoch, _ = result
        assert sv_label == "E01"
        assert epoch == np.datetime64('2026-02-10T00:10:00', 'ns')

    def test_gal_iodnav_field(self):
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        _, _, fields = result
        assert fields['IODnav'] == pytest.approx(77.0)

    def test_gal_sisa_field(self):
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        _, _, fields = result
        assert fields['SISA'] == pytest.approx(3.12)

    def test_gal_bgd_field(self):
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        _, _, fields = result
        assert fields['BGDe5b'] == pytest.approx(-5.122274160385e-10)
        assert fields['BGDe5a'] == pytest.approx(-4.656612873077e-10)

    def test_gal_sqrtA_field(self):
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        _, _, fields = result
        assert fields['sqrtA'] == pytest.approx(5440.588268280)

    def test_gal_datasrc_field(self):
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        _, _, fields = result
        assert fields['DataSrc'] == pytest.approx(516.0)

    def test_gal_galweek_field(self):
        result = convert_eph._parse_rinex3_nav_record(
            self.GAL_RECORD, 'E', field_names=convert_eph.RINEX3_GAL_FIELDS)
        _, _, fields = result
        assert fields['GALWeek'] == pytest.approx(1357.0)

    def test_parse_rinex3_nav_with_galileo(self):
        """Test parse_rinex3_nav extracts Galileo records."""
        header = (
            "     3.05           N: GNSS NAV DATA    M: Mixed            RINEX VERSION / TYPE\n"
            "                                                            END OF HEADER\n"
        )
        content = header + "".join(self.GAL_RECORD)
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.rnx', delete=False)
        tmp.write(content)
        tmp.close()
        try:
            ds = convert_eph.parse_rinex3_nav(tmp.name, systems='E')
            assert ds is not None
            svs = [str(s) for s in ds.coords['sv'].values]
            assert 'E01' in svs
            assert 'IODnav' in ds.data_vars
            assert 'SISA' in ds.data_vars
            assert 'BGDe5b' in ds.data_vars
            val = float(ds['sqrtA'].sel(sv='E01').values[0])
            assert val == pytest.approx(5440.588268280)
        finally:
            os.unlink(tmp.name)


# ============================================================
# Galileo integration tests with real data
# ============================================================

class TestGalileoIntegration:
    """Integration tests with real RINEX 3 data containing Galileo."""

    IGS_MIXED = os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.igs_file = None
        if os.path.exists(self.IGS_MIXED):
            import gzip
            import shutil
            out = tmp_path / "BRDC00WRD_R_20260410000_01D_MN.rnx"
            with gzip.open(self.IGS_MIXED, 'rb') as f_in, open(out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            if out.exists():
                self.igs_file = str(out)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")),
        reason="IGS RINEX test file not available")
    def test_galileo_svs_loaded(self):
        """Test that Galileo SVs are loaded from mixed RINEX 3."""
        ds = convert_eph.load_rinex_nav(self.igs_file)
        assert ds is not None
        svs = [str(s) for s in ds.coords['sv'].values]
        e_count = sum(1 for s in svs if s.startswith('E'))
        assert e_count >= 20

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")),
        reason="IGS RINEX test file not available")
    def test_galileo_conversion(self):
        """Test Galileo ephemeris conversion produces msg_id=0x02."""
        ds = convert_eph.load_rinex_nav(self.igs_file)
        results = convert_eph.convert_rinex(ds, systems={'GAL'})
        assert len(results) >= 20

        for _, msg_bytes, _, _ in results:
            cls_id, msg_id, payload = parse_ubx_frame(msg_bytes)
            assert cls_id == 0x13
            assert msg_id == 0x02  # MGA-GAL
            assert len(payload) == 76

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_TESTDATA, "BRDC00WRD_R_20260410000_01D_MN.rnx.gz")),
        reason="IGS RINEX test file not available")
    def test_full_conversion_includes_galileo(self):
        """Test full conversion includes GPS, GLONASS, and Galileo."""
        ds = convert_eph.load_rinex_nav(self.igs_file)
        results = convert_eph.convert_rinex(ds)
        msg_ids = set()
        for _, msg_bytes, _, _ in results:
            _, msg_id, _ = parse_ubx_frame(msg_bytes)
            msg_ids.add(msg_id)
        assert 0x00 in msg_ids  # GPS
        assert 0x02 in msg_ids  # Galileo
        assert 0x06 in msg_ids  # GLONASS
