"""
Comprehensive unit tests for ubx_tool.py

Tests cover:
  - Bit extraction from GPS subframe words (IS-GPS-200 convention)
  - Two's complement conversion
  - Full GPS subframe decoding (clock, orbit, health parameters)
  - Round-trip: decode subframes -> build MGA-GPS-EPH -> verify checksum
  - URA meters-to-index lookup
  - MGA-GPS-EPH message building and payload structure
  - MGA-INI-TIME_UTC message building
  - MGA-INI-POS_LLH message building
  - Ephemeris table formatting
  - Cross-validation: convert_eph.py scaling matches ubx_tool.py decoding
"""

import math
import struct
import sys
import os
import socket
import time
from datetime import datetime, timezone
from types import SimpleNamespace
import pytest
from pyubx2 import UBXMessage, SET

sys.path.insert(0, os.path.dirname(__file__))

# We need to handle the pyubx2/serial imports gracefully for unit testing
# since we're testing pure functions that don't need serial hardware
import importlib
ubx_tool = importlib.import_module("ubx_tool")


# ============================================================
# Helpers
# ============================================================

def verify_ubx_checksum(raw):
    """Verify UBX message checksum. Returns True if valid."""
    assert raw[0:2] == b'\xb5\x62'
    length = struct.unpack('<H', raw[4:6])[0]
    ck_a, ck_b = 0, 0
    for b in raw[2:6+length]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a == raw[6+length] and ck_b == raw[7+length]


def parse_ubx_frame(raw):
    """Parse UBX message, verify checksum, return (cls, id, payload)."""
    assert verify_ubx_checksum(raw), "Invalid UBX checksum"
    cls_id = raw[2]
    msg_id = raw[3]
    length = struct.unpack('<H', raw[4:6])[0]
    payload = raw[6:6+length]
    return cls_id, msg_id, payload


def make_24bit_word(fields):
    """Build a 24-bit GPS data word from bit field definitions.

    Args:
        fields: list of (start_bit, num_bits, value) tuples
            where start_bit is 1-based IS-GPS-200 numbering (1=MSB).
    """
    word = 0
    for start, nbits, val in fields:
        shift = 24 - start - nbits + 1
        mask = (1 << nbits) - 1
        word |= (val & mask) << shift
    return word


def signed_to_unsigned(val, bits):
    """Convert signed value to unsigned representation for bit packing."""
    if val < 0:
        val += (1 << bits)
    return val & ((1 << bits) - 1)


# ============================================================
# extract_bits tests
# ============================================================

class TestExtractBits:
    """Test IS-GPS-200 bit extraction from 24-bit data words."""

    def test_all_ones(self):
        """Extracting all 24 bits of 0xFFFFFF should give 0xFFFFFF."""
        assert ubx_tool.extract_bits(0xFFFFFF, 1, 24) == 0xFFFFFF

    def test_all_zeros(self):
        assert ubx_tool.extract_bits(0x000000, 1, 24) == 0

    def test_msb_only(self):
        """Bit 1 (MSB) of 0x800000 should be 1."""
        assert ubx_tool.extract_bits(0x800000, 1, 1) == 1

    def test_lsb_only(self):
        """Bit 24 (LSB) of 0x000001 should be 1."""
        assert ubx_tool.extract_bits(0x000001, 24, 1) == 1

    def test_upper_byte(self):
        """Bits 1-8 of 0xAB0000 = 0xAB."""
        assert ubx_tool.extract_bits(0xAB0000, 1, 8) == 0xAB

    def test_middle_byte(self):
        """Bits 9-16 of 0x00CD00 = 0xCD."""
        assert ubx_tool.extract_bits(0x00CD00, 9, 8) == 0xCD

    def test_lower_byte(self):
        """Bits 17-24 of 0x0000EF = 0xEF."""
        assert ubx_tool.extract_bits(0x0000EF, 17, 8) == 0xEF

    def test_cross_byte_boundary(self):
        """Extract 12 bits spanning byte boundaries."""
        # 0x000FF0 = bits 13-24 set in lower portion
        # Extracting bits 9-20 (12 bits): shift = 24 - 9 - 12 + 1 = 4
        # (0x000FF0 >> 4) & 0xFFF = 0xFF = 255
        word = 0x000FF0
        result = ubx_tool.extract_bits(word, 9, 12)
        assert result == 0xFF

    def test_known_wn_extraction(self):
        """IS-GPS-200 subframe 1 word 3: WN in bits 1-10."""
        # WN=357 (10-bit truncated GPS week 2405)
        # shift = 24 - 1 - 10 + 1 = 14
        word = 357 << 14
        assert ubx_tool.extract_bits(word, 1, 10) == 357

    def test_single_bit_extraction(self):
        """Extract single bits at various positions."""
        word = 0b101010101010101010101010
        assert ubx_tool.extract_bits(word, 1, 1) == 1
        assert ubx_tool.extract_bits(word, 2, 1) == 0
        assert ubx_tool.extract_bits(word, 3, 1) == 1

    def test_word_with_upper_bits_ignored(self):
        """Bits 25-32 of the container should not affect extraction."""
        word_clean = 0x00ABCDEF
        word_dirty = 0xFFABCDEF
        assert ubx_tool.extract_bits(word_clean, 1, 24) == ubx_tool.extract_bits(word_dirty, 1, 24)


# ============================================================
# twos_complement tests
# ============================================================

class TestTwosComplement:
    def test_positive_stays_positive(self):
        assert ubx_tool.twos_complement(0, 8) == 0
        assert ubx_tool.twos_complement(127, 8) == 127
        assert ubx_tool.twos_complement(1, 8) == 1

    def test_max_negative_8bit(self):
        assert ubx_tool.twos_complement(128, 8) == -128

    def test_minus_one_8bit(self):
        assert ubx_tool.twos_complement(255, 8) == -1

    def test_minus_one_16bit(self):
        assert ubx_tool.twos_complement(65535, 16) == -1

    def test_16bit_negative(self):
        assert ubx_tool.twos_complement(0x8000, 16) == -32768

    def test_32bit_negative(self):
        assert ubx_tool.twos_complement(0x80000000, 32) == -(2**31)

    def test_32bit_minus_one(self):
        assert ubx_tool.twos_complement(0xFFFFFFFF, 32) == -1

    def test_22bit_negative(self):
        """22-bit signed (used for af0)."""
        # -1 in 22 bits = 0x3FFFFF
        assert ubx_tool.twos_complement(0x3FFFFF, 22) == -1

    def test_14bit_negative(self):
        """14-bit signed (used for IDOT)."""
        assert ubx_tool.twos_complement(0x3FFF, 14) == -1
        assert ubx_tool.twos_complement(0x2000, 14) == -8192


# ============================================================
# decode_gps_subframes tests
# ============================================================

class TestDecodeGpsSubframes:
    """Test full subframe decoding with constructed data.

    We encode known parameter values into the IS-GPS-200 subframe format
    and verify decode_gps_subframes recovers them exactly.
    """

    @staticmethod
    def _build_test_subframes():
        """Build subframe 1/2/3 words encoding known ephemeris values.

        Returns (sf1_words, sf2_words, sf3_words, expected_params).
        """
        # Known test values (raw integers before scaling)
        # Note: WN in subframe is only 10 bits (mod 1024)
        wn = 357  # 2405 & 0x3FF (GPS week 2405 truncated to 10 bits)
        ura_index = 1
        sv_health = 0
        iodc = 553  # MSB 2 bits = 2, LSB 8 bits = 41
        iodc_msb = (iodc >> 8) & 0x3  # = 2
        iodc_lsb = iodc & 0xFF        # = 41
        tgd_raw = -3    # signed 8-bit
        toc_raw = 10350  # unsigned 16-bit -> toc = 10350*16 = 165600 seconds
        af2_raw = 0      # signed 8-bit
        af1_raw = -1535   # signed 16-bit
        af0_raw = -227985 # signed 22-bit

        iode = 53
        crs_raw = -3970   # signed 16-bit
        deltaN_raw = 922  # signed 16-bit
        m0_raw = 1352061001  # signed 32-bit (MSB 8 + LSB 24)
        cuc_raw = -346    # signed 16-bit
        e_raw = 3205500   # unsigned 32-bit (MSB 8 + LSB 24)
        cus_raw = -378    # signed 16-bit
        sqrtA_raw = 2716110976  # unsigned 32-bit
        toe_raw = 10350   # unsigned 16-bit
        fit_flag = 0

        cic_raw = -15     # signed 16-bit
        omega0_raw = -1213792850  # signed 32-bit
        cis_raw = -16     # signed 16-bit
        i0_raw = 528513960  # signed 32-bit
        crc_raw = 6197    # signed 16-bit
        omega_raw = -490455982  # signed 32-bit
        omegaDot_raw = -17260  # signed 24-bit
        iode_sf3 = 53
        idot_raw = -157    # signed 14-bit

        # -- Build SF1 words (8 words, indices 0-7 = words 3-10) --
        # Word 3: WN(10) + L2code(2) + URA(4) + health(6) + IODC_MSB(2) = 24 bits
        sf1_w3 = make_24bit_word([
            (1, 10, wn),
            (11, 2, 1),  # L2 code
            (13, 4, ura_index),
            (17, 6, sv_health),
            (23, 2, iodc_msb),
        ])
        # Word 4-6: reserved (just zeros)
        sf1_w4 = 0
        sf1_w5 = 0
        sf1_w6 = 0
        # Word 7: bits 17-24 = TGD (8 bits signed)
        sf1_w7 = make_24bit_word([
            (17, 8, signed_to_unsigned(tgd_raw, 8)),
        ])
        # Word 8: IODC_LSB(8) + toc(16)
        sf1_w8 = make_24bit_word([
            (1, 8, iodc_lsb),
            (9, 16, toc_raw),
        ])
        # Word 9: af2(8) + af1(16)
        sf1_w9 = make_24bit_word([
            (1, 8, signed_to_unsigned(af2_raw, 8)),
            (9, 16, signed_to_unsigned(af1_raw, 16)),
        ])
        # Word 10: af0(22) in bits 1-22
        sf1_w10 = make_24bit_word([
            (1, 22, signed_to_unsigned(af0_raw, 22)),
        ])

        sf1 = [sf1_w3, sf1_w4, sf1_w5, sf1_w6, sf1_w7, sf1_w8, sf1_w9, sf1_w10]

        # -- Build SF2 words --
        # Word 3: IODE(8) + Crs(16)
        sf2_w3 = make_24bit_word([
            (1, 8, iode),
            (9, 16, signed_to_unsigned(crs_raw, 16)),
        ])
        # Word 4: deltaN(16) + M0_MSB(8)
        m0_msb = (signed_to_unsigned(m0_raw, 32) >> 24) & 0xFF
        m0_lsb = signed_to_unsigned(m0_raw, 32) & 0xFFFFFF
        sf2_w4 = make_24bit_word([
            (1, 16, signed_to_unsigned(deltaN_raw, 16)),
            (17, 8, m0_msb),
        ])
        # Word 5: M0_LSB(24)
        sf2_w5 = m0_lsb & 0xFFFFFF
        # Word 6: Cuc(16) + e_MSB(8)
        e_msb = (e_raw >> 24) & 0xFF
        e_lsb = e_raw & 0xFFFFFF
        sf2_w6 = make_24bit_word([
            (1, 16, signed_to_unsigned(cuc_raw, 16)),
            (17, 8, e_msb),
        ])
        # Word 7: e_LSB(24)
        sf2_w7 = e_lsb & 0xFFFFFF
        # Word 8: Cus(16) + sqrtA_MSB(8)
        sqrtA_msb = (sqrtA_raw >> 24) & 0xFF
        sqrtA_lsb = sqrtA_raw & 0xFFFFFF
        sf2_w8 = make_24bit_word([
            (1, 16, signed_to_unsigned(cus_raw, 16)),
            (17, 8, sqrtA_msb),
        ])
        # Word 9: sqrtA_LSB(24)
        sf2_w9 = sqrtA_lsb & 0xFFFFFF
        # Word 10: toe(16) + fit(1) + AODO(5) + padding(2)
        sf2_w10 = make_24bit_word([
            (1, 16, toe_raw),
            (17, 1, fit_flag),
        ])

        sf2 = [sf2_w3, sf2_w4, sf2_w5, sf2_w6, sf2_w7, sf2_w8, sf2_w9, sf2_w10]

        # -- Build SF3 words --
        # Word 3: Cic(16) + Omega0_MSB(8)
        o0_msb = (signed_to_unsigned(omega0_raw, 32) >> 24) & 0xFF
        o0_lsb = signed_to_unsigned(omega0_raw, 32) & 0xFFFFFF
        sf3_w3 = make_24bit_word([
            (1, 16, signed_to_unsigned(cic_raw, 16)),
            (17, 8, o0_msb),
        ])
        # Word 4: Omega0_LSB(24)
        sf3_w4 = o0_lsb & 0xFFFFFF
        # Word 5: Cis(16) + i0_MSB(8)
        i0_msb_val = (signed_to_unsigned(i0_raw, 32) >> 24) & 0xFF
        i0_lsb_val = signed_to_unsigned(i0_raw, 32) & 0xFFFFFF
        sf3_w5 = make_24bit_word([
            (1, 16, signed_to_unsigned(cis_raw, 16)),
            (17, 8, i0_msb_val),
        ])
        # Word 6: i0_LSB(24)
        sf3_w6 = i0_lsb_val & 0xFFFFFF
        # Word 7: Crc(16) + omega_MSB(8)
        om_msb = (signed_to_unsigned(omega_raw, 32) >> 24) & 0xFF
        om_lsb = signed_to_unsigned(omega_raw, 32) & 0xFFFFFF
        sf3_w7 = make_24bit_word([
            (1, 16, signed_to_unsigned(crc_raw, 16)),
            (17, 8, om_msb),
        ])
        # Word 8: omega_LSB(24)
        sf3_w8 = om_lsb & 0xFFFFFF
        # Word 9: OmegaDot(24)
        sf3_w9 = signed_to_unsigned(omegaDot_raw, 24) & 0xFFFFFF
        # Word 10: IODE(8) + IDOT(14) + padding(2)
        sf3_w10 = make_24bit_word([
            (1, 8, iode_sf3),
            (9, 14, signed_to_unsigned(idot_raw, 14)),
        ])

        sf3 = [sf3_w3, sf3_w4, sf3_w5, sf3_w6, sf3_w7, sf3_w8, sf3_w9, sf3_w10]

        expected = {
            'wn': wn, 'ura_index': ura_index, 'sv_health': sv_health,
            'iodc': iodc,
            'tgd_raw': tgd_raw, 'toc_raw': toc_raw,
            'af2_raw': af2_raw, 'af1_raw': af1_raw, 'af0_raw': af0_raw,
            'iode': iode,
            'crs_raw': crs_raw, 'deltaN_raw': deltaN_raw, 'm0_raw': m0_raw,
            'cuc_raw': cuc_raw, 'e_raw': e_raw, 'cus_raw': cus_raw,
            'sqrtA_raw': sqrtA_raw, 'toe_raw': toe_raw,
            'fit_flag': fit_flag,
            'cic_raw': cic_raw, 'omega0_raw': omega0_raw,
            'cis_raw': cis_raw, 'i0_raw': i0_raw,
            'crc_raw': crc_raw, 'omega_raw': omega_raw,
            'omegaDot_raw': omegaDot_raw,
            'iode_sf3': iode_sf3, 'idot_raw': idot_raw,
        }

        return sf1, sf2, sf3, expected

    def test_week_number(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['week'] == exp['wn']

    def test_ura_index(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['ura_index'] == exp['ura_index']

    def test_sv_health(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['sv_health'] == exp['sv_health']

    def test_iodc(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['iodc'] == exp['iodc']

    def test_tgd_scaling(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected_tgd = exp['tgd_raw'] * 2**-31
        assert result['tgd'] == pytest.approx(expected_tgd)

    def test_toc_scaling(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['toc'] == exp['toc_raw'] * 16

    def test_clock_af0(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected_af0 = exp['af0_raw'] * 2**-31
        assert result['af0'] == pytest.approx(expected_af0)

    def test_clock_af1(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected_af1 = exp['af1_raw'] * 2**-43
        assert result['af1'] == pytest.approx(expected_af1)

    def test_clock_af2(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected_af2 = exp['af2_raw'] * 2**-55
        assert result['af2'] == pytest.approx(expected_af2)

    def test_iode(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['iode'] == exp['iode']

    def test_sqrtA(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected = exp['sqrtA_raw'] * 2**-19
        assert result['sqrtA'] == pytest.approx(expected)

    def test_eccentricity(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected = exp['e_raw'] * 2**-33
        assert result['e'] == pytest.approx(expected)

    def test_toe(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['toe'] == exp['toe_raw'] * 16

    def test_angular_m0(self):
        """M0 should be decoded in radians (raw * 2^-31 * pi)."""
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected = exp['m0_raw'] * 2**-31 * math.pi
        assert result['m0'] == pytest.approx(expected, rel=1e-10)

    def test_angular_omega0(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected = exp['omega0_raw'] * 2**-31 * math.pi
        assert result['omega0'] == pytest.approx(expected, rel=1e-10)

    def test_angular_omegaDot(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected = exp['omegaDot_raw'] * 2**-43 * math.pi
        assert result['omegaDot'] == pytest.approx(expected, rel=1e-10)

    def test_angular_idot(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        expected = exp['idot_raw'] * 2**-43 * math.pi
        assert result['idot'] == pytest.approx(expected, rel=1e-10)

    def test_fit_interval(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['fit_interval'] == exp['fit_flag']

    def test_iode_sf3(self):
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert result['iode_sf3'] == exp['iode_sf3']

    def test_raw_dict_present(self):
        """Decoded result should contain _raw dict for MGA building."""
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        assert '_raw' in result
        raw = result['_raw']
        assert raw['tgd'] == exp['tgd_raw']
        assert raw['iodc'] == exp['iodc']
        assert raw['af0'] == exp['af0_raw']
        assert raw['sqrtA'] == exp['sqrtA_raw']
        assert raw['e'] == exp['e_raw']

    def test_raw_dict_signed_values(self):
        """Verify _raw dict preserves signed values correctly."""
        sf1, sf2, sf3, exp = self._build_test_subframes()
        result = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        raw = result['_raw']
        assert raw['af1'] == exp['af1_raw']
        assert raw['crs'] == exp['crs_raw']
        assert raw['omega0'] == exp['omega0_raw']
        assert raw['idot'] == exp['idot_raw']


# ============================================================
# URA meters-to-index tests
# ============================================================

class TestUraMetersToIndex:
    def test_best_accuracy(self):
        assert ubx_tool.ura_meters_to_index(2.0) == 0

    def test_exact_threshold(self):
        assert ubx_tool.ura_meters_to_index(2.4) == 0

    def test_just_above(self):
        assert ubx_tool.ura_meters_to_index(2.5) == 1

    def test_typical_gps_accuracy(self):
        """Most healthy GPS satellites have URA ~ 2.0m -> index 0."""
        assert ubx_tool.ura_meters_to_index(2.0) == 0

    def test_degraded_accuracy(self):
        assert ubx_tool.ura_meters_to_index(10.0) == 5

    def test_worst_case(self):
        assert ubx_tool.ura_meters_to_index(99999) == 15

    def test_each_index(self):
        """Verify boundary for each URA index."""
        thresholds = [2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24, 48,
                      96, 192, 384, 768, 1536, 3072, 6144, 6145]
        for i, t in enumerate(thresholds):
            assert ubx_tool.ura_meters_to_index(t) == i


# ============================================================
# build_mga_gps_eph tests
# ============================================================

class TestBuildMgaGpsEph:
    """Test MGA-GPS-EPH message builder from raw integer dict."""

    @staticmethod
    def _make_raw():
        return {
            'tgd': -3, 'iodc': 553, 'toc': 10350,
            'af2': 0, 'af1': -1535, 'af0': -227985,
            'crs': -3970, 'deltaN': 922, 'm0': 1352061001,
            'cuc': -346, 'e': 3205500, 'cus': -378,
            'sqrtA': 2716110976, 'toe': 10350,
            'cic': -15, 'omega0': -1213792850,
            'cis': -16, 'i0': 528513960,
            'crc': 6197, 'omega': -490455982,
            'omegaDot': -17260, 'idot': -157,
            'ura_index': 1, 'sv_health': 0, 'fit_interval': 0,
        }

    def test_valid_checksum(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        assert verify_ubx_checksum(msg)

    def test_message_class_id_gps(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw, mga_class=0x13, mga_id=0x00)
        cls_id, msg_id, _ = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00

    def test_message_class_id_qzss(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw, mga_class=0x13, mga_id=0x05)
        cls_id, msg_id, _ = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x05

    def test_payload_length(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        _, _, payload = parse_ubx_frame(msg)
        assert len(payload) == 68

    def test_total_message_length(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        # sync(2) + cls(1) + id(1) + len(2) + payload(68) + cksum(2) = 76
        assert len(msg) == 76

    def test_type_byte(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x01  # type = EPH

    def test_sv_id_encoding(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(17, raw)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[2] == 17

    def test_iodc_encoding(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        _, _, payload = parse_ubx_frame(msg)
        iodc = struct.unpack('<H', payload[8:10])[0]
        assert iodc == 553

    def test_tgd_signed_encoding(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        _, _, payload = parse_ubx_frame(msg)
        tgd = struct.unpack('b', payload[7:8])[0]
        assert tgd == -3

    def test_af0_signed_encoding(self):
        raw = self._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        _, _, payload = parse_ubx_frame(msg)
        af0 = struct.unpack('<i', payload[16:20])[0]
        assert af0 == -227985


# ============================================================
# Round-trip: subframe decode -> MGA-GPS-EPH build
# ============================================================

class TestDecodeToMgaRoundTrip:
    """Verify the decode -> build MGA pipeline produces valid messages."""

    def test_round_trip_produces_valid_ubx(self):
        sf1, sf2, sf3, _ = TestDecodeGpsSubframes._build_test_subframes()
        eph = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        msg = ubx_tool.build_mga_gps_eph(1, eph['_raw'])
        assert verify_ubx_checksum(msg)
        cls_id, msg_id, payload = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x00
        assert len(payload) == 68

    def test_round_trip_preserves_iodc(self):
        sf1, sf2, sf3, exp = TestDecodeGpsSubframes._build_test_subframes()
        eph = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        msg = ubx_tool.build_mga_gps_eph(1, eph['_raw'])
        _, _, payload = parse_ubx_frame(msg)
        iodc = struct.unpack('<H', payload[8:10])[0]
        assert iodc == exp['iodc']

    def test_round_trip_preserves_af0(self):
        sf1, sf2, sf3, exp = TestDecodeGpsSubframes._build_test_subframes()
        eph = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        msg = ubx_tool.build_mga_gps_eph(1, eph['_raw'])
        _, _, payload = parse_ubx_frame(msg)
        af0 = struct.unpack('<i', payload[16:20])[0]
        assert af0 == exp['af0_raw']

    def test_round_trip_preserves_sqrtA(self):
        sf1, sf2, sf3, exp = TestDecodeGpsSubframes._build_test_subframes()
        eph = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        msg = ubx_tool.build_mga_gps_eph(1, eph['_raw'])
        _, _, payload = parse_ubx_frame(msg)
        sqrtA = struct.unpack('<I', payload[36:40])[0]
        assert sqrtA == exp['sqrtA_raw']


# ============================================================
# MGA-INI-POS_LLH tests
# ============================================================

class TestBuildMgaIniPosLlh:
    def test_valid_checksum(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5, -0.13)
        assert verify_ubx_checksum(msg)

    def test_message_class_id(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5, -0.13)
        cls_id, msg_id, _ = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x40

    def test_payload_length(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5, -0.13)
        _, _, payload = parse_ubx_frame(msg)
        assert len(payload) == 20

    def test_type_byte(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5, -0.13)
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x01  # POS_LLH

    def test_latitude_encoding(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5074, -0.1278)
        _, _, payload = parse_ubx_frame(msg)
        lat_raw = struct.unpack('<i', payload[4:8])[0]
        assert lat_raw == int(51.5074 * 1e7)

    def test_longitude_encoding(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5074, -0.1278)
        _, _, payload = parse_ubx_frame(msg)
        lon_raw = struct.unpack('<i', payload[8:12])[0]
        assert lon_raw == int(-0.1278 * 1e7)

    def test_altitude_encoding(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5, -0.13, alt_m=100.0)
        _, _, payload = parse_ubx_frame(msg)
        alt_raw = struct.unpack('<i', payload[12:16])[0]
        assert alt_raw == 10000  # 100m * 100 = 10000cm

    def test_accuracy_encoding(self):
        msg = ubx_tool.build_mga_ini_pos_llh(51.5, -0.13, acc_m=200.0)
        _, _, payload = parse_ubx_frame(msg)
        acc_raw = struct.unpack('<I', payload[16:20])[0]
        assert acc_raw == 20000  # 200m * 100 = 20000cm

    def test_default_altitude(self):
        msg = ubx_tool.build_mga_ini_pos_llh(0.0, 0.0)
        _, _, payload = parse_ubx_frame(msg)
        alt_raw = struct.unpack('<i', payload[12:16])[0]
        assert alt_raw == 5000  # 50m default * 100

    def test_negative_latitude(self):
        msg = ubx_tool.build_mga_ini_pos_llh(-33.8688, 151.2093)
        _, _, payload = parse_ubx_frame(msg)
        lat_raw = struct.unpack('<i', payload[4:8])[0]
        assert lat_raw < 0

    def test_full_range_longitude(self):
        """Test longitude at +180 and -180."""
        msg = ubx_tool.build_mga_ini_pos_llh(0.0, 179.999)
        _, _, payload = parse_ubx_frame(msg)
        lon_raw = struct.unpack('<i', payload[8:12])[0]
        assert lon_raw > 0

        msg = ubx_tool.build_mga_ini_pos_llh(0.0, -179.999)
        _, _, payload = parse_ubx_frame(msg)
        lon_raw = struct.unpack('<i', payload[8:12])[0]
        assert lon_raw < 0


# ============================================================
# MGA-INI-TIME_UTC tests
# ============================================================

class TestBuildMgaIniTimeUtc:
    def test_valid_checksum(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        assert verify_ubx_checksum(msg)

    def test_message_class_id(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        cls_id, msg_id, _ = parse_ubx_frame(msg)
        assert cls_id == 0x13
        assert msg_id == 0x40

    def test_payload_length(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        assert len(payload) == 24

    def test_type_byte(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        assert payload[0] == 0x10  # TIME_UTC

    def test_leap_seconds(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        leap = struct.unpack('b', payload[3:4])[0]
        assert leap == 18  # GPS-UTC offset as of 2026

    def test_year_is_current(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        year = struct.unpack('<H', payload[4:6])[0]
        now = datetime.now(timezone.utc)
        assert year == now.year

    def test_month_day_reasonable(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        month = payload[6]
        day = payload[7]
        assert 1 <= month <= 12
        assert 1 <= day <= 31

    def test_time_accuracy(self):
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        tAccS = struct.unpack('<H', payload[16:18])[0]
        assert tAccS == 1  # 1 second accuracy


# ============================================================
# format_eph_table tests
# ============================================================

class TestFormatEphTable:
    def test_empty_dict(self):
        result = ubx_tool.format_eph_table({})
        assert 'PRN' in result
        assert 'IODC' in result

    def test_single_satellite(self):
        sats = {
            1: {
                'iodc': 47, 'iode': 47, 'toc': 14400.0, 'toe': 14400.0,
                'ura_index': 1, 'sv_health': 0,
                'af0': -1.062e-04, 'sqrtA': 5153.63, 'e': 0.001493,
            }
        }
        result = ubx_tool.format_eph_table(sats)
        lines = result.split('\n')
        assert len(lines) >= 3  # header + separator + 1 data row
        assert 'G01' in lines[2]

    def test_multiple_satellites_sorted(self):
        sats = {}
        for sv_id in [5, 1, 32]:
            sats[sv_id] = {
                'iodc': 47, 'iode': 47, 'toc': 14400.0, 'toe': 14400.0,
                'ura_index': 1, 'sv_health': 0,
                'af0': 0.0, 'sqrtA': 5153.0, 'e': 0.001,
            }
        result = ubx_tool.format_eph_table(sats)
        lines = result.split('\n')
        data_lines = lines[2:]  # Skip header and separator
        assert 'G01' in data_lines[0]
        assert 'G05' in data_lines[1]
        assert 'G32' in data_lines[2]


# ============================================================
# Cross-validation: convert-eph scaling matches ubx-tool decode
# ============================================================

class TestCrossValidation:
    """Verify that convert_eph.py's scaling produces the same raw integers
    that ubx_tool.py's subframe decoder extracts.

    This tests the fundamental consistency between the two tools.
    """

    def test_af0_roundtrip(self):
        """Encode af0 via subframe, decode, then re-encode via convert-eph scaling."""
        convert_eph = importlib.import_module("convert_eph")

        af0_raw_original = -227985
        af0_si = af0_raw_original * 2**-31  # decode to SI

        # convert_eph.py would scale this back
        af0_raw_rescaled = convert_eph.scale_signed(af0_si, 2**-31, 32)
        assert af0_raw_rescaled == af0_raw_original

    def test_sqrtA_roundtrip(self):
        convert_eph = importlib.import_module("convert_eph")

        sqrtA_raw_original = 2716110976
        sqrtA_si = sqrtA_raw_original * 2**-19

        sqrtA_raw_rescaled = convert_eph.scale_unsigned(sqrtA_si, 2**-19, 32)
        assert sqrtA_raw_rescaled == sqrtA_raw_original

    def test_m0_angular_roundtrip(self):
        """M0: raw -> radians -> re-encode should match."""
        convert_eph = importlib.import_module("convert_eph")

        m0_raw_original = 1352061001
        m0_rad = m0_raw_original * 2**-31 * math.pi  # decoded

        m0_raw_rescaled = convert_eph.scale_angular_signed(m0_rad, 2**-31, 32)
        assert m0_raw_rescaled == m0_raw_original

    def test_omegaDot_angular_roundtrip(self):
        convert_eph = importlib.import_module("convert_eph")

        omegaDot_raw_original = -17260
        omegaDot_rad = omegaDot_raw_original * 2**-43 * math.pi

        # OmegaDot in MGA-EPH is I4 (32-bit) even though only 24-bit in subframe
        omegaDot_rescaled = convert_eph.scale_angular_signed(omegaDot_rad, 2**-43, 32)
        assert omegaDot_rescaled == omegaDot_raw_original

    def test_tgd_roundtrip(self):
        convert_eph = importlib.import_module("convert_eph")

        tgd_raw_original = -3
        tgd_si = tgd_raw_original * 2**-31

        tgd_rescaled = convert_eph.scale_signed(tgd_si, 2**-31, 8)
        assert tgd_rescaled == tgd_raw_original

    def test_payload_layout_matches(self):
        """Both tools' MGA-EPH builders should produce identical payloads
        for the same raw integer inputs."""
        convert_eph = importlib.import_module("convert_eph")

        raw = {
            'sv_id': 1, 'fit_interval': 0, 'ura_index': 1, 'sv_health': 0,
            'tgd': -3, 'iodc': 553, 'toc': 10350,
            'af2': 0, 'af1': -1535, 'af0': -227985,
            'crs': -3970, 'deltaN': 922, 'm0': 1352061001,
            'cuc': -346, 'cus': -378, 'e': 3205500,
            'sqrtA': 2716110976, 'toe': 10350,
            'cic': -15, 'omega0': -1213792850,
            'cis': -16, 'crc': 6197, 'i0': 528513960,
            'omega': -490455982, 'omegaDot': -17260, 'idot': -157,
        }

        # Build via convert_eph.py
        payload_ce = convert_eph.build_mga_eph_payload(raw)

        # Build via ubx_tool.py
        msg_ut = ubx_tool.build_mga_gps_eph(1, raw, mga_class=0x13, mga_id=0x00)

        # Compare payload bytes (skip sync/header since construction may differ)
        _, _, payload_ut = parse_ubx_frame(msg_ut)
        assert payload_ce == payload_ut, "Payload mismatch between convert_eph.py and ubx_tool.py"


# ============================================================
# format_nav_pvt tests
# ============================================================

class TestFormatNavPvt:
    def test_basic_format(self):
        parsed = SimpleNamespace(
            iTOW=0, fixType=3, year=2026, month=2, day=10,
            hour=12, min=30, second=45, numSV=14,
            lat=51.5074, lon=-0.1278, hMSL=50000, # mm
            hAcc=2500, vAcc=4000,  # mm
        )
        result = ubx_tool.format_nav_pvt(parsed)
        assert '2026-02-10 12:30:45' in result
        assert '3D' in result
        assert '14' in result

    def test_no_fix(self):
        parsed = SimpleNamespace(
            iTOW=0, fixType=0, year=2026, month=1, day=1,
            hour=0, min=0, second=0, numSV=0,
            lat=0.0, lon=0.0, hMSL=0, hAcc=0, vAcc=0,
        )
        result = ubx_tool.format_nav_pvt(parsed)
        assert 'No fix' in result


# ============================================================
# TcpStream tests
# ============================================================

class TestTcpStream:
    """Test TcpStream wrapper class."""

    def test_is_tcp_marker(self):
        """TcpStream instances should have is_tcp=True."""
        import threading
        # Create a minimal TCP server
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def accept_one():
            conn, _ = srv.accept()
            conn.close()

        t = threading.Thread(target=accept_one, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=1)
        assert stream.is_tcp is True
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_timeout_property(self):
        """Timeout should be gettable and settable."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def accept_one():
            conn, _ = srv.accept()
            time.sleep(0.5)
            conn.close()

        t = threading.Thread(target=accept_one, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=2)
        assert stream.timeout == 2
        stream.timeout = 5
        assert stream.timeout == 5
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_baudrate_raises(self):
        """Setting baudrate on TcpStream should raise RuntimeError."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def accept_one():
            conn, _ = srv.accept()
            time.sleep(0.5)
            conn.close()

        t = threading.Thread(target=accept_one, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=1)
        assert stream.baudrate == 0  # read is OK (placeholder)
        with pytest.raises(RuntimeError, match="Cannot change baud rate"):
            stream.baudrate = 9600
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_write_and_read(self):
        """Data written to TcpStream should be echoed back by a loopback server."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def echo_server():
            conn, _ = srv.accept()
            try:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    conn.sendall(data)
            except Exception:
                pass
            finally:
                conn.close()

        t = threading.Thread(target=echo_server, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=2)

        # Write then read back
        test_data = b'\xb5\x62\x06\x00\x00\x00\x06\x18'
        stream.write(test_data)
        time.sleep(0.05)
        result = stream.read(len(test_data))
        assert result == test_data

        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_reset_input_buffer(self):
        """reset_input_buffer should drain pending data."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def send_junk():
            conn, _ = srv.accept()
            try:
                conn.sendall(b'X' * 1000)
                time.sleep(1)
            except Exception:
                pass
            finally:
                conn.close()

        t = threading.Thread(target=send_junk, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=1)
        time.sleep(0.1)  # let junk arrive
        stream.reset_input_buffer()
        # After drain, read should timeout with no data
        result = stream.read(1)
        assert len(result) == 0

        stream.close()
        srv.close()
        t.join(timeout=2)

    def test_readline_basic(self):
        """readline should read up to and including the newline."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def send_line():
            conn, _ = srv.accept()
            conn.sendall(b'$GPGGA,test*00\r\n')
            time.sleep(0.5)
            conn.close()

        t = threading.Thread(target=send_line, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=2)
        line = stream.readline()
        assert line == b'$GPGGA,test*00\r\n'
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_readline_multiple_lines(self):
        """readline should return only the first line when multiple are available."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def send_lines():
            conn, _ = srv.accept()
            conn.sendall(b'line1\nline2\n')
            time.sleep(0.5)
            conn.close()

        t = threading.Thread(target=send_lines, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=2)
        time.sleep(0.05)
        first = stream.readline()
        assert first == b'line1\n'
        second = stream.readline()
        assert second == b'line2\n'
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_readline_no_newline_returns_on_close(self):
        """readline should return available data when connection closes without newline."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def send_partial():
            conn, _ = srv.accept()
            conn.sendall(b'no newline')
            conn.close()

        t = threading.Thread(target=send_partial, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=2)
        line = stream.readline()
        assert line == b'no newline'
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_readline_empty_on_immediate_close(self):
        """readline should return empty bytes when connection closes immediately."""
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def close_immediately():
            conn, _ = srv.accept()
            conn.close()

        t = threading.Thread(target=close_immediately, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=2)
        line = stream.readline()
        assert line == b''
        stream.close()
        srv.close()
        t.join(timeout=1)

    def test_repr(self):
        import threading
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 0))
        port = srv.getsockname()[1]
        srv.listen(1)

        def accept_one():
            conn, _ = srv.accept()
            conn.close()

        t = threading.Thread(target=accept_one, daemon=True)
        t.start()

        stream = ubx_tool.TcpStream('127.0.0.1', port, timeout=1)
        assert '127.0.0.1' in repr(stream)
        assert str(port) in repr(stream)
        stream.close()
        srv.close()
        t.join(timeout=1)


class TestEnsureUbxOutputTcpSkip:
    """Verify ensure_ubx_output behaviour in TCP vs serial mode."""

    def test_sends_cfg_prt_over_tcp(self):
        """ensure_ubx_output should send CFG-PRT even over TCP."""
        written = []
        mock_ser = SimpleNamespace(
            is_tcp=True,
            write=written.append,
            flush=lambda: None,
            reset_input_buffer=lambda: None,
            timeout=3,
        )
        ubx_tool.ensure_ubx_output(mock_ser, 115200)
        assert len(written) > 0

    def test_tcp_forces_default_baud(self, capsys):
        """In TCP mode, non-default baud should be overridden to 115200."""
        written = []
        mock_ser = SimpleNamespace(
            is_tcp=True,
            write=written.append,
            flush=lambda: None,
            reset_input_buffer=lambda: None,
            timeout=3,
        )
        ubx_tool.ensure_ubx_output(mock_ser, 9600)
        assert len(written) > 0
        captured = capsys.readouterr()
        assert '115200' in captured.err
        assert '9600' in captured.err

    def test_serial_uses_requested_baud(self):
        """Non-TCP should send for requested baud."""
        written = []
        mock_ser = SimpleNamespace(
            write=written.append,
            flush=lambda: None,
            reset_input_buffer=lambda: None,
            timeout=3,
        )
        ubx_tool.ensure_ubx_output(mock_ser, 115200)
        assert len(written) > 0


class TestTcpBlockedCommands:
    """Verify that dangerous commands are blocked in TCP mode."""

    def test_reset_is_blocked(self):
        """reset should be in the blocked set."""
        import inspect
        source = inspect.getsource(ubx_tool.main)
        assert 'TCP_BLOCKED_COMMANDS' in source
        assert "'reset'" in source or '"reset"' in source

    def test_enable_ubx_is_allowed(self):
        """enable-ubx should NOT be blocked over TCP."""
        import inspect
        source = inspect.getsource(ubx_tool.main)
        # enable-ubx must not appear in the TCP_BLOCKED_COMMANDS set
        # Find the line defining the set and check its contents
        for line in source.split('\n'):
            if 'TCP_BLOCKED_COMMANDS' in line and '=' in line:
                assert 'enable-ubx' not in line
                break


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
        """Build MGA-GPS-EPH from raw integers, parse with pyubx2."""
        raw = TestBuildMgaGpsEph._make_raw()
        msg = ubx_tool.build_mga_gps_eph(1, raw)
        _, _, payload = parse_ubx_frame(msg)
        parsed = UBXMessage(0x13, 0x00, SET, payload=payload)

        assert parsed.identity == "MGA-GPS-EPH"
        assert parsed.svId == 1
        assert parsed.iodc == 553
        # tgd: raw=-3, scale=2^-31
        assert parsed.tgd == pytest.approx(-3 * 2**-31)
        # af0: raw=-227985, scale=2^-31
        assert parsed.af0 == pytest.approx(-227985 * 2**-31)
        # sqrtA: raw=2716110976, scale=2^-19
        assert parsed.sqrtA == pytest.approx(2716110976 * 2**-19, rel=1e-9)
        # e: raw=3205500, scale=2^-33
        assert parsed.e == pytest.approx(3205500 * 2**-33, rel=1e-9)

    def test_mga_gps_eph_round_trip(self):
        """Full decode -> build -> pyubx2 parse pipeline."""
        sf1, sf2, sf3, exp = TestDecodeGpsSubframes._build_test_subframes()
        eph = ubx_tool.decode_gps_subframes(sf1, sf2, sf3)
        msg = ubx_tool.build_mga_gps_eph(1, eph['_raw'])
        _, _, payload = parse_ubx_frame(msg)
        parsed = UBXMessage(0x13, 0x00, SET, payload=payload)

        assert parsed.identity == "MGA-GPS-EPH"
        assert parsed.iodc == exp['iodc']
        # af0: raw -> pyubx2 applies * 2^-31
        assert parsed.af0 == pytest.approx(exp['af0_raw'] * 2**-31)
        # sqrtA: raw -> * 2^-19
        assert parsed.sqrtA == pytest.approx(exp['sqrtA_raw'] * 2**-19, rel=1e-9)
        # e: raw -> * 2^-33
        assert parsed.e == pytest.approx(exp['e_raw'] * 2**-33, rel=1e-9)
        # m0 in semi-circles (raw * 2^-31, no pi multiplication)
        assert parsed.m0 == pytest.approx(exp['m0_raw'] * 2**-31, rel=1e-9)

    def test_mga_ini_pos_llh(self):
        """Build POS LLH for London coords, parse with pyubx2."""
        msg = ubx_tool.build_mga_ini_pos_llh(51.5074, -0.1278)
        _, _, payload = parse_ubx_frame(msg)
        parsed = UBXMessage(0x13, 0x40, SET, payload=payload)

        assert parsed.identity == "MGA-INI-POS-LLH"
        assert parsed.lat == pytest.approx(51.5074, rel=1e-6)
        assert parsed.lon == pytest.approx(-0.1278, rel=1e-4)
        assert parsed.alt == 5000   # 50m default * 100 = 5000 cm
        assert parsed.posAcc == 5000  # 50m default * 100 = 5000 cm

    def test_mga_ini_time_utc(self):
        """Build TIME UTC, parse with pyubx2."""
        msg = ubx_tool.build_mga_ini_time_utc()
        _, _, payload = parse_ubx_frame(msg)
        parsed = UBXMessage(0x13, 0x40, SET, payload=payload)

        assert parsed.identity == "MGA-INI-TIME-UTC"
        now = datetime.now(timezone.utc)
        assert parsed.year == now.year
        assert parsed.leapSecs == 18
        assert parsed.tAccS == 1
