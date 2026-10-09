"""Tests for RINEX header/record parsing, leap second fields and Galileo I/NAV selection."""
import convert_eph


def _write(tmp_path, lines):
    p = tmp_path / "nav.rnx"
    p.write_text("".join(l.ljust(60) + lbl + "\n" for l, lbl in lines))
    return str(p)


def test_rinex3_gal_line_with_three_coefficients(tmp_path):
    path = _write(tmp_path, [
        ("     3.04           N: GNSS NAV DATA    M: MIXED", "RINEX VERSION / TYPE"),
        ("GAL    8.4750D+01 -1.6016D-01  4.5166D-03", "IONOSPHERIC CORR"),
        ("GPSA   1.9558D-08  0.0000D+00 -1.1921D-07  0.0000D+00", "IONOSPHERIC CORR"),
        ("GPSB   1.2288D+05 -1.6384D+04 -2.6214D+05  1.9661D+05", "IONOSPHERIC CORR"),
        ("GPUT -4.6566128731D-09-1.154631946D-14  61440 2440", "TIME SYSTEM CORR"),
        ("    18    18  1929     7", "LEAP SECONDS"),
        ("", "END OF HEADER"),
    ])
    h = convert_eph.parse_rinex_header(path)
    assert h["GAL"] == [84.75, -0.16016, 0.0045166, 0.0]
    assert h["GPSA"][0] == 1.9558e-08
    assert h["GPUT"] == {"a0": -4.6566128731e-09, "a1": -1.154631946e-14, "tot": 61440, "wnt": 2440}
    assert h["LEAP"] == 18


def test_rinex2_gps_header(tmp_path):
    path = _write(tmp_path, [
        ("     2.11           N: GPS NAV DATA", "RINEX VERSION / TYPE"),
        ("    0.1956D-07  0.0000D+00 -0.1192D-06  0.0000D+00", "ION ALPHA"),
        ("    0.1229D+06 -0.1638D+05 -0.2621D+06  0.1966D+06", "ION BETA"),
        ("   -0.372529029846D-08-0.106581410364D-13    61440     2440", "DELTA-UTC: A0,A1,T,W"),
        ("    18", "LEAP SECONDS"),
        ("", "END OF HEADER"),
    ])
    h = convert_eph.parse_rinex_header(path)
    assert h["GPSA"] == [1.956e-08, 0.0, -1.192e-07, 0.0]
    assert h["GPSB"] == [122900.0, -16380.0, -262100.0, 196600.0]
    assert h["GPUT"] == {"a0": -3.72529029846e-09, "a1": -1.06581410364e-14, "tot": 61440, "wnt": 2440}
    assert h["LEAP"] == 18


def test_malformed_line_does_not_abort(tmp_path):
    path = _write(tmp_path, [
        ("GPSA   x.xxxxD-08", "IONOSPHERIC CORR"),
        ("GPUT -4.6566128731D-09-1.154631946D-14  61440 2440", "TIME SYSTEM CORR"),
        ("    18", "LEAP SECONDS"),
        ("", "END OF HEADER"),
    ])
    h = convert_eph.parse_rinex_header(path)
    assert "GPSA" not in h
    assert h["GPUT"]["wnt"] == 2440 and h["LEAP"] == 18


RINEX4 = """\
     4.01           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE
    18                                                      LEAP SECONDS
                                                            END OF HEADER
> STO G21 LNAV
    2026 10 10 19 50 24 GPUT                                  UTC(USNO)
     4.752840000000E+05-3.725290298462E-09-1.243449787580E-14 0.000000000000E+00
> STO G20 LNAV
    2026 10 11 17 04 00 GPUT                                  UTC(USNO)
    -1.265160000000E+05-4.656612873077E-09-1.154631945610E-14 0.000000000000E+00
> STO E19 IFNV
    2026 10 09 00 00 00 GAUT                                  UTCGAL
     4.320000000000E+05 9.313225746155E-10 0.000000000000E+00 0.000000000000E+00
> ION G21 LNAV
    2026 10 09 11 59 48 1.955777406693E-08 0.000000000000E+00-1.192092895508E-07
     0.000000000000E+00 1.228800000000E+05-1.638400000000E+04-2.621440000000E+05
     1.966080000000E+05 0.000000000000E+00
> ION E19 IFNV
    2026 10 09 11 59 57 8.475000000000E+01-1.601562500000E-01 4.516601562500E-03
     0.000000000000E+00
> ION E19 IFNV
    2026 10 09 12 31 06 8.950000000000E+01-2.187500000000E-01 4.425048828125E-03
     0.000000000000E+00
"""


def test_rinex4_sto_ion_records(tmp_path):
    p = tmp_path / "nav4.rnx"
    p.write_text(RINEX4)
    r = convert_eph.parse_rinex4_records(str(p))
    assert r["GPUT"] == {"a0": -4.656612873077e-09, "a1": -1.15463194561e-14, "tot": 61440, "wnt": 2440}
    assert r["GAUT"]["tot"] == 432000 and r["GAUT"]["wnt"] == 2439
    assert r["GPSA"] == [1.955777406693e-08, 0.0, -1.192092895508e-07, 0.0]
    assert r["GPSB"] == [122880.0, -16384.0, -262144.0, 196608.0]
    assert r["GAL"][0] == 89.5
    assert convert_eph.parse_rinex_header(str(p)) == {"LEAP": 18}


def test_rinex4_records_ignored_for_rinex3(tmp_path):
    p = tmp_path / "nav3.rnx"
    p.write_text(RINEX4.replace("     4.01", "     3.04", 1))
    assert not convert_eph.parse_rinex4_records(str(p))
