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
