"""Tests for RINEX header/record parsing, leap second fields and Galileo I/NAV selection."""
import math

import pytest

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


def test_leap_seconds_future_fields(tmp_path):
    path = _write(tmp_path, [
        ("     3.04           N: GNSS NAV DATA    M: MIXED", "RINEX VERSION / TYPE"),
        ("    18    18  1929     7", "LEAP SECONDS"),
        ("", "END OF HEADER"),
    ])
    h = convert_eph.parse_rinex_header(path)
    assert h["LEAP"] == 18
    assert h["LEAP_FUTURE"] == {"dtlsf": 18, "wnlsf": 1929, "dn": 7}


def test_leap_seconds_current_only_and_bds(tmp_path):
    p1 = _write(tmp_path, [("    18", "LEAP SECONDS"), ("", "END OF HEADER")])
    assert convert_eph.parse_rinex_header(p1) == {"LEAP": 18}
    p2 = _write(tmp_path, [("     4     4  1929     7BDS", "LEAP SECONDS"), ("", "END OF HEADER")])
    assert "LEAP_FUTURE" not in convert_eph.parse_rinex_header(p2)


def test_gps_and_gal_utc_leap_event_fields():
    utc = {"a0": 0.0, "a1": 0.0, "tot": 61440, "wnt": 2440}
    fut = {"dtlsf": 19, "wnlsf": 2450, "dn": 3}
    for build in (convert_eph.build_mga_gps_utc, convert_eph.build_mga_gal_utc):
        p = build(utc, 18, fut)[6:-2]
        assert p[12] == 18
        assert p[15] == 2450 & 0xFF and p[16] == 3 and p[17] == 19
        p = build(utc, 18)[6:-2]
        assert p[15] == 0 and p[16] == 0 and p[17] == 18


def test_gal_utc_tot_scale():
    p = convert_eph.build_mga_gal_utc({"a0": 0.0, "a1": 0.0, "tot": 432000, "wnt": 2439}, 18)[6:-2]
    assert p[13] == 120
    p = convert_eph.build_mga_gps_utc({"a0": 0.0, "a1": 0.0, "tot": 61440, "wnt": 2440}, 18)[6:-2]
    assert p[13] == 15


GAL_FNAV = """E36 2026 10 09 11 10 00-6.332660559565D-04-5.172751116334D-12 0.000000000000D+00
     2.500000000000D+01-1.524062500000D+02 2.391171030443D-09 3.112872742410D+00
    -7.016584277153D-06 2.225677017123D-04 1.202337443829D-05 5.440608079910D+03
     4.722000000000D+05 2.421438694000D-08-5.917720083995D-01-2.607703208923D-08
     9.971050159405D-01 9.618750000000D+01-1.374700419368D+00-5.286291623836D-09
    -3.860875106660D-10 2.580000000000D+02 2.439000000000D+03 0.000000000000D+00
     3.120000000000D+00 0.000000000000D+00 3.492459654808D-09 0.000000000000D+00
     4.729500000000D+05
"""
GAL_INAV = """E36 2026 10 09 11 10 00-6.332648918033D-04-5.186961971049D-12 0.000000000000D+00
     2.500000000000D+01-1.524062500000D+02 2.391171030443D-09 3.112872742410D+00
    -7.016584277153D-06 2.225677017123D-04 1.202337443829D-05 5.440608079910D+03
     4.722000000000D+05 2.421438694000D-08-5.917720083995D-01-2.607703208923D-08
     9.971050159405D-01 9.618750000000D+01-1.374700419368D+00-5.286291623836D-09
    -3.860875106660D-10 5.130000000000D+02 2.439000000000D+03 0.000000000000D+00
     3.120000000000D+00 0.000000000000D+00 3.492459654808D-09 3.958120942116D-09
     4.728640000000D+05
"""
HDR3 = "     3.04           N: GNSS NAV DATA    E: GALILEO          RINEX VERSION / TYPE\n" \
       "                                                            END OF HEADER\n"
HDR4 = "     4.01           N: GNSS NAV DATA    E: GALILEO          RINEX VERSION / TYPE\n" \
       "                                                            END OF HEADER\n"
INAV_AF0 = -6.332648918033e-04


def _gal_af0(path):
    nav = convert_eph.load_rinex_nav(path)
    d = nav.sel(sv="E36").dropna(dim="time", how="all")
    return float(d["SVclockBias"].values[0]), int(d["DataSrc"].values[0])


@pytest.mark.parametrize("order", ["fnav_first", "inav_first"])
def test_galileo_prefers_inav_rinex3(tmp_path, order):
    body = GAL_FNAV + GAL_INAV if order == "fnav_first" else GAL_INAV + GAL_FNAV
    p = tmp_path / "gal3.rnx"
    p.write_text(HDR3 + body)
    af0, dsrc = _gal_af0(str(p))
    assert dsrc % 2 == 1 and af0 == INAV_AF0


@pytest.mark.parametrize("order", ["fnav_first", "inav_first"])
def test_galileo_prefers_inav_rinex4(tmp_path, order):
    f = "> EPH E36 FNAV\n" + GAL_FNAV
    i = "> EPH E36 INAV\n" + GAL_INAV
    p = tmp_path / "gal4.rnx"
    p.write_text(HDR4 + (f + i if order == "fnav_first" else i + f))
    af0, dsrc = _gal_af0(str(p))
    assert dsrc % 2 == 1 and af0 == INAV_AF0


GAL_1210_INAV_THEN_FNAV = """E36 2026 10 09 12 10 00-6.332836346701D-04-5.201172825764D-12 0.000000000000D+00
     3.100000000000D+01-1.487187500000D+02 2.395814080987D-09-2.726300187540D+00
    -6.781890988350D-06 2.229309175164D-04 1.220777630806D-05 5.440606832504D+03
     4.758000000000D+05 2.607703208923D-08-5.917909765953D-01-1.303851604462D-08
     9.971036042245D-01 9.287500000000D+01-1.372402375485D+00-5.294149093989D-09
    -3.903734034763D-10 5.130000000000D+02 2.439000000000D+03 0.000000000000D+00
     3.120000000000D+00 0.000000000000D+00 3.492459654808D-09 3.958120942116D-09
     4.767940000000D+05
E36 2026 10 09 12 10 00-6.332848570310D-04-5.201172825764D-12 0.000000000000D+00
     3.100000000000D+01-1.487187500000D+02 2.395814080987D-09-2.726300187540D+00
    -6.781890988350D-06 2.229309175164D-04 1.220777630806D-05 5.440606832504D+03
     4.758000000000D+05 2.607703208923D-08-5.917909765953D-01-1.303851604462D-08
     9.971036042245D-01 9.287500000000D+01-1.372402375485D+00-5.294149093989D-09
    -3.903734034763D-10 2.580000000000D+02 2.439000000000D+03 0.000000000000D+00
     3.120000000000D+00 0.000000000000D+00 3.492459654808D-09 0.000000000000D+00
     4.768000000000D+05
"""


def test_galileo_mixed_order_across_epochs_rinex3(tmp_path):
    """11:10 has F/NAV first, 12:10 has I/NAV first (as in real station files)."""
    p = tmp_path / "gal3mix.rnx"
    p.write_text(HDR3 + GAL_FNAV + GAL_INAV + GAL_1210_INAV_THEN_FNAV)
    nav = convert_eph.load_rinex_nav(str(p))
    d = nav.sel(sv="E36").dropna(dim="time", how="all")
    assert [int(x) % 2 for x in d["DataSrc"].values] == [1, 1]
    assert float(d["SVclockBias"].values[0]) == INAV_AF0
    assert float(d["SVclockBias"].values[1]) == -6.332836346701e-04


GAL_E02_513_THEN_516 = """E02 2026 02 09 19 00 00 1.683452865109e-05 2.856381797756e-12 0.000000000000e+00
     4.000000000000e+00 6.421875000000e+01 3.215491080967e-09 2.222952385518e+00
     3.041699528694e-06 5.290118278936e-04 8.843839168549e-06 5.440619571686e+03
     1.548000000000e+05-1.676380634308e-08 1.521238415926e+00 1.303851604462e-08
     9.618345189427e-01 1.539062500000e+02 8.163085603338e-01-5.470585014681e-09
    -1.071473202588e-10 5.130000000000e+02 2.405000000000e+03 0.000000000000e+00
     3.120000000000e+00 0.000000000000e+00-3.492459654808e-09-4.190951585770e-09
     1.554640000000e+05                                                         
E02 2026 02 09 21 10 00 1.685688039288e-05 2.870592652471e-12 0.000000000000e+00
     1.700000000000e+01 7.206250000000e+01 3.095486082277e-09-3.091703890631e+00
     3.160908818245e-06 5.274643190205e-04 9.210780262947e-06 5.440622989655e+03
     1.626000000000e+05-4.842877388000e-08 1.521195874268e+00 0.000000000000e+00
     9.618337787062e-01 1.418125000000e+02 8.147877019211e-01-5.459155967187e-09
    -1.296482575132e-10 5.160000000000e+02 2.405000000000e+03 0.000000000000e+00
     3.120000000000e+00 0.000000000000e+00-3.492459654808e-09-4.190951585770e-09
     1.632640000000e+05                                                         
"""


def test_galileo_e5b_inav_record_is_kept(tmp_path):
    """DataSrc 516 (I/NAV E5b-I) is I/NAV like 513 and must not be dropped as F/NAV."""
    p = tmp_path / "gal516.rnx"
    p.write_text(HDR3 + GAL_E02_513_THEN_516)
    nav = convert_eph.load_rinex_nav(str(p))
    d = nav.sel(sv="E02").dropna(dim="time", how="all")
    assert [int(x) for x in d["DataSrc"].values] == [513, 516]
    assert float(d["SVclockBias"].values[1]) == 1.685688039288e-05
    assert convert_eph._is_inav(516) and convert_eph._is_inav(513) and convert_eph._is_inav(517)
    assert not convert_eph._is_inav(258) and not convert_eph._is_inav(2)


GAL_BLANK_DSRC = GAL_INAV.replace(" 5.130000000000D+02", " " * 19)


def test_rinex4_blank_datasrc_record_loads(tmp_path):
    """A blank DataSrc field (NaN) must not abort loading."""
    p = tmp_path / "blank4.rnx"
    p.write_text(HDR4 + "> EPH E36 INAV\n" + GAL_BLANK_DSRC)
    d = convert_eph.load_rinex_nav(str(p)).sel(sv="E36")
    assert math.isnan(float(d["DataSrc"].values[0]))
    assert float(d["SVclockBias"].values[0]) == INAV_AF0


def test_rinex4_inav_wins_over_blank_datasrc(tmp_path):
    p = tmp_path / "blank4dup.rnx"
    p.write_text(HDR4 + "> EPH E36 INAV\n" + GAL_INAV + "> EPH E36 INAV\n" + GAL_BLANK_DSRC)
    nav = convert_eph.load_rinex_nav(str(p))
    assert float(nav.sel(sv="E36")["DataSrc"].values[0]) == 513.0


def test_is_inav_edge_values():
    assert not convert_eph._is_inav(float("nan"))
    assert not convert_eph._is_inav(0)
    assert convert_eph._is_inav(4.0) and convert_eph._is_inav(517.0)


def test_rinex4_records_with_blank_fields(tmp_path):
    """Blank or malformed STO/ION fields: the record is skipped, the scan goes on."""
    body = (
        "> STO G21 LNAV\n"
        "    2026 10 10 19 50 24 GPUT                                  UTC(USNO)\n"
        "     4.752840000000E+05" + " " * 19 + "-1.243449787580E-14 0.000000000000E+00\n"
        "> STO E19 IFNV\n"
        "    2026 10 09 00 00 00 GAUT                                  UTCGAL\n"
        "     4.320000000000E+05 x.xxx 0.000000000000E+00 0.000000000000E+00\n"
        "> ION G21 LNAV\n"
        "    2026 10 09 11 59 48 1.955777406693E-08" + " " * 19 + "-1.192092895508E-07\n"
        "     0.000000000000E+00 1.228800000000E+05-1.638400000000E+04-2.621440000000E+05\n"
        "     1.966080000000E+05 0.000000000000E+00\n"
        "> ION E19 IFNV\n"
        "    2026 10 09 11 59 57\n"
    )
    p = tmp_path / "blank.rnx"
    p.write_text(HDR4 + body)
    r = convert_eph.parse_rinex4_records(str(p))
    assert not r


def test_leap_seconds_partial_future_fields(tmp_path):
    path = _write(tmp_path, [("    18    18  1929", "LEAP SECONDS"), ("", "END OF HEADER")])
    h = convert_eph.parse_rinex_header(path)
    assert h["LEAP"] == 18 and "LEAP_FUTURE" not in h
    path = _write(tmp_path, [("    xx", "LEAP SECONDS"), ("    18", "LEAP SECONDS"), ("", "END OF HEADER")])
    assert convert_eph.parse_rinex_header(path)["LEAP"] == 18
