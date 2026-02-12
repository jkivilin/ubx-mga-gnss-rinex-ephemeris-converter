#!/usr/bin/env python3
"""
ubx_tool.py -- UBX serial tool for u-blox receivers.

Interact with u-blox GNSS receivers over UART serial or TCP:
  - Poll and display status messages (NAV-SAT, NAV-ORB, etc.)
  - Dump navigation database (MGA-DBD) for backup/comparison
  - Send UBX binary files to receiver (MGA data, config, etc.)
  - Reset receiver navigation database (cold/warm/hot start)
  - Measure Time To First Fix (TTFF)
  - Monitor receiver output in real-time

Supports direct serial and TCP connection via serial-mux server.
TCP mode skips UART configuration and blocks commands that would
change baud rate or reset the receiver.

Requires: pyserial, pyubx2

Usage (serial):
  ubx_tool.py /dev/ttyXXX poll NAV-SAT
  ubx_tool.py /dev/ttyXXX dump-dbd [output.ubx]
  ubx_tool.py /dev/ttyXXX send input.ubx [-v] [--assist --lat LAT --lon LON]
  ubx_tool.py /dev/ttyXXX reset cold|warm|hot|eph
  ubx_tool.py /dev/ttyXXX ttff [--timeout 120] [--3d]

Usage (TCP via serial-mux):
  ubx_tool.py --tcp localhost:4000 poll NAV-SAT
  ubx_tool.py --tcp localhost:4000 send mga-data.ubx --assist --lat 65 --lon 25
  ubx_tool.py --tcp localhost:4000 dump-dbd backup.ubx
  ubx_tool.py --tcp localhost:4000 ttff --3d

TTFF test workflow:
  ubx_tool.py /dev/ttyXXX reset cold
  ubx_tool.py /dev/ttyXXX send eph.ubx --assist --lat LAT --lon LON -v
  ubx_tool.py /dev/ttyXXX ttff --3d
"""

import argparse
import sys
import time
import struct
import math
import io
import socket
import select
from datetime import datetime, timezone

import serial
from pyubx2 import (
    UBXMessage, UBXReader, UBXParseError,
    POLL, GET, SET,
    UBX_PROTOCOL, NMEA_PROTOCOL, ERR_LOG,
)

# Default serial settings matching the C config tool
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 3  # seconds
INTER_MSG_DELAY = 0.05  # 50ms between sent messages


class TcpStream:
    """TCP socket wrapper with pyserial-compatible interface.

    Allows pyubx2 UBXReader (which expects a serial-like stream) to work
    over a TCP connection to a serial-mux server.  The mux broadcasts
    receiver data to all TCP clients and writes client data to the serial
    port, but is not UBX-aware -- so commands that change the UART
    configuration (baud rate, port settings, hardware reset) must not be
    sent through this interface.
    """

    def __init__(self, host, port, timeout=DEFAULT_TIMEOUT):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.connect((host, port))
        except (socket.error, OSError) as e:
            print(f"Error connecting to {host}:{port}: {e}", file=sys.stderr)
            sys.exit(1)
        self._buf = b''
        self.is_tcp = True  # marker for TCP mode checks

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value
        if value is not None:
            self._sock.settimeout(value)
        else:
            self._sock.settimeout(None)  # blocking

    def read(self, size=1):
        """Read up to `size` bytes, respecting timeout like pyserial."""
        # Return from internal buffer first
        if len(self._buf) >= size:
            data, self._buf = self._buf[:size], self._buf[size:]
            return data

        # Need more data from socket
        result = bytearray(self._buf)
        self._buf = b''
        remaining = size - len(result)
        deadline = time.monotonic() + (self._timeout or 60)

        while remaining > 0:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                ready, _, _ = select.select([self._sock], [], [], min(left, 0.5))
                if ready:
                    chunk = self._sock.recv(remaining)
                    if not chunk:
                        break  # connection closed
                    result.extend(chunk)
                    remaining -= len(chunk)
            except (socket.timeout, socket.error):
                break

        return bytes(result)

    def write(self, data):
        """Send data over TCP."""
        try:
            self._sock.sendall(data)
            return len(data)
        except socket.error as e:
            raise serial.SerialException(f"TCP write error: {e}") from e

    def flush(self):
        """No-op for TCP (TCP_NODELAY is set)."""

    def reset_input_buffer(self):
        """Drain any pending data from the socket."""
        self._buf = b''
        self._sock.setblocking(False)
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
        except (BlockingIOError, socket.error):
            pass
        finally:
            self._sock.settimeout(self._timeout)

    @property
    def baudrate(self):
        """Not meaningful for TCP -- return a placeholder."""
        return 0

    @baudrate.setter
    def baudrate(self, value):
        """Baud rate changes are not allowed over TCP."""
        raise RuntimeError(
            "Cannot change baud rate over TCP -- the serial-mux "
            "server manages the physical UART"
        )

    def readline(self):
        """Read until newline, matching pyserial's readline interface."""
        result = bytearray()
        while True:
            byte = self.read(1)
            if not byte:
                break
            result.extend(byte)
            if byte == b'\n':
                break
        return bytes(result)

    def close(self):
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def __repr__(self):
        return f"TcpStream({self._host}:{self._port})"


def open_serial(port, baud=DEFAULT_BAUD, timeout=DEFAULT_TIMEOUT):
    """Open serial port for UBX communication."""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        # Flush any pending data
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"Error opening {port}: {e}", file=sys.stderr)
        sys.exit(1)


def send_ubx(ser, msg, delay=INTER_MSG_DELAY):
    """Send a UBXMessage and wait briefly for receiver to process."""
    ser.write(msg.serialize())
    ser.flush()
    if delay:
        time.sleep(delay)


def read_responses(ser, timeout=None, msg_filter=None, max_messages=None):
    """
    Read UBX/NMEA messages from serial until timeout.

    Args:
        ser: serial port
        timeout: override port timeout for this read session
        msg_filter: if set, only yield messages matching this identity string
        max_messages: stop after this many matching messages

    Yields:
        (raw_bytes, parsed_message) tuples
    """
    old_timeout = ser.timeout
    if timeout is not None:
        ser.timeout = timeout

    ubr = UBXReader(
        ser,
        protfilter=UBX_PROTOCOL | NMEA_PROTOCOL,
        quitonerror=ERR_LOG,
        msgmode=GET,
    )

    count = 0
    try:
        for raw, parsed in ubr:
            if parsed is None:
                continue
            if msg_filter and parsed.identity != msg_filter:
                continue
            yield raw, parsed
            count += 1
            if max_messages and count >= max_messages:
                break
    except (UBXParseError, serial.SerialException, socket.error) as e:
        print(f"Read error: {e}", file=sys.stderr)
    finally:
        ser.timeout = old_timeout


def send_cfg_prt_ubx_nmea(ser, baud=DEFAULT_BAUD):
    """Send CFG-PRT to enable UBX+NMEA on UART1 input and output."""
    msg = UBXMessage('CFG', 'CFG-PRT', SET,
        portID=1, reserved0=0,
        enable=0, pol=0, pin=0, thres=0,
        charLen=3, parity=4, nStopBits=0,  # 8N1
        baudRate=baud,
        inUBX=1, inNMEA=1, inRTCM=0, inRTCM3=0,
        outUBX=1, outNMEA=1, outRTCM3=0,
        extendedTxTimeout=0, reserved1=0,
    )
    send_ubx(ser, msg)


def ensure_ubx_output(ser, baud=DEFAULT_BAUD):
    """
    Make sure UBX output is enabled. Sends CFG-PRT twice:
    once at current baud (in case UBX input is already enabled),
    and the message is idempotent so no harm if already configured.

    In TCP mode, always uses DEFAULT_BAUD (115200) regardless of the
    requested baud -- the serial-mux manages the physical UART and
    expects the receiver to stay at the default rate.
    """
    if getattr(ser, 'is_tcp', False):
        if baud != DEFAULT_BAUD:
            print(f"  TCP mode: using {DEFAULT_BAUD} baud (ignoring --baud {baud})",
                  file=sys.stderr)
            baud = DEFAULT_BAUD
    print("Enabling UBX output protocol...", file=sys.stderr)
    send_cfg_prt_ubx_nmea(ser, baud)
    time.sleep(0.1)
    # Flush any response
    ser.reset_input_buffer()


def wait_for_ack(ser, msg_class, msg_id, timeout=2.0):
    """
    Wait for ACK-ACK or ACK-NAK for a specific message class/id.

    Returns:
        True if ACK-ACK received
        False if ACK-NAK received
        None if timeout
    """
    deadline = time.monotonic() + timeout
    for _, parsed in read_responses(ser, timeout=timeout):
        if parsed.identity == 'ACK-ACK':
            if parsed.clsID == msg_class and parsed.msgID == msg_id:
                return True
        elif parsed.identity == 'ACK-NAK':
            if parsed.clsID == msg_class and parsed.msgID == msg_id:
                return False
        # Check deadline (NMEA keeps flowing, so the generator won't time out)
        if time.monotonic() > deadline:
            break
    return None


# ---- Subcommand: poll ----

# Known poll-able messages and their expected response
POLL_MESSAGES = {
    'NAV-SAT':  ('NAV', 'NAV-SAT'),
    'NAV-ORB':  ('NAV', 'NAV-ORB'),
    'NAV-PVT':  ('NAV', 'NAV-PVT'),
    'NAV-STATUS': ('NAV', 'NAV-STATUS'),
    'NAV-CLOCK': ('NAV', 'NAV-CLOCK'),
    'NAV-POSLLH': ('NAV', 'NAV-POSLLH'),
    'NAV-TIMEGPS': ('NAV', 'NAV-TIMEGPS'),
    'NAV-TIMEUTC': ('NAV', 'NAV-TIMEUTC'),
    'CFG-PRT':  ('CFG', 'CFG-PRT'),
    'CFG-GNSS': ('CFG', 'CFG-GNSS'),
    'CFG-NAV5': ('CFG', 'CFG-NAV5'),
    'MON-VER':  ('MON', 'MON-VER'),
    'MON-HW':   ('MON', 'MON-HW'),
}


def format_nav_sat(parsed):
    """Format NAV-SAT response into readable table."""
    lines = []
    lines.append(f"NAV-SAT: iTOW={parsed.iTOW}ms, version={parsed.version}, "
                 f"numSvs={parsed.numSvs}")
    lines.append("")
    lines.append(f"{'SV':>4s} {'GNSS':>6s} {'Elev':>5s} {'Azim':>5s} "
                 f"{'CN0':>4s} {'PRres':>7s} {'Quality':>8s} "
                 f"{'Health':>7s} {'Eph':>4s} {'Alm':>4s}")
    lines.append("-" * 70)

    gnss_names = {0: 'GPS', 1: 'SBAS', 2: 'GAL', 3: 'BDS',
                  4: 'IMES', 5: 'QZSS', 6: 'GLO'}

    # Standard RINEX constellation prefixes
    gnss_prefix = {0: 'G', 1: 'S', 2: 'E', 3: 'C',
                   4: 'I', 5: 'J', 6: 'R'}

    quality_names = {
        0: 'none', 1: 'search', 2: 'acquir', 3: 'unusbl',
        4: 'locked', 5: 'code+t', 6: 'code+c', 7: 'carr',
    }

    health_names = {0: '---', 1: 'OK', 2: 'bad', 3: '???'}

    # pyubx2 repeated block field naming: try common patterns
    def get_field(name, idx):
        """Try different pyubx2 naming conventions for repeated fields."""
        for fmt in [f'{name}_{idx:02d}', f'{name}_{idx:01d}', f'{name}{idx:02d}']:
            val = getattr(parsed, fmt, None)
            if val is not None:
                return val
        return None

    rows_printed = 0
    for i in range(1, parsed.numSvs + 1):
        gnss_id = get_field('gnssId', i)
        sv_id = get_field('svId', i)

        if gnss_id is None or sv_id is None:
            break

        cn0 = get_field('cno', i) or 0
        elev = get_field('elev', i) or 0
        azim = get_field('azim', i) or 0
        pr_res = get_field('prRes', i) or 0.0
        quality = get_field('qualityInd', i)
        health = get_field('health', i)  # pyubx2 uses 'health', not 'svHealth'
        eph_avail = get_field('ephAvail', i)
        alm_avail = get_field('almAvail', i)

        gnss_name = gnss_names.get(gnss_id, f'?{gnss_id}')
        qual_name = quality_names.get(quality, f'?{quality}') if quality is not None else '?'
        hlth_name = health_names.get(health, f'?{health}') if health is not None else '?'
        eph_str = 'Y' if eph_avail else 'N' if eph_avail is not None else '?'
        alm_str = 'Y' if alm_avail else 'N' if alm_avail is not None else '?'

        prefix = gnss_prefix.get(gnss_id, '?')
        sv_label = f"{prefix}{sv_id:02d}"
        pr_res_str = f"{pr_res:>7.1f}" if isinstance(pr_res, (int, float)) else f"{'?':>7s}"

        lines.append(
            f"{sv_label:>4s} {gnss_name:>6s} {elev:>5d} {azim:>5d} "
            f"{cn0:>4d} {pr_res_str} {qual_name:>8s} {hlth_name:>7s} "
            f"{eph_str:>4s} {alm_str:>4s}"
        )
        rows_printed += 1

    if rows_printed == 0 and parsed.numSvs > 0:
        # Field naming didn't match - fall back to raw output
        lines.append("  (could not parse satellite rows - showing raw)")
        lines.append(str(parsed))

    return "\n".join(lines)


def format_nav_pvt(parsed):
    """Format NAV-PVT response."""
    fix_types = {0: 'No fix', 1: '2D', 2: '3D', 3: '3D+DGPS',
                 4: 'Time only', 5: 'Dead reck'}
    fix = fix_types.get(parsed.fixType, f'?{parsed.fixType}')
    dt = f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d} " \
         f"{parsed.hour:02d}:{parsed.min:02d}:{parsed.second:02d}"
    # pyubx2 already applies 1e-7 scaling: parsed.lat/lon are in degrees
    lat = parsed.lat
    lon = parsed.lon
    alt = parsed.hMSL * 1e-3  # mm to m

    lines = [
        f"NAV-PVT: {dt} UTC",
        f"  Fix: {fix}, satellites: {parsed.numSV}",
        f"  Position: {lat:.7f}deg N {lon:.7f}deg E, alt {alt:.1f}m",
        f"  Accuracy: horiz {parsed.hAcc*1e-3:.1f}m, vert {parsed.vAcc*1e-3:.1f}m",
    ]
    return "\n".join(lines)


def format_generic(parsed):
    """Generic message formatter - just use pyubx2's __str__."""
    return str(parsed)


FORMATTERS = {
    'NAV-SAT': format_nav_sat,
    'NAV-PVT': format_nav_pvt,
}


def cmd_poll(ser, args):
    """Poll a specific UBX message and display the response."""
    msg_name = args.message.upper()

    if msg_name not in POLL_MESSAGES:
        print(f"Unknown message: {msg_name}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(POLL_MESSAGES.keys()))}", file=sys.stderr)
        sys.exit(1)

    ensure_ubx_output(ser, args.baud)

    cls_name, msg_id_name = POLL_MESSAGES[msg_name]

    # Special case: CFG-PRT poll needs portID
    if msg_name == 'CFG-PRT':
        poll_msg = UBXMessage(cls_name, msg_id_name, POLL, portID=1)
    else:
        poll_msg = UBXMessage(cls_name, msg_id_name, POLL)

    print(f"Polling {msg_name}...", file=sys.stderr)
    send_ubx(ser, poll_msg)

    # Read response
    for _, parsed in read_responses(ser, timeout=2.0):
        if parsed.identity == msg_name:
            formatter = FORMATTERS.get(msg_name, format_generic)
            print(formatter(parsed))
            return

    print(f"No response received for {msg_name}", file=sys.stderr)
    sys.exit(1)


# ---- Subcommand: dump-dbd ----

def cmd_dump_dbd(ser, args):
    """Dump MGA-DBD navigation database to file."""
    output = args.output or f"mga-dbd-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.ubx"

    ensure_ubx_output(ser, args.baud)

    poll_msg = UBXMessage('MGA', 'MGA-DBD', POLL)

    print("Polling MGA-DBD (navigation database dump)...", file=sys.stderr)
    ser.reset_input_buffer()
    send_ubx(ser, poll_msg)

    mga_messages = []

    # MGA-DBD response: multiple messages, terminated by empty payload MGA-DBD.
    # Problem: NMEA keeps flowing, so serial timeout never fires.
    # Use deadline: if no new MGA-DBD received within gap_timeout, assume done.
    gap_timeout = 3.0  # seconds with no MGA-DBD = done
    overall_timeout = 30.0  # hard limit

    start_time = time.monotonic()
    last_dbd_time = start_time

    ubr = UBXReader(
        ser,
        protfilter=UBX_PROTOCOL | NMEA_PROTOCOL,
        quitonerror=ERR_LOG,
        msgmode=GET,
    )

    try:
        for raw, parsed in ubr:
            now = time.monotonic()

            # Hard timeout
            if now - start_time > overall_timeout:
                print(f"  Overall timeout ({overall_timeout}s) reached.", file=sys.stderr)
                break

            # Gap timeout: no MGA-DBD for too long after we started receiving them
            if mga_messages and (now - last_dbd_time > gap_timeout):
                break

            # Initial timeout: no MGA-DBD at all yet
            if not mga_messages and (now - start_time > gap_timeout * 2):
                break

            if parsed is None:
                continue

            if parsed.identity == 'MGA-DBD':
                payload_len = len(raw) - 8  # sync(2) + cls(1) + id(1) + len(2) + cksum(2)
                if payload_len <= 0:
                    # Empty MGA-DBD = end of dump
                    break
                mga_messages.append(raw)
                last_dbd_time = now
                if len(mga_messages) % 50 == 0:
                    print(f"  ...received {len(mga_messages)} messages", file=sys.stderr)

    except (UBXParseError, serial.SerialException, socket.error) as e:
        print(f"Read error: {e}", file=sys.stderr)

    if not mga_messages:
        print("No MGA-DBD data received. Is the receiver running and has a fix?",
              file=sys.stderr)
        sys.exit(1)

    # Write to file
    with open(output, 'wb') as f:
        for msg in mga_messages:
            f.write(msg)

    total_bytes = sum(len(m) for m in mga_messages)
    print(f"Saved {len(mga_messages)} MGA-DBD messages ({total_bytes} bytes) to {output}",
          file=sys.stderr)
    print(output)


# ---- GPS Subframe Decoding (IS-GPS-200) ----

def extract_bits(word24, start_bit, num_bits):
    """
    Extract bits from a 24-bit GPS data word (as stored by u-blox in AID-EPH).

    u-blox AID-EPH stores 24 data bits per word (parity already stripped),
    in bits [23:0] of a 32-bit container. Bits [31:24] are zero/unused.

    IS-GPS-200 bit numbering: bit 1 = MSB (first transmitted data bit),
    bit 24 = LSB (last data bit before parity).

    Args:
        word24: 32-bit value with 24-bit GPS data word in bits [23:0]
        start_bit: IS-GPS-200 bit number (1-based, 1=MSB)
        num_bits: number of bits to extract
    Returns:
        extracted unsigned value
    """
    # IS-GPS-200 bit N maps to container bit (24 - N)
    shift = 24 - start_bit - num_bits + 1
    mask = (1 << num_bits) - 1
    return (word24 >> shift) & mask


def twos_complement(val, bits):
    """Convert unsigned value to signed (two's complement)."""
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def decode_gps_subframes(sf1_words, sf2_words, sf3_words):
    """
    Decode GPS ephemeris from subframe 1/2/3 words.

    Args:
        sf1_words: list of 8 U32 values (subframe 1, words 3-10)
        sf2_words: list of 8 U32 values (subframe 2, words 3-10)
        sf3_words: list of 8 U32 values (subframe 3, words 3-10)

    Returns:
        dict of ephemeris parameters in RINEX-equivalent units (SI, radians)
    """
    # Subframe 1: Clock and health parameters
    # Word 3 (sf1_words[0]): WN, L2 code, URA, SV health, IODC MSBs
    wn = extract_bits(sf1_words[0], 1, 10)
    _ = extract_bits(sf1_words[0], 11, 2)  # l2_code (not used in MGA-GPS-EPH)
    ura_index = extract_bits(sf1_words[0], 13, 4)
    sv_health = extract_bits(sf1_words[0], 17, 6)
    iodc_msb = extract_bits(sf1_words[0], 23, 2)

    # Word 4 (sf1_words[1]): L2 P flag, reserved
    # Word 5 (sf1_words[2]): reserved
    # Word 6 (sf1_words[3]): reserved

    # Word 7 (sf1_words[4]): TGD (bits 17-24)
    tgd_raw = twos_complement(extract_bits(sf1_words[4], 17, 8), 8)
    tgd = tgd_raw * (2**-31)  # seconds

    # Word 8 (sf1_words[5]): IODC LSBs (bits 1-8), toc (bits 9-24)
    iodc_lsb = extract_bits(sf1_words[5], 1, 8)
    iodc = (iodc_msb << 8) | iodc_lsb
    toc_raw = extract_bits(sf1_words[5], 9, 16)
    toc = toc_raw * 16  # seconds

    # Word 9 (sf1_words[6]): af2 (bits 1-8), af1 (bits 9-24)
    af2_raw = twos_complement(extract_bits(sf1_words[6], 1, 8), 8)
    af1_raw = twos_complement(extract_bits(sf1_words[6], 9, 16), 16)
    af2 = af2_raw * (2**-55)  # sec/sec^2
    af1 = af1_raw * (2**-43)  # sec/sec

    # Word 10 (sf1_words[7]): af0 (bits 1-22)
    af0_raw = twos_complement(extract_bits(sf1_words[7], 1, 22), 22)
    af0 = af0_raw * (2**-31)  # seconds

    # Subframe 2: Orbit parameters (part 1)
    # Word 3 (sf2_words[0]): IODE (bits 1-8), Crs (bits 9-24)
    iode_sf2 = extract_bits(sf2_words[0], 1, 8)
    crs_raw = twos_complement(extract_bits(sf2_words[0], 9, 16), 16)
    crs = crs_raw * (2**-5)  # meters

    # Word 4 (sf2_words[1]): deltaN (bits 1-16), M0 MSBs (bits 17-24)
    deltaN_raw = twos_complement(extract_bits(sf2_words[1], 1, 16), 16)
    m0_msb = extract_bits(sf2_words[1], 17, 8)

    # Word 5 (sf2_words[2]): M0 LSBs (bits 1-24)
    m0_lsb = extract_bits(sf2_words[2], 1, 24)
    m0_raw = twos_complement((m0_msb << 24) | m0_lsb, 32)

    deltaN = deltaN_raw * (2**-43) * math.pi  # rad/sec
    m0 = m0_raw * (2**-31) * math.pi  # radians

    # Word 6 (sf2_words[3]): Cuc (bits 1-16), e MSBs (bits 17-24)
    cuc_raw = twos_complement(extract_bits(sf2_words[3], 1, 16), 16)
    e_msb = extract_bits(sf2_words[3], 17, 8)

    # Word 7 (sf2_words[4]): e LSBs (bits 1-24)
    e_lsb = extract_bits(sf2_words[4], 1, 24)
    e_raw = (e_msb << 24) | e_lsb  # unsigned

    cuc = cuc_raw * (2**-29)  # radians
    e = e_raw * (2**-33)  # dimensionless

    # Word 8 (sf2_words[5]): Cus (bits 1-16), sqrtA MSBs (bits 17-24)
    cus_raw = twos_complement(extract_bits(sf2_words[5], 1, 16), 16)
    sqrtA_msb = extract_bits(sf2_words[5], 17, 8)

    # Word 9 (sf2_words[6]): sqrtA LSBs (bits 1-24)
    sqrtA_lsb = extract_bits(sf2_words[6], 1, 24)
    sqrtA_raw = (sqrtA_msb << 24) | sqrtA_lsb  # unsigned

    cus = cus_raw * (2**-29)  # radians
    sqrtA = sqrtA_raw * (2**-19)  # m^0.5

    # Word 10 (sf2_words[7]): toe (bits 1-16), fit interval (bit 17), AODO (bits 18-22)
    toe_raw = extract_bits(sf2_words[7], 1, 16)
    fit_interval_flag = extract_bits(sf2_words[7], 17, 1)
    toe = toe_raw * 16  # seconds

    # Subframe 3: Orbit parameters (part 2)
    # Word 3 (sf3_words[0]): Cic (bits 1-16), Omega0 MSBs (bits 17-24)
    cic_raw = twos_complement(extract_bits(sf3_words[0], 1, 16), 16)
    omega0_msb = extract_bits(sf3_words[0], 17, 8)

    # Word 4 (sf3_words[1]): Omega0 LSBs (bits 1-24)
    omega0_lsb = extract_bits(sf3_words[1], 1, 24)
    omega0_raw = twos_complement((omega0_msb << 24) | omega0_lsb, 32)

    cic = cic_raw * (2**-29)  # radians
    omega0 = omega0_raw * (2**-31) * math.pi  # radians

    # Word 5 (sf3_words[2]): Cis (bits 1-16), i0 MSBs (bits 17-24)
    cis_raw = twos_complement(extract_bits(sf3_words[2], 1, 16), 16)
    i0_msb = extract_bits(sf3_words[2], 17, 8)

    # Word 6 (sf3_words[3]): i0 LSBs (bits 1-24)
    i0_lsb = extract_bits(sf3_words[3], 1, 24)
    i0_raw = twos_complement((i0_msb << 24) | i0_lsb, 32)

    cis = cis_raw * (2**-29)  # radians
    i0 = i0_raw * (2**-31) * math.pi  # radians

    # Word 7 (sf3_words[4]): Crc (bits 1-16), omega MSBs (bits 17-24)
    crc_raw = twos_complement(extract_bits(sf3_words[4], 1, 16), 16)
    omega_msb = extract_bits(sf3_words[4], 17, 8)

    # Word 8 (sf3_words[5]): omega LSBs (bits 1-24)
    omega_lsb = extract_bits(sf3_words[5], 1, 24)
    omega_raw = twos_complement((omega_msb << 24) | omega_lsb, 32)

    crc = crc_raw * (2**-5)  # meters
    omega = omega_raw * (2**-31) * math.pi  # radians

    # Word 9 (sf3_words[6]): OmegaDot (bits 1-24)
    omegaDot_raw = twos_complement(extract_bits(sf3_words[6], 1, 24), 24)
    omegaDot = omegaDot_raw * (2**-43) * math.pi  # rad/sec

    # Word 10 (sf3_words[7]): IODE (bits 1-8), IDOT (bits 9-22)
    iode_sf3 = extract_bits(sf3_words[7], 1, 8)
    idot_raw = twos_complement(extract_bits(sf3_words[7], 9, 14), 14)
    idot = idot_raw * (2**-43) * math.pi  # rad/sec

    return {
        # Subframe 1
        'week': wn,
        'ura_index': ura_index,
        'sv_health': sv_health,
        'iodc': iodc,
        'tgd': tgd,
        'toc': toc,
        'af2': af2,
        'af1': af1,
        'af0': af0,
        # Subframe 2
        'iode': iode_sf2,
        'crs': crs,
        'deltaN': deltaN,
        'm0': m0,
        'cuc': cuc,
        'e': e,
        'cus': cus,
        'sqrtA': sqrtA,
        'toe': toe,
        'fit_interval': fit_interval_flag,
        # Subframe 3
        'cic': cic,
        'omega0': omega0,
        'cis': cis,
        'i0': i0,
        'crc': crc,
        'omega': omega,
        'omegaDot': omegaDot,
        'idot': idot,
        'iode_sf3': iode_sf3,
        # Raw integer values (for MGA-GPS-EPH packing)
        '_raw': {
            'tgd': tgd_raw, 'iodc': iodc, 'toc': toc_raw, 'af2': af2_raw,
            'af1': af1_raw, 'af0': af0_raw, 'crs': crs_raw, 'deltaN': deltaN_raw,
            'm0': m0_raw, 'cuc': cuc_raw, 'e': e_raw, 'cus': cus_raw,
            'sqrtA': sqrtA_raw, 'toe': toe_raw, 'cic': cic_raw, 'omega0': omega0_raw,
            'cis': cis_raw, 'i0': i0_raw, 'crc': crc_raw, 'omega': omega_raw,
            'omegaDot': omegaDot_raw, 'idot': idot_raw,
            'ura_index': ura_index, 'sv_health': sv_health,
            'fit_interval': fit_interval_flag,
        },
    }


# URA index to meters lookup (IS-GPS-200)
URA_TABLE = [2.4, 3.4, 4.85, 6.85, 9.65, 13.65, 24, 48,
             96, 192, 384, 768, 1536, 3072, 6144, 6145]


def ura_meters_to_index(ura_m):
    """Convert URA in meters to URA index (0-15)."""
    for i, threshold in enumerate(URA_TABLE):
        if ura_m <= threshold:
            return i
    return 15


def build_mga_gps_eph(sv_id, raw, mga_class=0x13, mga_id=0x00):
    """
    Build UBX-MGA-GPS-EPH (or MGA-QZSS-EPH) payload from raw integer values.

    Args:
        sv_id: satellite PRN (1-32 for GPS, 1-10 for QZSS)
        raw: dict of raw integer values from decode_gps_subframes()['_raw']
        mga_class: 0x13 (MGA)
        mga_id: 0x00 (GPS) or 0x05 (QZSS)

    Returns:
        complete UBX message bytes (with sync, class, id, length, payload, checksum)
    """
    # Build 68-byte payload per pyubx2-verified field order
    payload = struct.pack('<BBBBBBBb',
        0x01,                   # type
        0x00,                   # version
        sv_id,                  # svId
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
        raw['af0'],             # af0 (I4) -- 22-bit value sign-extended to 32
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
        raw['omegaDot'],        # omegaDot (I4) -- 24-bit sign-extended to 32
    )
    payload += struct.pack('<hH',
        raw['idot'],            # idot (I2) -- 14-bit sign-extended to 16
        0x0000,                 # reserved2 (U2)
    )

    assert len(payload) == 68, f"MGA-GPS-EPH payload must be 68 bytes, got {len(payload)}"

    # Build complete UBX frame
    header = struct.pack('<2B2BH', 0xB5, 0x62, mga_class, mga_id, len(payload))
    frame_data = header[2:] + payload  # class + id + length + payload (for checksum)
    ck_a, ck_b = 0, 0
    for b in frame_data:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return header + payload + bytes([ck_a, ck_b])


def format_eph_table(satellites):
    """Format decoded ephemeris into a readable table."""
    lines = []
    lines.append(f"{'PRN':>4s} {'IODC':>5s} {'IODE':>5s} {'Toc':>7s} {'Toe':>7s} "
                 f"{'URA':>4s} {'Hlth':>5s} {'af0':>13s} {'sqrtA':>12s} {'e':>12s}")
    lines.append("-" * 90)

    for sv_id, eph in sorted(satellites.items()):
        lines.append(
            f"G{sv_id:02d}  {eph['iodc']:5d} {eph['iode']:5d} "
            f"{eph['toc']:7.0f} {eph['toe']:7.0f} "
            f"{eph['ura_index']:4d} {eph['sv_health']:5d} "
            f"{eph['af0']:13.6e} {eph['sqrtA']:12.4f} {eph['e']:12.10f}"
        )

    return "\n".join(lines)


# ---- Subcommand: dump-eph ----

def cmd_dump_eph(ser, args):
    """Poll AID-EPH, decode GPS subframes, optionally export as MGA-GPS-EPH."""
    output_raw = args.output
    output_mga = args.mga
    ts_str = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    if output_raw is None and not args.no_save:
        output_raw = f"aid-eph-{ts_str}.ubx"

    ensure_ubx_output(ser, args.baud)

    # Poll AID-EPH (empty payload = all SVs)
    poll_msg = UBXMessage('AID', 'AID-EPH', POLL)

    print("Polling AID-EPH (all satellites)...", file=sys.stderr)
    ser.reset_input_buffer()
    send_ubx(ser, poll_msg)

    # Collect responses - expect one AID-EPH per SV (up to 32)
    # SVs without ephemeris return 8-byte response (svid + how only)
    eph_messages_raw = []
    gap_timeout = 2.0
    overall_timeout = 15.0
    start_time = time.monotonic()
    last_eph_time = start_time

    ubr = UBXReader(
        ser,
        protfilter=UBX_PROTOCOL | NMEA_PROTOCOL,
        quitonerror=ERR_LOG,
        msgmode=GET,
    )

    try:
        for raw, parsed in ubr:
            now = time.monotonic()
            if now - start_time > overall_timeout:
                break
            if eph_messages_raw and (now - last_eph_time > gap_timeout):
                break
            if not eph_messages_raw and (now - start_time > gap_timeout * 2):
                break
            if parsed is None:
                continue
            if parsed.identity == 'AID-EPH':
                eph_messages_raw.append((raw, parsed))
                last_eph_time = now
    except (UBXParseError, serial.SerialException, socket.error) as e:
        print(f"Read error: {e}", file=sys.stderr)

    if not eph_messages_raw:
        print("No AID-EPH data received.", file=sys.stderr)
        sys.exit(1)

    # Save raw AID-EPH file
    if output_raw:
        with open(output_raw, 'wb') as f:
            for raw, parsed in eph_messages_raw:
                f.write(raw)
        total_bytes = sum(len(r) for r, _ in eph_messages_raw)
        print(f"Saved {len(eph_messages_raw)} AID-EPH messages ({total_bytes} bytes) to {output_raw}",
              file=sys.stderr)

    # Decode subframe words
    satellites = {}
    no_eph_count = 0

    for raw, parsed in eph_messages_raw:
        sv_id = parsed.svid
        # Check if ephemeris data is present (payload > 8 bytes = has subframe words)
        payload_len = len(raw) - 8  # sync(2) + cls(1) + id(1) + len(2) + cksum(2)
        if payload_len <= 8:
            no_eph_count += 1
            continue

        # Extract subframe words - try both pyubx2 naming conventions
        def get_sf_word(name, _parsed=parsed):
            # pyubx2 optBlock uses _01 suffix
            val = getattr(_parsed, f'{name}_01', None)
            if val is not None:
                return val
            val = getattr(_parsed, name, None)
            return val if val is not None else 0

        sf1 = [get_sf_word(f'sf1d{i}') for i in range(1, 9)]
        sf2 = [get_sf_word(f'sf2d{i}') for i in range(1, 9)]
        sf3 = [get_sf_word(f'sf3d{i}') for i in range(1, 9)]

        try:
            eph = decode_gps_subframes(sf1, sf2, sf3)
            satellites[sv_id] = eph
        except Exception as exc:
            print(f"  Warning: failed to decode G{sv_id:02d}: {exc}", file=sys.stderr)

    print(f"\nDecoded {len(satellites)} satellites with ephemeris "
          f"({no_eph_count} without)", file=sys.stderr)

    # Display table
    if satellites:
        print()
        print(format_eph_table(satellites))

    # Export as MGA-GPS-EPH
    if output_mga:
        mga_data = bytearray()
        for sv_id, eph in sorted(satellites.items()):
            msg = build_mga_gps_eph(sv_id, eph['_raw'])
            mga_data.extend(msg)
        with open(output_mga, 'wb') as f:
            f.write(mga_data)
        print(f"\nExported {len(satellites)} MGA-GPS-EPH messages to {output_mga}",
              file=sys.stderr)
        print(output_mga)


# ---- Subcommand: send ----

MGA_ACK_INFO_CODES = {
    0: 'accepted',
    1: 'no time (send MGA-INI-TIME first)',
    2: 'type not supported',
    3: 'size mismatch',
    4: 'database store failed',
    5: 'not ready',
    6: 'unknown type',
}


def enable_mga_ack(ser):
    """Enable MGA-ACK flow control via CFG-NAVX5."""
    # mask1 bit 10 = ackAid
    msg = UBXMessage('CFG', 'CFG-NAVX5', SET,
        version=0, mask1=0x0400, mask2=0, ackAiding=1)
    send_ubx(ser, msg)
    # Wait for CFG ACK
    ack = wait_for_ack(ser, 0x06, 0x23, timeout=1.0)
    return ack is True


def wait_for_mga_ack(ser, timeout=2.0, debug=False):
    """Wait for UBX-MGA-ACK-DATA0 (0x13 0x60) response.

    Returns:
        dict with 'type', 'infoCode', 'msgId' if received
        None if timeout
    """
    deadline = time.monotonic() + timeout
    for raw, parsed in read_responses(ser, timeout=timeout):
        if debug:
            print(f"    [recv] {parsed.identity} ({raw[:min(12,len(raw))].hex()}...)",
                  file=sys.stderr)
        if parsed.identity == 'MGA-ACK-DATA0':
            return {
                'type': parsed.type,
                'infoCode': parsed.infoCode,
                'msgId': parsed.msgId,
            }
        # Check deadline (NMEA keeps flowing, so the generator won't time out)
        if time.monotonic() > deadline:
            break
    return None


def cmd_send(ser, args):
    """Send a UBX binary file to the receiver."""
    input_file = args.input_file
    debug = getattr(args, 'debug', False)
    assist = getattr(args, 'assist', False)

    ensure_ubx_output(ser, args.baud)

    # Build list of (name, raw_bytes) to send
    all_raw = []

    # If --assist, prepend MGA-INI-TIME_UTC and MGA-INI-POS_LLH
    if assist:
        lat = getattr(args, 'lat', None)
        lon = getattr(args, 'lon', None)

        time_msg = build_mga_ini_time_utc()

        now = datetime.now(timezone.utc)
        print("Prepending assistance data:", file=sys.stderr)
        print(f"  MGA-INI-TIME_UTC: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC "
              f"(+/-1s)", file=sys.stderr)

        all_raw.append(('MGA-INI-TIME_UTC', time_msg))

        if lat is not None and lon is not None:
            pos_msg = build_mga_ini_pos_llh(lat, lon)
            print(f"  MGA-INI-POS_LLH:  {lat:.7f}N {lon:.7f}E", file=sys.stderr)
            all_raw.append(('MGA-INI-POS_LLH', pos_msg))
        else:
            print("  MGA-INI-POS_LLH:  skipped (use --lat/--lon to provide position)",
                  file=sys.stderr)

    # Parse file messages
    with open(input_file, 'rb') as f:
        data = f.read()

    stream = io.BytesIO(data)
    ubr = UBXReader(stream, msgmode=SET, protfilter=UBX_PROTOCOL)

    for raw, parsed in ubr:
        if parsed:
            all_raw.append((parsed.identity, raw))

    if not all_raw:
        print(f"No valid UBX messages found in {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Sending {len(all_raw)} UBX messages...", file=sys.stderr)

    for i, (name, raw) in enumerate(all_raw, 1):
        ser.write(raw)
        ser.flush()

        if args.verbose or debug:
            extra = f" ({len(raw)}B, {raw[:8].hex()}...)" if debug else ""
            print(f"  [{i}/{len(all_raw)}] {name}{extra}")

        time.sleep(INTER_MSG_DELAY)

    print(f"\nDone: {len(all_raw)} messages sent ({sum(len(r) for _, r in all_raw)} bytes)",
          file=sys.stderr)


# ---- Subcommand: reset ----

def save_config_to_bbr(ser):
    """Save current config to BBR using CFG-CFG so it survives hardware reset."""
    msg = UBXMessage('CFG', 'CFG-CFG', SET,
        clearMask=b'\x00\x00\x00\x00',
        saveMask=b'\x1f\x00\x00\x00',   # all config sections
        loadMask=b'\x00\x00\x00\x00',
        devBBR=1, devFlash=0, devEEPROM=0, devSpiFlash=0,
    )
    send_ubx(ser, msg)
    return wait_for_ack(ser, 0x06, 0x09, timeout=2.0) is True


def cmd_reset(ser, args):
    """Reset receiver navigation database."""
    mode = args.mode

    ensure_ubx_output(ser, args.baud)

    masks = {
        'cold':   (0xFFFF, 'Cold start (clear all nav data)'),
        'warm':   (0x0001, 'Warm start (clear ephemeris)'),
        'hot':    (0x0000, 'Hot start (keep everything)'),
        'eph':    (0x0001, 'Clear ephemeris only'),
    }

    navBbrMask, desc = masks[mode]

    # Hardware reset (0x00) is needed to actually clear BBR nav data.
    # GNSS-only reset (0x02) just restarts tasks without clearing.
    # But HW reset reloads port config, so save it to BBR first.
    if navBbrMask != 0x0000:
        resetMode = 0x00  # Hardware reset (watchdog) immediately
        reset_desc = "hardware watchdog"
        wait = 4.0

        # Save port config to BBR so baud rate survives the reset
        print("Saving port config to BBR...", file=sys.stderr)
        if save_config_to_bbr(ser):
            print("  Config saved", file=sys.stderr)
        else:
            print("  Warning: CFG-CFG not acknowledged", file=sys.stderr)
    else:
        resetMode = 0x02  # GNSS-only for hot start (no clear needed)
        reset_desc = "GNSS only"
        wait = 2.0

    print(desc, file=sys.stderr)
    print(f"  navBbrMask=0x{navBbrMask:04x}, resetMode=0x{resetMode:02x} ({reset_desc})",
          file=sys.stderr)

    # Build CFG-RST manually -- pyubx2 silently zeros X2 bitfield types
    payload = struct.pack('<HBB', navBbrMask, resetMode, 0)
    header = struct.pack('<2B2BH', 0xB5, 0x62, 0x06, 0x04, len(payload))
    raw = header + payload
    ck_a, ck_b = 0, 0
    for b in raw[2:]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    raw += struct.pack('<BB', ck_a, ck_b)
    ser.write(raw)
    ser.flush()

    print(f"  Waiting {wait:.0f}s for restart...", file=sys.stderr)
    time.sleep(wait)
    ser.reset_input_buffer()

    # Check if receiver responds at current baud
    alive = False
    poll_msg = UBXMessage('NAV', 'NAV-PVT', POLL)
    send_ubx(ser, poll_msg)
    deadline = time.monotonic() + 2.0
    for raw, parsed in read_responses(ser, timeout=2.0):
        if parsed.identity == 'NAV-PVT':
            alive = True
            break
        if time.monotonic() > deadline:
            break

    if not alive and resetMode == 0x00:
        # HW reset may have reverted to 9600 baud (factory default)
        print(f"  No response at {args.baud}, trying 9600 baud...", file=sys.stderr)
        ser.baudrate = 9600
        ser.reset_input_buffer()
        time.sleep(0.2)

        # Reconfigure receiver to target baud
        send_cfg_prt_ubx_nmea(ser, args.baud)
        time.sleep(0.2)
        ser.baudrate = args.baud
        ser.reset_input_buffer()
        time.sleep(0.2)

        # Save again so next reset preserves baud
        save_config_to_bbr(ser)

        # Retry
        send_ubx(ser, poll_msg)
        deadline = time.monotonic() + 2.0
        for raw, parsed in read_responses(ser, timeout=2.0):
            if parsed.identity == 'NAV-PVT':
                alive = True
                print("  Recovered via 9600 baud fallback", file=sys.stderr)
                break
            if time.monotonic() > deadline:
                break

    if alive:
        fix_map = {0: 'No fix', 1: 'DR', 2: '2D', 3: '3D', 4: '3D+DR', 5: 'Time'}
        fix_str = fix_map.get(parsed.fixType, f'type={parsed.fixType}')
        print(f"  Receiver alive: {fix_str}, {parsed.numSV} SVs", file=sys.stderr)
    else:
        print("  Warning: No response after reset", file=sys.stderr)


# ---- Subcommand: ttff ----

def build_mga_ini_time_utc():
    """Build MGA-INI-TIME_UTC message with current system time.

    Returns raw UBX message bytes.
    """
    now = datetime.now(timezone.utc)

    # MGA-INI-TIME_UTC: class=0x13, id=0x40, 24-byte payload
    # type=0x10, version=0x00, ref=0x00 (on receipt), leapSecs=18
    payload = struct.pack('<BBBb',
        0x10,   # type (TIME_UTC)
        0x00,   # version
        0x00,   # ref (none = on receipt of message)
        18,     # leapSecs (GPS-UTC as of 2026)
    )
    payload += struct.pack('<HBBBBB',
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
    )
    payload += struct.pack('<B', 0)     # bitfield0 (reserved)
    payload += struct.pack('<I', now.microsecond * 1000)  # ns
    payload += struct.pack('<H', 1)     # tAccS (1 second accuracy -- no hw sync)
    payload += struct.pack('<H', 0)     # reserved1
    payload += struct.pack('<I', 0)     # tAccNs

    assert len(payload) == 24

    # Wrap in UBX frame
    header = struct.pack('<2B2BH', 0xB5, 0x62, 0x13, 0x40, len(payload))
    raw = header + payload
    # Checksum over class, id, length, payload
    ck_a, ck_b = 0, 0
    for b in raw[2:]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    raw += struct.pack('<BB', ck_a, ck_b)
    return raw


def build_mga_ini_pos_llh(lat_deg, lon_deg, alt_m=50.0, acc_m=50.0):
    """Build MGA-INI-POS_LLH message.

    Args:
        lat_deg: latitude in degrees
        lon_deg: longitude in degrees
        alt_m: altitude in meters (default 50)
        acc_m: position accuracy in meters (default 50)

    Returns raw UBX message bytes.
    """
    # MGA-INI-POS_LLH: class=0x13, id=0x40, 20-byte payload
    payload = struct.pack('<BBH',
        0x01,   # type (POS_LLH)
        0x00,   # version
        0x0000, # reserved1
    )
    payload += struct.pack('<iiiI',
        int(lat_deg * 1e7),     # lat (1e-7 degrees, I4)
        int(lon_deg * 1e7),     # lon (1e-7 degrees, I4)
        int(alt_m * 100),       # alt (cm, I4)
        int(acc_m * 100),       # posAcc (cm, U4)
    )

    assert len(payload) == 20

    # Wrap in UBX frame
    header = struct.pack('<2B2BH', 0xB5, 0x62, 0x13, 0x40, len(payload))
    raw = header + payload
    ck_a, ck_b = 0, 0
    for b in raw[2:]:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    raw += struct.pack('<BB', ck_a, ck_b)
    return raw


def cmd_ttff(ser, args):
    """Measure Time To First Fix by polling NAV-PVT."""
    timeout = args.timeout
    min_fix = args.min_fix  # minimum fix type (2=2D, 3=3D)

    ensure_ubx_output(ser, args.baud)

    fix_names = {0: 'No fix', 1: 'DR only', 2: '2D', 3: '3D', 4: '3D+DR', 5: 'Time only'}

    print(f"Measuring TTFF (waiting for {'3D' if min_fix >= 3 else '2D'} fix, "
          f"timeout {timeout}s)...", file=sys.stderr)

    start = time.monotonic()
    last_print = 0
    fix_time = None

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            break

        # Poll NAV-PVT
        poll_msg = UBXMessage('NAV', 'NAV-PVT', POLL)
        send_ubx(ser, poll_msg)

        for _, parsed in read_responses(ser, timeout=1.5):
            if parsed.identity != 'NAV-PVT':
                continue

            elapsed = time.monotonic() - start
            fix_type = parsed.fixType
            num_sv = parsed.numSV
            fix_str = fix_names.get(fix_type, f'type={fix_type}')

            # Print status every second
            if elapsed - last_print >= 0.9:
                # Check for DGPS/RTK flags
                flags = ''
                if hasattr(parsed, 'flags'):
                    f = parsed.flags
                    if isinstance(f, int):
                        if f & 0x02:
                            flags = '+DGPS'

                hacc = getattr(parsed, 'hAcc', 0) / 1000.0  # mm -> m
                print(f"  {elapsed:5.1f}s: {fix_str}{flags}, {num_sv:2d} SVs, "
                      f"hAcc={hacc:.1f}m", file=sys.stderr)
                last_print = elapsed

            if fix_type >= min_fix:
                fix_time = elapsed
                lat = parsed.lat
                lon = parsed.lon
                hacc = getattr(parsed, 'hAcc', 0) / 1000.0
                print(f"\n  TTFF: {fix_time:.1f}s ({fix_str}, {num_sv} SVs, "
                      f"{lat:.7f}deg N {lon:.7f}deg E, hAcc={hacc:.1f}m)", file=sys.stderr)
                return

            break  # got NAV-PVT, go to next poll cycle

        # Small delay between polls
        time.sleep(0.8)

    print(f"\n  No fix after {timeout:.0f}s", file=sys.stderr)
    sys.exit(1)


# ---- Subcommand: monitor ----

def cmd_monitor(ser, args):
    """Monitor receiver output in real-time."""
    duration = args.duration
    ubx_only = args.ubx_only

    ensure_ubx_output(ser, args.baud)

    print("Monitoring receiver output (Ctrl+C to stop)...", file=sys.stderr)
    if duration:
        print(f"  Duration: {duration}s", file=sys.stderr)
    if ubx_only:
        print("  Showing UBX messages only (use without --ubx-only to see NMEA too)",
              file=sys.stderr)

    start_time = time.monotonic()
    msg_count = 0

    # Use longer serial timeout to avoid exiting between 1Hz NMEA bursts
    old_timeout = ser.timeout
    ser.timeout = 2.0

    ubr = UBXReader(
        ser,
        protfilter=UBX_PROTOCOL | NMEA_PROTOCOL,
        quitonerror=ERR_LOG,
        msgmode=GET,
    )

    try:
        for _, parsed in ubr:
            if parsed is None:
                continue

            # Check duration limit
            if duration and (time.monotonic() - start_time) > duration:
                break

            # Skip NMEA if --ubx-only
            if ubx_only and not isinstance(parsed, UBXMessage):
                continue

            ts = datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]

            if isinstance(parsed, UBXMessage):
                formatter = FORMATTERS.get(parsed.identity, format_generic)
                output = formatter(parsed)
                # Indent multi-line output
                if '\n' in output:
                    print(f"[{ts}] {parsed.identity}:")
                    for line in output.split('\n'):
                        print(f"  {line}")
                else:
                    print(f"[{ts}] {output}")
            else:
                # NMEA
                print(f"[{ts}] {parsed}")

            msg_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        ser.timeout = old_timeout

    elapsed = time.monotonic() - start_time
    print(f"\n{msg_count} messages in {elapsed:.1f}s", file=sys.stderr)


# ---- Subcommand: enable-ubx ----

def cmd_enable_ubx(ser, args):
    """Enable UBX protocol on UART output (persistent until power cycle)."""
    ensure_ubx_output(ser, args.baud)
    print("UBX output protocol enabled.", file=sys.stderr)

    # Verify by polling CFG-PRT
    poll_msg = UBXMessage('CFG', 'CFG-PRT', POLL, portID=1)
    send_ubx(ser, poll_msg)

    for _, parsed in read_responses(ser, timeout=2.0):
        if parsed.identity == 'CFG-PRT':
            print(f"  Port config: {parsed}", file=sys.stderr)
            return

    print("  (could not verify - no CFG-PRT response)", file=sys.stderr)


# ---- Subcommand: parse ----

def cmd_parse(ser_unused, args):
    """Parse a UBX binary file and display contents (offline, no serial needed)."""
    input_file = args.input_file

    # Determine parse mode
    mode_map = {'get': GET, 'set': SET, 'auto': None}
    mode = mode_map.get(args.mode, None)

    with open(input_file, 'rb') as f:
        data = f.read()

    if mode is not None:
        modes_to_try = [mode]
    else:
        # Auto: try SET first (typical for generated files), then GET
        modes_to_try = [SET, GET]

    best_results = []
    for try_mode in modes_to_try:
        stream = io.BytesIO(data)
        ubr = UBXReader(stream, protfilter=UBX_PROTOCOL, quitonerror=ERR_LOG,
                        msgmode=try_mode)
        results = []
        unknown = 0
        for raw, parsed in ubr:
            if parsed:
                results.append((raw, parsed))
                if 'Unknown' in str(parsed):
                    unknown += 1
        # Pick the mode with more successfully parsed messages
        if len(results) > len(best_results) or (
            len(results) == len(best_results) and unknown < getattr(cmd_parse, '_best_unknown', 999)):
            best_results = results
            cmd_parse._best_unknown = unknown
            if unknown == 0:
                break  # Perfect parse, no need to try other mode

    for i, (raw, parsed) in enumerate(best_results):
        formatter = FORMATTERS.get(parsed.identity, format_generic)
        output = formatter(parsed)
        if '\n' in output:
            print(f"[{i}] {parsed.identity}:")
            for line in output.split('\n'):
                print(f"  {line}")
        else:
            print(f"[{i}] {output}")

    print(f"\n{len(best_results)} messages parsed from {input_file}", file=sys.stderr)


# ---- Subcommand: decode-eph (offline) ----

def cmd_decode_eph(ser_unused, args):
    """Decode AID-EPH file offline: extract ephemeris params, optionally export MGA-GPS-EPH."""
    input_file = args.input_file
    output_mga = args.mga

    with open(input_file, 'rb') as f:
        data = f.read()

    stream = io.BytesIO(data)
    ubr = UBXReader(stream, protfilter=UBX_PROTOCOL, quitonerror=ERR_LOG, msgmode=GET)

    satellites = {}
    no_eph_count = 0
    other_count = 0

    for raw, parsed in ubr:
        if parsed is None:
            continue
        if parsed.identity != 'AID-EPH':
            other_count += 1
            continue

        sv_id = parsed.svid
        payload_len = len(raw) - 8
        if payload_len <= 8:
            no_eph_count += 1
            continue

        def get_sf_word(name, _parsed=parsed):
            val = getattr(_parsed, f'{name}_01', None)
            if val is not None:
                return val
            val = getattr(_parsed, name, None)
            return val if val is not None else 0

        sf1 = [get_sf_word(f'sf1d{i}') for i in range(1, 9)]
        sf2 = [get_sf_word(f'sf2d{i}') for i in range(1, 9)]
        sf3 = [get_sf_word(f'sf3d{i}') for i in range(1, 9)]

        try:
            eph = decode_gps_subframes(sf1, sf2, sf3)
            satellites[sv_id] = eph
        except Exception as exc:
            print(f"  Warning: failed to decode G{sv_id:02d}: {exc}", file=sys.stderr)

    print(f"Decoded {len(satellites)} satellites with ephemeris "
          f"({no_eph_count} without, {other_count} non-AID-EPH skipped)", file=sys.stderr)

    if satellites:
        print()
        print(format_eph_table(satellites))

    if output_mga:
        mga_data = bytearray()
        for sv_id, eph in sorted(satellites.items()):
            msg = build_mga_gps_eph(sv_id, eph['_raw'])
            mga_data.extend(msg)
        with open(output_mga, 'wb') as f:
            f.write(mga_data)
        print(f"\nExported {len(satellites)} MGA-GPS-EPH messages to {output_mga}",
              file=sys.stderr)
        print(output_mga)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='UBX serial tool for u-blox receivers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  poll MSG       Poll a UBX message and display response
  dump-dbd       Dump MGA-DBD navigation database to file
  dump-eph       Poll AID-EPH, decode ephemeris, export as MGA-GPS-EPH
  send FILE      Send UBX binary file to receiver
  monitor        Monitor receiver output in real-time
  enable-ubx     Enable UBX output protocol on UART
  parse FILE     Parse and display a UBX binary file (offline)

Connection:
  By default, connects via serial port.  Use --tcp to connect through
  a TCP serial-mux server instead.  In TCP mode, commands that change
  UART settings (reset, enable-ubx) are not available.

Examples:
  %(prog)s /dev/ttyS2 poll NAV-SAT
  %(prog)s /dev/ttyS2 poll NAV-PVT
  %(prog)s /dev/ttyS2 dump-dbd backup.ubx
  %(prog)s /dev/ttyS2 dump-eph --mga receiver-eph.ubx
  %(prog)s /dev/ttyS2 send mga-data.ubx --verbose
  %(prog)s /dev/ttyS2 monitor --duration 30
  %(prog)s /dev/null parse captured.ubx
  %(prog)s --tcp localhost:4000 poll NAV-SAT
  %(prog)s --tcp localhost:4000 send mga-data.ubx --assist --lat 65 --lon 25
  %(prog)s --tcp 192.168.1.10:4000 dump-dbd backup.ubx
        """,
    )

    parser.add_argument('port',
                        help='Serial port (e.g. /dev/ttyS2) or host:port with --tcp')
    parser.add_argument('--tcp', action='store_true',
                        help='Connect via TCP instead of serial '
                             '(port argument becomes host:port)')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD,
                        help=f'Baud rate (default: {DEFAULT_BAUD})')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'Serial/TCP timeout in seconds (default: {DEFAULT_TIMEOUT})')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # poll
    p_poll = subparsers.add_parser('poll', help='Poll a UBX message')
    p_poll.add_argument('message', help=f'Message to poll ({", ".join(sorted(POLL_MESSAGES))})')

    # dump-dbd
    p_dump = subparsers.add_parser('dump-dbd', help='Dump MGA-DBD navigation database')
    p_dump.add_argument('output', nargs='?', help='Output filename (default: auto-generated)')

    # dump-eph
    p_eph = subparsers.add_parser('dump-eph',
        help='Poll AID-EPH, decode GPS ephemeris, optionally export as MGA-GPS-EPH')
    p_eph.add_argument('output', nargs='?', default=None,
        help='Output raw AID-EPH filename (default: auto-generated)')
    p_eph.add_argument('--mga', metavar='FILE',
        help='Also export decoded ephemeris as MGA-GPS-EPH binary file')
    p_eph.add_argument('--no-save', action='store_true',
        help='Do not save raw AID-EPH data (display only)')

    # send
    p_send = subparsers.add_parser('send', help='Send UBX file to receiver')
    p_send.add_argument('input_file', help='UBX binary file to send')
    p_send.add_argument('-v', '--verbose', action='store_true',
                        help='Show per-message status')
    p_send.add_argument('--debug', action='store_true',
                        help='Show raw hex for each message')
    p_send.add_argument('--assist', action='store_true',
                        help='Prepend MGA-INI-TIME_UTC and MGA-INI-POS_LLH '
                             '(required for ephemeris to work after cold start)')
    p_send.add_argument('--lat', type=float, default=None,
                        help='Latitude for --assist position (degrees)')
    p_send.add_argument('--lon', type=float, default=None,
                        help='Longitude for --assist position (degrees)')

    # reset
    p_rst = subparsers.add_parser('reset',
        help='Reset receiver navigation database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reset modes:
  cold   Clear all nav data (ephemeris, almanac, position, time, etc.)
  warm   Clear ephemeris only (keep almanac, position, time)
  hot    Keep everything (just restart GNSS tasks)
  eph    Same as warm (clear ephemeris only)
        """)
    p_rst.add_argument('mode', choices=['cold', 'warm', 'hot', 'eph'],
                       help='Reset mode')

    # ttff
    p_ttff = subparsers.add_parser('ttff',
        help='Measure Time To First Fix')
    p_ttff.add_argument('--timeout', type=float, default=120.0,
                        help='Maximum wait time in seconds (default: 120)')
    p_ttff.add_argument('--3d', dest='min_fix', action='store_const',
                        const=3, default=2,
                        help='Wait for 3D fix (default: 2D or better)')

    # monitor
    p_mon = subparsers.add_parser('monitor', help='Monitor receiver output')
    p_mon.add_argument('--duration', type=float, help='Duration in seconds (default: infinite)')
    p_mon.add_argument('--ubx-only', action='store_true', help='Only show UBX messages (hide NMEA)')

    # enable-ubx
    subparsers.add_parser('enable-ubx', help='Enable UBX output protocol')

    # parse (offline)
    p_parse = subparsers.add_parser('parse', help='Parse UBX file (offline, no serial)')
    p_parse.add_argument('input_file', help='UBX binary file to parse')
    p_parse.add_argument('--mode', choices=['get', 'set', 'auto'], default='auto',
                        help='Parse mode: get=receiver output, set=receiver input, '
                             'auto=try both (default: auto)')

    # decode-eph (offline)
    p_deceph = subparsers.add_parser('decode-eph',
        help='Decode AID-EPH file offline, optionally export as MGA-GPS-EPH')
    p_deceph.add_argument('input_file', help='Saved AID-EPH .ubx file')
    p_deceph.add_argument('--mga', metavar='FILE',
        help='Export decoded ephemeris as MGA-GPS-EPH binary file')

    args = parser.parse_args()

    # parse and decode-eph commands don't need serial
    if args.command == 'parse':
        cmd_parse(None, args)
        return
    if args.command == 'decode-eph':
        cmd_decode_eph(None, args)
        return

    # Commands that change UART config -- not allowed over TCP because
    # the serial-mux server is not UBX-aware and would not know to
    # adjust the baud rate on the physical port.
    TCP_BLOCKED_COMMANDS = {'reset'}

    if args.tcp:
        if args.command in TCP_BLOCKED_COMMANDS:
            print(f"Error: '{args.command}' is not available in TCP mode "
                  f"(it changes UART settings on the receiver)",
                  file=sys.stderr)
            sys.exit(1)

        # Parse host:port
        addr = args.port
        if ':' not in addr:
            print(f"Error: --tcp requires host:port (got '{addr}')",
                  file=sys.stderr)
            sys.exit(1)
        host, port_str = addr.rsplit(':', 1)
        try:
            tcp_port = int(port_str)
        except ValueError:
            print(f"Error: invalid port number '{port_str}'",
                  file=sys.stderr)
            sys.exit(1)

        ser = TcpStream(host, tcp_port, timeout=args.timeout)
        print(f"Connected to {host}:{tcp_port} (TCP)", file=sys.stderr)
    else:
        ser = open_serial(args.port, args.baud, args.timeout)

    try:
        commands = {
            'poll': cmd_poll,
            'dump-dbd': cmd_dump_dbd,
            'dump-eph': cmd_dump_eph,
            'send': cmd_send,
            'reset': cmd_reset,
            'ttff': cmd_ttff,
            'monitor': cmd_monitor,
            'enable-ubx': cmd_enable_ubx,
        }
        commands[args.command](ser, args)
    finally:
        ser.close()


if __name__ == '__main__':
    main()
