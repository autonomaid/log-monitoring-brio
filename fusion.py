import csv
import math
import multiprocessing as mp
import os
import queue
import random
import signal
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

# Library sensor diimpor dengan aman: kalau tidak terpasang, hanya
# sensor yang bersangkutan yang gagal, bukan seluruh program.
try:
    import usb.core
    import usb.util
except ImportError:
    usb = None

try:
    import can
except ImportError:
    can = None

try:
    import cv2
except ImportError:
    cv2 = None


# =============================================================
# PENGATURAN WAKTU / SINKRONISASI (SAMA UNTUK SEMUA SENSOR)
# =============================================================

# Interval tulis CSV semua sensor (detik)
TICK_INTERVAL = 0.1

# JEDA: data untuk tick ke-n ditulis BUFFER_DELAY detik setelah
# waktu tick-nya, supaya buffer sempat terisi dari semua sensor.
# Nilai ini dipakai SAMA oleh semua sensor (lewat jam bersama, lihat
# TickClock / make_clock); jangan membuat jeda tulis sendiri per sensor.
BUFFER_DELAY = 1.0

# Berapa lama data disimpan di buffer (detik). Harus > BUFFER_DELAY.
# Selisih (BUFFER_KEEP - BUFFER_DELAY) adalah toleransi penulisan yang
# tersendat: kalau penulisan disk macet lebih lama dari selisih ini,
# tick yang terlambat mendapat data yang kurang tepat (atau NO_DATA).
BUFFER_KEEP = BUFFER_DELAY + 3.0

# Sensor dianggap tidak mengirim data kalau data terdekat ke waktu tick
# lebih jauh dari ini (detik). Tick tetap ditulis (status NO_DATA).
STALE_TIMEOUT = 1.0

# Tick dianggap terlambat kalau menyimpang dari jadwal lebih dari ini
LATE_TICK_THRESHOLD_MS = 10.0

# Batas waktu menunggu semua sensor siap (detik)
READY_TIMEOUT = 20.0

# Tampilkan 1 baris konsol setiap N tick per sensor (1 = setiap tick)
PRINT_EVERY_N_TICKS = 1

# Jumlah sample moving average pada grafik writing speed
WRITING_SPEED_SMOOTHING = 10


# =============================================================
# PENGATURAN OUTPUT
# =============================================================

# Frekuensi output Xsens (Hz), dipakai saat konfigurasi sensor
IMU_FREQUENCY = 10
GNSS_FREQUENCY = 10

# Receiver GNSS MTi-G-710 hanya menghasilkan data maksimal 4 Hz
GNSS_PVT_FREQUENCY = 4

XSENS_FOLDER = "data_xsens"
RADAR_FOLDER = "data_radar"
ZED_FOLDER = os.path.join("dataset", "stereo")
PEAKCAN_FOLDER = "data_peakcan"


# =============================================================
# PENGATURAN RADAR MR76
# =============================================================

CAN_DEVICE = "/dev/ttyACM0"
BITRATE = 500000

CAN_ID_MEASUREMENT = 0x60A
CAN_ID_OBJECT = 0x60B


# =============================================================
# PENGATURAN PEAK CAN (CAN MOBIL, SocketCAN)
# =============================================================

# Antarmuka SocketCAN harus sudah aktif (ip link set can0 up type can
# bitrate ...); install_autostart.sh menaikkannya otomatis saat boot.
# Adaptor PCAN-USB terdeteksi sebagai can0.
# Bisa diganti lewat environment variable PEAKCAN_CHANNEL.
PEAKCAN_CHANNEL = os.environ.get("PEAKCAN_CHANNEL", "can0")

# Posisi berikut adalah indeks BYTE mulai dari 0, bukan posisi bit.
PEAKCAN_SIGNAL_BYTES = {
    0x1AB: (("setir_byte_1", 1), ("setir_byte_2", 2)),
    0x17C: (("gas_byte_0", 0), ("rem_byte_4", 4)),
    0x1D0: (
        ("roda_kiri_belakang_byte_4", 4),
        ("roda_kiri_belakang_byte_5", 5),
        ("roda_kanan_belakang_byte_6", 6),
        ("roda_kanan_belakang_byte_7", 7),
    ),
    0x294: (("sein_byte_0", 0),),
    0x16A: (("lampu_jarak_jauh_byte_0", 0),),
    0x1A6: (("hazard_byte_3", 3), ("hazard_byte_4", 4)),
}

# Mask 11 bit cocok tepat dengan ID standard; ID extended tidak diterima.
PEAKCAN_FILTERS = [
    {"can_id": can_id, "can_mask": 0x7FF, "extended": False}
    for can_id in PEAKCAN_SIGNAL_BYTES
]


# =============================================================
# PENGATURAN KAMERA ZED
# =============================================================

ZED_RESOLUTION = "HD720"          # VGA / HD720 / HD1080 / HD2K
ZED_FPS = 30

# Jarak capture (detik). Capture terjadi pada frame kamera, jadi
# terkuantisasi ke kelipatan 1/ZED_FPS. Tidak perlu mepet ke interval
# tick lagi karena tick memilih capture TERDEKAT dari buffer.
ZED_CAPTURE_INTERVAL = 0.060

# Kamera dijalankan dulu selama ini sebelum capture pertama
# (auto-exposure dll stabil). Ini hanya menunda ISI BUFFER ZED; t0
# baru ditetapkan setelah SEMUA sensor ready, jadi warm-up tidak
# menggeser ZED terhadap sensor lain.
ZED_WARMUP = 1.0

ZED_RESIZE_SCALE = 1.0            # 1.0 = ukuran asli
ZED_IMAGE_FORMAT = "jpg"          # "jpg" / "png"
ZED_JPEG_QUALITY = 65

# Preview hanya aktif kalau ada layar (DISPLAY). Saat jalan otomatis
# dari systemd (tanpa layar) preview mati supaya OpenCV tidak crash.
ZED_PREVIEW = bool(os.environ.get("DISPLAY"))   # tekan Q di jendela preview untuk berhenti
ZED_PREVIEW_MAX_WIDTH = 1280

ZED_SMOOTHING = 3                 # moving average grafik writing speed ZED


# =============================================================
# JAM TICK BERSAMA
# =============================================================

class TickClock:
    """
    Jam tick yang identik di semua proses.

    t0_mono  : waktu monotonic tick ke-0 (CLOCK_MONOTONIC bersifat
               system-wide di Linux, jadi sama di semua proses)
    wall0_ms : waktu dinding (epoch, milidetik) pada saat yang sama
    """

    def __init__(self, t0_mono, wall0_ms, interval, delay):

        self.t0 = float(t0_mono)
        self.wall0_ms = int(wall0_ms)
        self.interval = interval
        self.delay = delay
        self.interval_ms = int(round(interval * 1000.0))

    def target(self, n):
        """Waktu (monotonic) data yang diwakili tick ke-n."""
        return self.t0 + n * self.interval

    def write_time(self, n):
        """Waktu (monotonic) tick ke-n ditulis ke CSV."""
        return self.target(n) + self.delay

    def _tick_ms(self, n):
        # Bilangan bulat -> selisih antar tick SELALU tepat interval
        return self.wall0_ms + n * self.interval_ms

    def stamp(self, n):
        """Timestamp CSV: YYYY-MM-DD HH:MM:SS.mmm"""
        ms = self._tick_ms(n)
        return (
            datetime.fromtimestamp(ms // 1000).strftime("%Y-%m-%d %H:%M:%S")
            + f".{ms % 1000:03d}"
        )

    def file_stamp(self, n):
        """Untuk nama file: YYYYMMDD_HHMMSS_mmm"""
        ms = self._tick_ms(n)
        return (
            datetime.fromtimestamp(ms // 1000).strftime("%Y%m%d_%H%M%S")
            + f"_{ms % 1000:03d}"
        )


def make_clock(shared):

    # [0]=t0_mono, [1]=wall0_ms, [2]=interval, [3]=jeda tulis.
    # Semua diisi SEKALI oleh program utama, jadi setiap sensor
    # memakai jeda dan interval yang persis sama.
    return TickClock(
        shared["clock"][0],
        shared["clock"][1],
        shared["clock"][2],
        shared["clock"][3]
    )


# =============================================================
# BUFFER WAKTU
# =============================================================

class TimedBuffer:
    """
    Menyimpan (waktu_monotonic, data) selama BUFFER_KEEP detik.
    Data terbaru tidak pernah dibuang, jadi buffer tidak pernah
    kosong setelah data pertama masuk.
    """

    def __init__(self, keep_seconds=BUFFER_KEEP):

        self.keep = keep_seconds
        self.items = deque()
        self.lock = threading.Lock()
        self.total_added = 0

    def add(self, payload, mono=None):

        if mono is None:
            mono = time.monotonic()

        with self.lock:

            self.items.append((mono, payload))
            self.total_added += 1

            cutoff = mono - self.keep

            while len(self.items) > 1 and self.items[0][0] < cutoff:
                self.items.popleft()

    def nearest(self, target):
        """(mono, payload) yang waktunya paling dekat ke target, atau None."""

        with self.lock:

            if not self.items:
                return None

            return min(
                self.items,
                key=lambda it: abs(it[0] - target)
            )

    def __len__(self):

        with self.lock:
            return len(self.items)


def pick_sample(buffer, target):
    """
    Return (item, offset_s). item None = belum pernah ada data atau
    data terdekat lebih jauh dari STALE_TIMEOUT (sensor berhenti).
    """

    item = buffer.nearest(target)

    if item is None:
        return None, None

    offset = item[0] - target

    if abs(offset) > STALE_TIMEOUT:
        return None, offset

    return item, offset


# =============================================================
# CSV SINK + STATISTIK
# =============================================================

class CsvSink:

    def __init__(self, path, fieldnames):

        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        self.path = path

        self.file = open(path, "w", newline="", encoding="utf-8")

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fieldnames,
            restval=""
        )

        self.writer.writeheader()
        self.file.flush()

        self.start = time.perf_counter()
        self.times = []
        self.speeds = []

    def write_rows(self, rows):
        """Tulis semua baris satu tick, flush SEKALI. Ukur writing speed."""

        try:
            before = os.fstat(self.file.fileno()).st_size
        except Exception:
            before = 0

        t_start = time.perf_counter()

        self.writer.writerows(rows)
        self.file.flush()

        t_end = time.perf_counter()

        try:
            after = os.fstat(self.file.fileno()).st_size
        except Exception:
            after = before

        nbytes = after - before
        write_time = t_end - t_start

        if write_time > 0 and nbytes > 0:
            speed = nbytes / 1e6 / write_time
        else:
            speed = 0.0

        self.times.append(t_end - self.start)
        self.speeds.append(speed)

        return {
            "bytes": nbytes,
            "write_time": write_time,
            "speed": speed
        }

    def close(self):

        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


def new_stats():

    return {
        "ticks": 0,
        "NEW": 0,
        "REPEAT": 0,
        "NO_OBJECT": 0,
        "NO_DATA": 0,
        "ERROR": 0,
        "late_ticks": 0,
        "max_late_ms": 0.0,
        "max_offset_ms": 0.0
    }


def note_status(stats, status, offset):

    stats[status] += 1

    if offset is not None:

        stats["max_offset_ms"] = max(
            stats["max_offset_ms"],
            abs(offset) * 1000.0
        )


def run_tick_loop(name, shared, tick_fn, stats):
    """
    Loop tick sebuah sensor (dijalankan di thread penulis).

    Tick ke-n ditulis tepat pada clock.write_time(n). Tick yang
    terlambat tidak dilewati. Setelah stop, sisa tick sampai tick
    terakhir yang sama untuk semua sensor ditulis langsung.
    """

    go = shared["go"]
    stop = shared["stop"]

    while not go.wait(0.1):

        if stop.is_set():
            return

    clock = make_clock(shared)

    def do_tick(n, late_ms):

        try:
            tick_fn(n, clock, late_ms)

        except Exception as error:
            print(f"[{name}] ERROR tick {n}: {error!r}", flush=True)

        stats["ticks"] += 1

        if late_ms is not None:

            stats["max_late_ms"] = max(stats["max_late_ms"], late_ms)

            if late_ms > LATE_TICK_THRESHOLD_MS:
                stats["late_ticks"] += 1

    n = 0

    while not stop.is_set():

        due = clock.write_time(n)
        wait = due - time.monotonic()

        if wait > 0 and stop.wait(wait):
            break

        late_ms = max(0.0, (time.monotonic() - due) * 1000.0)

        do_tick(n, late_ms)

        n += 1

    # Tulis sisa tick sampai tick terakhir yang sama di semua sensor
    stop_time = shared["stop_mono"].value or time.monotonic()

    last_n = int((stop_time - clock.t0) / clock.interval + 1e-9)

    while n <= last_n:

        do_tick(n, None)

        n += 1


def request_stop(shared):
    """Minta semua proses berhenti. Waktu berhenti dicatat SEKALI."""

    with shared["stop_mono"].get_lock():

        if shared["stop_mono"].value == 0.0:
            shared["stop_mono"].value = time.monotonic()

    shared["stop"].set()


def process_common_init():

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Ctrl+C ditangani proses utama, yang lalu memberi tahu semua
    # proses lewat stop_event.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def print_summary(name, stats, sink):

    print()
    print("=" * 70)
    print(f"RINGKASAN {name}")
    print("=" * 70)
    print(f"Total tick           : {stats['ticks']}")
    print(f"Tick terlambat       : {stats['late_ticks']} "
          f"(> {LATE_TICK_THRESHOLD_MS:.0f} ms)")

    if sink is not None and sink.speeds:

        speeds = sink.speeds

        print(f"Writing speed         : rata2 {sum(speeds) / len(speeds):.3f} | "
              f"min {min(speeds):.3f} | maks {max(speeds):.3f} MB/s")

    print("=" * 70)


def save_speed_graph(path, title, times, speeds, smoothing):
    """Grafik writing speed (PNG, tanpa jendela supaya aman di Jetson)."""

    if not times:
        print(f"Tidak ada data writing speed untuk {title}.")
        return

    try:

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = np.asarray(times, dtype=float)
        speeds = np.asarray(speeds, dtype=float)

        window = min(smoothing, len(speeds))

        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(times, speeds, color="gray", alpha=0.4,
                linewidth=1, label="Raw")

        if window > 1:

            smooth = np.convolve(
                speeds,
                np.ones(window) / window,
                mode="valid"
            )

            ax.plot(times[window - 1:], smooth, color="blue",
                    linewidth=2.5, label=f"Moving average ({window})")

        ax.axhline(float(np.mean(speeds)), color="green", linestyle="--",
                   label=f"Rata-rata {np.mean(speeds):.3f} MB/s")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Elapsed Time (seconds)")
        ax.set_ylabel("Writing Speed (MB/s)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

        print(f"Grafik tersimpan: {os.path.abspath(path)}")

    except Exception as error:

        print(f"Gagal membuat grafik {title}: {error!r}")




# =============================================================
# === BAGIAN XSENS: konstanta USB, XBUS, parser, konfigurasi ===
# === (disalin apa adanya dari program Xsens)                ===
# =============================================================

VID = 0x2639
PID = 0x0017

USB_INTERFACE = 1

EP_IN = 0x83
EP_OUT = 0x02

USB_TIMEOUT_MS = 1000


# =============================================================
# XBUS MESSAGE IDS
# =============================================================

MID_GOTO_MEASUREMENT = 0x10
MID_GOTO_CONFIG = 0x30

MID_GOTO_MEASUREMENT_ACK = 0x11
MID_GOTO_CONFIG_ACK = 0x31

MID_MT_DATA2 = 0x36

MID_SET_OUTPUT_CONFIGURATION = 0xC0
MID_SET_OUTPUT_CONFIGURATION_ACK = 0xC1


# =============================================================
# XDI IDENTIFIERS
# =============================================================

XDI_QUATERNION = 0x2010
XDI_EULER_ANGLES = 0x2030

XDI_ACCELERATION = 0x4020
XDI_RATE_OF_TURN = 0x8020
XDI_MAGNETIC_FIELD = 0xC020

# -------------------------------------------------------------
# GNSS/PVT (Fp1632 fixed-point format, BUKAN float32/float64)
#
# Referensi: jiminghe/Xsens_MTi_Serial_Reader (SetOutput.py),
# konfigurasi GNSS/INS standar untuk MTi-G-710/680 memakai
# suffix format 0x2 (Fp1632) untuk LatLon/Altitude/Velocity,
# bukan kode dasar 0x5040/0x5020 seperti sebelumnya.
# -------------------------------------------------------------

XDI_LAT_LON_FP1632 = 0x5042
XDI_ALTITUDE_ELLIPSOID_FP1632 = 0x5022
XDI_VELOCITY_FP1632 = 0xD012

# -------------------------------------------------------------
# Field tambahan (direkomendasikan repo untuk sesi GNSS/INS,
# berguna untuk validasi urutan/continuity paket & fix GPS)
# -------------------------------------------------------------

XDI_PACKET_COUNTER = 0x1020
XDI_SAMPLE_TIME_FINE = 0x1060
XDI_UTC_TIME = 0x1010
XDI_STATUS_WORD = 0xE020

# GNSS PVT langsung dari receiver (satelit, fix, akurasi, UTC, kecepatan)
XDI_GNSS_PVT_DATA = 0x7010


# =============================================================
# CSV COLUMNS
# =============================================================

CSV_COLUMNS = [
    "timestamp",

    "acc_x",
    "acc_y",
    "acc_z",

    "gyro_x",
    "gyro_y",
    "gyro_z",

    "mag_x",
    "mag_y",
    "mag_z",

    "q0",
    "q1",
    "q2",
    "q3",

    "roll",
    "pitch",
    "yaw",

    "status_word",

    "selftest_ok",
    "filter_valid",
    "gnss_fix_flag",

    "gnss_valid",
    "gnss_fix",
    "gnss_num_sv",
    "gnss_h_acc",
    "gnss_v_acc",
    "gnss_speed",
    "gnss_lat",
    "gnss_lon",
    "gnss_alt"
]


# =============================================================
# USB DEVICE
# =============================================================

def find_xsens():

    print()
    print("Searching for Xsens...")
    print(
        f"VID:PID = {VID:04X}:{PID:04X}"
    )

    dev = usb.core.find(
        idVendor=VID,
        idProduct=PID
    )

    if dev is None:

        print()
        print("ERROR: Xsens tidak ditemukan.")
        print()
        print("Coba:")
        print("  lsusb")
        print()
        print(
            "Harus terlihat:"
        )
        print(
            "  ID 2639:0017 Xsens MTi-G 7xx GNSS/INS"
        )

        return None

    print()
    print("Xsens ditemukan!")

    return dev


# =============================================================
# PRINT USB INFORMATION
# =============================================================

def print_usb_info(dev):

    print()
    print("=" * 70)
    print("USB CONFIGURATION")
    print("=" * 70)

    try:

        cfg = dev.get_active_configuration()

    except usb.core.USBError:

        print(
            "Active configuration tidak tersedia."
        )

        print(
            "Mencoba set configuration..."
        )

        try:

            dev.set_configuration()

            cfg = dev.get_active_configuration()

        except usb.core.USBError as error:

            print(
                "ERROR mendapatkan USB configuration:"
            )

            print(error)

            raise

    for intf in cfg:

        print()
        print(
            f"Interface {intf.bInterfaceNumber}, "
            f"alternate {intf.bAlternateSetting}"
        )

        print(
            f"  Class    : {intf.bInterfaceClass}"
        )

        print(
            f"  SubClass : {intf.bInterfaceSubClass}"
        )

        print(
            f"  Protocol : {intf.bInterfaceProtocol}"
        )

        for ep in intf:

            direction = (
                "IN"
                if usb.util.endpoint_direction(
                    ep.bEndpointAddress
                ) == usb.util.ENDPOINT_IN
                else "OUT"
            )

            ep_type = usb.util.endpoint_type(
                ep.bmAttributes
            )

            print(
                f"    Endpoint "
                f"0x{ep.bEndpointAddress:02X} "
                f"{direction} "
                f"type={ep_type} "
                f"max_packet={ep.wMaxPacketSize}"
            )

    print()
    print("=" * 70)


# =============================================================
# CLAIM USB INTERFACE
# =============================================================

def prepare_usb(dev):

    print()
    print("Preparing USB interface...")

    try:

        dev.set_configuration()

    except usb.core.USBError as error:

        # Some devices are already configured.
        print(
            f"set_configuration(): {error}"
        )

    cfg = dev.get_active_configuration()

    intf = usb.util.find_descriptor(
        cfg,
        bInterfaceNumber=USB_INTERFACE
    )

    if intf is None:

        raise RuntimeError(
            f"USB interface {USB_INTERFACE} tidak ditemukan."
        )

    print(
        f"Interface {USB_INTERFACE} ditemukan."
    )

    # ---------------------------------------------------------
    # Detach kernel driver if necessary
    # ---------------------------------------------------------

    try:

        active = dev.is_kernel_driver_active(
            USB_INTERFACE
        )

    except (NotImplementedError, usb.core.USBError):

        active = False

    if active:

        print(
            f"Kernel driver aktif pada interface "
            f"{USB_INTERFACE}."
        )

        print(
            "Detach kernel driver..."
        )

        try:

            dev.detach_kernel_driver(
                USB_INTERFACE
            )

            print(
                "Kernel driver berhasil di-detach."
            )

        except usb.core.USBError as error:

            print(
                "WARNING: tidak dapat detach kernel driver:"
            )

            print(error)

    # ---------------------------------------------------------
    # Claim interface
    # ---------------------------------------------------------

    print(
        f"Claim interface {USB_INTERFACE}..."
    )

    usb.util.claim_interface(
        dev,
        USB_INTERFACE
    )

    print(
        "Interface berhasil di-claim."
    )


# =============================================================
# RELEASE USB
# =============================================================

def release_usb(dev):

    print()
    print("Releasing USB interface...")

    try:

        usb.util.release_interface(
            dev,
            USB_INTERFACE
        )

    except Exception as error:

        print(
            f"release_interface(): {error}"
        )

    try:

        usb.util.dispose_resources(
            dev
        )

    except Exception as error:

        print(
            f"dispose_resources(): {error}"
        )

    print(
        "USB resources released."
    )


# =============================================================
# XBUS CHECKSUM
# =============================================================

def calculate_checksum(data):

    return (
        -sum(data)
    ) & 0xFF


# =============================================================
# CREATE XBUS PACKET
# =============================================================

def create_packet(mid, payload=b""):

    payload_length = len(payload)

    packet = bytearray()

    # Preamble
    packet.append(0xFA)

    # Bus ID
    packet.append(0xFF)

    # Message ID
    packet.append(mid)

    # ---------------------------------------------------------
    # Normal length
    # ---------------------------------------------------------

    if payload_length < 0xFF:

        packet.append(
            payload_length
        )

    # ---------------------------------------------------------
    # Extended length
    # ---------------------------------------------------------

    else:

        packet.append(0xFF)

        packet.extend(
            struct.pack(
                ">H",
                payload_length
            )
        )

    packet.extend(payload)

    # Checksum covers everything after preamble.
    checksum = calculate_checksum(
        packet[1:]
    )

    packet.append(
        checksum
    )

    return bytes(packet)


# =============================================================
# SEND USB PACKET
# =============================================================

def usb_send(dev, packet):

    print(
        "TX:",
        packet.hex(" ")
    )

    try:

        written = dev.write(
            EP_OUT,
            packet,
            timeout=USB_TIMEOUT_MS
        )

    except usb.core.USBError as error:

        print()
        print("USB WRITE ERROR:")
        print(error)

        raise

    print(
        f"TX bytes: {written}"
    )

    return written


# =============================================================
# READ RAW USB DATA
# =============================================================

def usb_read(dev, size=64, timeout=100):

    try:

        data = dev.read(
            EP_IN,
            size,
            timeout=timeout
        )

        return bytes(data)

    except usb.core.USBTimeoutError:

        return b""

    except usb.core.USBError as error:

        print(
            f"USB READ ERROR: {error}"
        )

        return b""


# =============================================================
# XBUS STREAM PARSER
# =============================================================

class XBusParser:

    def __init__(self):

        self.buffer = bytearray()

    def feed(self, data):

        if data:

            self.buffer.extend(data)

        packets = []

        while True:

            packet = self.extract_packet()

            if packet is None:

                break

            packets.append(packet)

        return packets

    def extract_packet(self):

        # Need at least:
        #
        # FA FF MID LEN CHECKSUM
        #
        if len(self.buffer) < 5:

            return None

        # -----------------------------------------------------
        # Find preamble
        # -----------------------------------------------------

        try:

            preamble_index = (
                self.buffer.index(0xFA)
            )

        except ValueError:

            self.buffer.clear()

            return None

        if preamble_index > 0:

            del self.buffer[
                0:preamble_index
            ]

        if len(self.buffer) < 5:

            return None

        # -----------------------------------------------------
        # Check bus ID
        # -----------------------------------------------------

        if self.buffer[1] != 0xFF:

            del self.buffer[0]

            return None

        mid = self.buffer[2]

        length_byte = self.buffer[3]

        header_size = 4

        # -----------------------------------------------------
        # Extended length
        # -----------------------------------------------------

        if length_byte == 0xFF:

            if len(self.buffer) < 6:

                return None

            payload_length = struct.unpack(
                ">H",
                bytes(
                    self.buffer[4:6]
                )
            )[0]

            header_size = 6

        else:

            payload_length = length_byte

        # -----------------------------------------------------
        # Total packet length
        # -----------------------------------------------------

        total_length = (
            header_size
            + payload_length
            + 1
        )

        if len(self.buffer) < total_length:

            return None

        packet = bytes(
            self.buffer[
                0:total_length
            ]
        )

        # -----------------------------------------------------
        # Remove packet from buffer
        # -----------------------------------------------------

        del self.buffer[
            0:total_length
        ]

        # -----------------------------------------------------
        # Verify checksum
        # -----------------------------------------------------

        if (
            sum(packet[1:])
            & 0xFF
        ) != 0:

            print()
            print(
                "WARNING: XBus checksum error:"
            )

            print(
                packet.hex(" ")
            )

            return None

        payload_start = header_size

        payload_end = (
            payload_start
            + payload_length
        )

        payload = packet[
            payload_start:payload_end
        ]

        return mid, payload


# =============================================================
# WAIT FOR XBUS RESPONSE
# =============================================================

def wait_for_response(
    dev,
    expected_mid,
    timeout=3.0
):

    parser = XBusParser()

    start = time.monotonic()

    while (
        time.monotonic() - start
        < timeout
    ):

        data = usb_read(
            dev,
            size=64,
            timeout=200
        )

        if not data:

            continue

        packets = parser.feed(
            data
        )

        for mid, payload in packets:

            print(
                f"RX MID=0x{mid:02X} "
                f"LEN={len(payload)} "
                f"DATA={payload.hex(' ')}"
            )

            if mid == expected_mid:

                return payload

    return None


# =============================================================
# SEND COMMAND AND WAIT
# =============================================================

def send_command_and_wait(
    dev,
    mid,
    expected_mid,
    payload=b"",
    timeout=3.0
):

    packet = create_packet(
        mid,
        payload
    )

    usb_send(
        dev,
        packet
    )

    response = wait_for_response(
        dev,
        expected_mid,
        timeout
    )

    return response


# =============================================================
# CONFIGURE OUTPUT
# =============================================================

def create_output_configuration():

    configuration = bytearray()

    # ---------------------------------------------------------
    # PacketCounter + SampleTimeFine + UtcTime
    #
    # Direkomendasikan repo referensi untuk sesi GNSS/INS --
    # 0xFFFF = output rate maksimum (mengikuti rate internal
    # sensor), berguna untuk mendeteksi paket yang hilang/skip
    # lewat packet_counter (harus naik 1 tiap paket).
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(">HH", XDI_PACKET_COUNTER, 0xFFFF)
    )

    configuration.extend(
        struct.pack(">HH", XDI_SAMPLE_TIME_FINE, 0xFFFF)
    )

    configuration.extend(
        struct.pack(">HH", XDI_UTC_TIME, 0xFFFF)
    )

    # ---------------------------------------------------------
    # Acceleration
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_ACCELERATION,
            IMU_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # Rate of turn
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_RATE_OF_TURN,
            IMU_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # Magnetic field
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_MAGNETIC_FIELD,
            IMU_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # Quaternion
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_QUATERNION,
            IMU_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # Euler angles
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_EULER_ANGLES,
            IMU_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # StatusWord (berisi bit GNSS fix, dsb.)
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(">HH", XDI_STATUS_WORD, 0xFFFF)
    )

    # ---------------------------------------------------------
    # Latitude / longitude -- FORMAT Fp1632 (BUKAN float)
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_LAT_LON_FP1632,
            GNSS_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # Ellipsoid altitude -- FORMAT Fp1632 (BUKAN float)
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_ALTITUDE_ELLIPSOID_FP1632,
            GNSS_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # Velocity XYZ -- FORMAT Fp1632
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_VELOCITY_FP1632,
            GNSS_FREQUENCY
        )
    )

    # ---------------------------------------------------------
    # GNSS PVT (fix, jumlah satelit, akurasi, UTC satelit)
    # ---------------------------------------------------------

    configuration.extend(
        struct.pack(
            ">HH",
            XDI_GNSS_PVT_DATA,
            GNSS_PVT_FREQUENCY
        )
    )

    return bytes(configuration)


# =============================================================
# CONFIGURE XSENS
# =============================================================

def configure_sensor(dev):

    print()
    print("=" * 70)
    print("XSENS CONFIGURATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # GoToConfig
    # ---------------------------------------------------------

    print()
    print("1. GoToConfig")

    response = send_command_and_wait(
        dev,
        MID_GOTO_CONFIG,
        MID_GOTO_CONFIG_ACK,
        timeout=3.0
    )

    if response is not None:

        print(
            "GoToConfig ACK diterima."
        )

    else:

        print(
            "WARNING: GoToConfig ACK tidak diterima."
        )

    time.sleep(0.2)

    # ---------------------------------------------------------
    # SetOutputConfiguration
    # ---------------------------------------------------------

    configuration = create_output_configuration()

    print()
    print(
        "2. SetOutputConfiguration"
    )

    print(
        f"Acceleration : {IMU_FREQUENCY} Hz"
    )

    print(
        f"Rate of Turn : {IMU_FREQUENCY} Hz"
    )

    print(
        f"Magnetic     : {IMU_FREQUENCY} Hz"
    )

    print(
        f"Quaternion   : {IMU_FREQUENCY} Hz"
    )

    print(
        f"Euler        : {IMU_FREQUENCY} Hz"
    )

    print(
        f"Latitude/Lon : {GNSS_FREQUENCY} Hz"
    )

    print(
        f"Altitude     : {GNSS_FREQUENCY} Hz"
    )

    print(
        f"GNSS PVT     : {GNSS_PVT_FREQUENCY} Hz"
    )

    print()
    print(
        "Configuration payload:"
    )

    print(
        configuration.hex(" ")
    )

    response = send_command_and_wait(
        dev,
        MID_SET_OUTPUT_CONFIGURATION,
        MID_SET_OUTPUT_CONFIGURATION_ACK,
        configuration,
        timeout=3.0
    )

    if response is not None:

        print()
        print(
            "SetOutputConfiguration ACK diterima."
        )

    else:

        print()
        print(
            "WARNING: SetOutputConfiguration "
            "ACK tidak diterima."
        )

    time.sleep(0.2)

    # ---------------------------------------------------------
    # GoToMeasurement
    # ---------------------------------------------------------

    print()
    print(
        "3. GoToMeasurement"
    )

    response = send_command_and_wait(
        dev,
        MID_GOTO_MEASUREMENT,
        MID_GOTO_MEASUREMENT_ACK,
        timeout=3.0
    )

    if response is not None:

        print(
            "GoToMeasurement ACK diterima."
        )

    else:

        print(
            "WARNING: GoToMeasurement "
            "ACK tidak diterima."
        )

    time.sleep(0.5)

    print()
    print(
        "XSens sekarang berada pada Measurement State."
    )


# =============================================================
# FLOAT32 HELPER
# =============================================================

def unpack_float32_triplet(value):

    if len(value) != 12:

        return None

    return struct.unpack(
        ">fff",
        value
    )


# =============================================================
# Fp1632 FIXED-POINT HELPER
#
# Format khusus Xsens untuk data presisi tinggi (LatLon,
# Altitude, Velocity): 6 byte per nilai = 4 byte pecahan
# (int32 signed) + 2 byte bagian bulat (int16 signed),
# digabung jadi fixed-point 48-bit lalu dibagi 2^32.
#
# Diadaptasi dari DataPacketParser.get_data_fp1632() di
# jiminghe/Xsens_MTi_Serial_Reader (repo referensi).
# =============================================================

def unpack_fp1632(payload, offset):

    if offset + 6 > len(payload):

        return None

    frac = struct.unpack_from(
        ">i",
        payload,
        offset
    )[0]

    whole = struct.unpack_from(
        ">h",
        payload,
        offset + 4
    )[0]

    fixed_point_64 = (
        (whole << 32)
        | (frac & 0xFFFFFFFF)
    )

    return fixed_point_64 / 4294967296.0


# =============================================================
# PARSE GNSS PVT (XDI_GnssPvtData 0x7010)
#
# Layout (big-endian, 94 byte; 80 byte pertama dipakai):
#   iTOW U4, year U2, month U1, day U1, hour U1, min U1, sec U1,
#   valid U1 (bit0 tanggal, bit1 jam, bit2 fully resolved),
#   tAcc U4, nano I4, fixType U1 (0 tanpa fix, 2 = 2D, 3 = 3D),
#   flags U1 (bit0 = fix valid), numSV U1, reserved U1,
#   lon I4 (1e-7 deg), lat I4 (1e-7 deg), height I4 (mm),
#   hMSL I4 (mm), hAcc U4 (mm), vAcc U4 (mm),
#   velN, velE, velD, gSpeed I4 (mm/s), headMot I4, sAcc U4,
#   headAcc U4, headVeh I4, ...
# =============================================================

def parse_gnss_pvt(value):

    (
        itow,
        year,
        month,
        day,
        hour,
        minute,
        sec,
        valid_flags,
        t_acc,
        nano,
        fix_type,
        fix_flags,
        num_sv,
        _reserved,
        lon,
        lat,
        height,
        h_msl,
        h_acc,
        v_acc,
        vel_n,
        vel_e,
        vel_d,
        g_speed,
        head_mot,
        s_acc,
        head_acc,
        head_veh
    ) = struct.unpack(
        ">IHBBBBBBIiBBBBiiiiIIiiiiiIIi",
        value[:80]
    )

    # Tanggal/jam dianggap valid hanya kalau bit0 dan bit1 aktif.
    # Tanpa itu receiver menampilkan tanggal bawaan (mis. 2015).
    utc = ""

    if (valid_flags & 0x03) == 0x03:

        utc = (
            f"{year:04d}-{month:02d}-{day:02d} "
            f"{hour:02d}:{minute:02d}:{sec:02d}"
        )

    return {
        "gnss_fix": fix_type,
        "gnss_num_sv": num_sv,
        "gnss_h_acc": h_acc / 1000.0,
        "gnss_v_acc": v_acc / 1000.0,
        "gnss_speed": g_speed / 1000.0,
        "gnss_lat": lat * 1e-7,
        "gnss_lon": lon * 1e-7,
        "gnss_alt": h_msl / 1000.0,
        "gnss_utc": utc
    }


# =============================================================
# PARSE MTData2
# =============================================================

def parse_mtdata2(payload):

    data = {}

    index = 0

    while index + 3 <= len(payload):

        try:

            data_id = struct.unpack(
                ">H",
                payload[
                    index:index + 2
                ]
            )[0]

            index += 2

            data_length = payload[index]

            index += 1

            if (
                index + data_length
                > len(payload)
            ):

                print(
                    "WARNING: MTData2 item length "
                    "melewati payload."
                )

                break

            value = payload[
                index:
                index + data_length
            ]

            index += data_length

        except Exception:

            break

        # -----------------------------------------------------
        # Acceleration
        # -----------------------------------------------------

        if data_id == XDI_ACCELERATION:

            result = unpack_float32_triplet(
                value
            )

            if result is not None:

                (
                    data["acc_x"],
                    data["acc_y"],
                    data["acc_z"]
                ) = result

        # -----------------------------------------------------
        # Rate of Turn / Gyroscope
        # -----------------------------------------------------

        elif data_id == XDI_RATE_OF_TURN:

            result = unpack_float32_triplet(
                value
            )

            if result is not None:

                (
                    data["gyro_x"],
                    data["gyro_y"],
                    data["gyro_z"]
                ) = result

        # -----------------------------------------------------
        # Magnetometer
        # -----------------------------------------------------

        elif data_id == XDI_MAGNETIC_FIELD:

            result = unpack_float32_triplet(
                value
            )

            if result is not None:

                (
                    data["mag_x"],
                    data["mag_y"],
                    data["mag_z"]
                ) = result

        # -----------------------------------------------------
        # Quaternion
        # -----------------------------------------------------

        elif data_id == XDI_QUATERNION:

            if len(value) == 16:

                (
                    data["q0"],
                    data["q1"],
                    data["q2"],
                    data["q3"]
                ) = struct.unpack(
                    ">ffff",
                    value
                )

        # -----------------------------------------------------
        # Euler angles
        # -----------------------------------------------------

        elif data_id == XDI_EULER_ANGLES:

            result = unpack_float32_triplet(
                value
            )

            if result is not None:

                (
                    data["roll"],
                    data["pitch"],
                    data["yaw"]
                ) = result

        # -----------------------------------------------------
        # Latitude + Longitude -- FORMAT Fp1632
        #
        # 12 byte = 2 x 6 byte fixed-point (BUKAN float64/
        # float32 seperti asumsi sebelumnya). Ini XDI yang
        # benar-benar diminta sekarang lewat
        # create_output_configuration() (0x5042).
        # -----------------------------------------------------

        elif data_id == XDI_LAT_LON_FP1632:

            if len(value) == 12:

                data["latitude"] = unpack_fp1632(value, 0)
                data["longitude"] = unpack_fp1632(value, 6)

        # -----------------------------------------------------
        # Ellipsoid altitude -- FORMAT Fp1632 (6 byte)
        # -----------------------------------------------------

        elif data_id == XDI_ALTITUDE_ELLIPSOID_FP1632:

            if len(value) == 6:

                data["altitude"] = unpack_fp1632(value, 0)

        # -----------------------------------------------------
        # Velocity XYZ -- FORMAT Fp1632 (3 x 6 byte = 18 byte)
        # -----------------------------------------------------

        elif data_id == XDI_VELOCITY_FP1632:

            if len(value) == 18:

                data["vel_x"] = unpack_fp1632(value, 0)
                data["vel_y"] = unpack_fp1632(value, 6)
                data["vel_z"] = unpack_fp1632(value, 12)

        # -----------------------------------------------------
        # PacketCounter -- untuk deteksi paket yang hilang
        # (harus naik 1 tiap paket; kalau ada lompatan berarti
        # ada paket yang skip, mirip kasus gap di zed_sensor.py)
        # -----------------------------------------------------

        elif data_id == XDI_PACKET_COUNTER:

            if len(value) == 2:

                data["packet_counter"] = struct.unpack(
                    ">H",
                    value
                )[0]

        # -----------------------------------------------------
        # SampleTimeFine
        # -----------------------------------------------------

        elif data_id == XDI_SAMPLE_TIME_FINE:

            if len(value) == 4:

                data["sample_time_fine"] = struct.unpack(
                    ">I",
                    value
                )[0]

        # -----------------------------------------------------
        # UtcTime
        # -----------------------------------------------------

        elif data_id == XDI_UTC_TIME:

            if len(value) == 12:

                nano = struct.unpack(">I", value[0:4])[0]
                year = struct.unpack(">H", value[4:6])[0]
                month = value[6]
                day = value[7]
                hour = value[8]
                minute = value[9]
                second = value[10]

                data["utc_time"] = (
                    f"{year:04d}-{month:02d}-{day:02d} "
                    f"{hour:02d}:{minute:02d}:{second:02d}."
                    f"{nano // 1_000_000:03d}"
                )

        # -----------------------------------------------------
        # StatusWord (berisi bit indikator GNSS fix, dll.)
        # -----------------------------------------------------

        elif data_id == XDI_STATUS_WORD:

            data["status_word"] = int.from_bytes(
                value,
                byteorder="big"
            )

            # Bit 0 = self-test, bit 1 = filter valid,
            # bit 2 = GNSS fix (MT Low Level Communication).
            status = data["status_word"]

            data["selftest_ok"] = status & 0x1
            data["filter_valid"] = (status >> 1) & 0x1
            data["gnss_fix_flag"] = (status >> 2) & 0x1

        # -----------------------------------------------------
        # GNSS PVT (dari receiver)
        # -----------------------------------------------------

        elif data_id == XDI_GNSS_PVT_DATA:

            if len(value) >= 80:

                data.update(
                    parse_gnss_pvt(
                        value
                    )
                )

    return data


# =============================================================
# === BAGIAN RADAR: decode object (disalin apa adanya)       ===
# =============================================================

# ============================================================
# OBJECT CLASS
# ============================================================

OBJECT_CLASS = {
    0: "POINT",
    1: "VEHICLE",
    2: "RESERVED_2",
    3: "RESERVED_3"
}


# ============================================================
# DYNAMIC PROPERTY
# ============================================================

DYNAMIC_PROPERTY = {
    0: "MOVING",
    1: "STATIONARY",
    2: "ONCOMING",
    3: "CROSSING_LEFT",
    4: "CROSSING_RIGHT",
    5: "UNKNOWN",
    6: "STOPPED",
    7: "UNKNOWN"
}


# ============================================================
# DECODE OBJECT CLASS
# ============================================================

def decode_object_class(raw_class):

    return OBJECT_CLASS.get(
        raw_class,
        "UNKNOWN"
    )


# ============================================================
# DECODE DYNAMIC PROPERTY
# ============================================================

def decode_dynamic_property(raw_dynamic):

    return DYNAMIC_PROPERTY.get(
        raw_dynamic,
        "UNKNOWN"
    )


# ============================================================
# GET TIMESTAMP
#
# Format:
# YYYY-MM-DD HH:MM:SS.mmm
# ============================================================

def get_timestamp():

    now = datetime.now()

    milliseconds = now.microsecond // 1000

    return (
        now.strftime("%Y-%m-%d %H:%M:%S")
        +
        f".{milliseconds:03d}"
    )


# ============================================================
# DECODE OBJECT 0x60B
# ============================================================

def decode_object(
    data,
    measurement_number=0
):

    if len(data) != 8:

        return None


    rxBuf = data


    # ========================================================
    # TIMESTAMP KOMPUTER
    # ========================================================

    timestamp = get_timestamp()


    # ========================================================
    # RAW BIT EXTRACTION
    #
    # RAW hanya digunakan INTERNAL.
    # RAW tidak dimasukkan CSV.
    # ========================================================

    raw_object_id = rxBuf[0]


    raw_distance_long = (
        (
            (rxBuf[1] << 8)
            |
            (rxBuf[2] & 0xF8)
        )
        >> 3
    )


    raw_distance_lat = (
        (
            (rxBuf[2] & 0x07)
            << 8
        )
        |
        rxBuf[3]
    )


    raw_velocity_long = (
        (
            (rxBuf[4] << 8)
            |
            (rxBuf[5] & 0xC0)
        )
        >> 6
    )


    raw_velocity_lat = (
        (
            (
                (rxBuf[5] & 0x3F)
                << 8
            )
            |
            (rxBuf[6] & 0xE0)
        )
        >> 5
    )


    raw_object_class = (
        (rxBuf[6] & 0x18)
        >> 3
    )


    raw_dynamic = (
        rxBuf[6] & 0x07
    )


    raw_rcs = rxBuf[7]


    # ========================================================
    # CONVERT RAW -> PHYSICAL
    # ========================================================

    object_id = raw_object_id + 1


    distance_long = (
        raw_distance_long * 0.2
        -
        500.0
    )


    distance_lat = (
        raw_distance_lat * 0.2
        -
        204.6
    )


    velocity_long = (
        raw_velocity_long * 0.25
        -
        128.0
    )


    velocity_lat = (
        raw_velocity_lat * 0.25
        -
        64.0
    )


    object_class = decode_object_class(
        raw_object_class
    )


    dynamic_property = decode_dynamic_property(
        raw_dynamic
    )


    rcs = (
        raw_rcs * 0.5
        -
        64.0
    )


    # ========================================================
    # TOTAL DISTANCE
    # ========================================================

    distance_total = math.sqrt(
        distance_long ** 2
        +
        distance_lat ** 2
    )


    # ========================================================
    # ANGLE
    # ========================================================

    angle = math.degrees(
        math.atan2(
            distance_lat,
            distance_long
        )
    )


    # ========================================================
    # TOTAL VELOCITY
    # ========================================================

    velocity_total = math.sqrt(
        velocity_long ** 2
        +
        velocity_lat ** 2
    )


    # ========================================================
    # MOVEMENT STATUS
    # ========================================================

    if velocity_long < -0.5:

        movement_status = "APPROACHING"

    elif velocity_long > 0.5:

        movement_status = "MOVING_AWAY"

    else:

        movement_status = "RELATIVELY_STATIC"


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "timestamp":
            timestamp,

        "measurement_number":
            measurement_number,

        "object_id":
            object_id,

        "distance_long":
            distance_long,

        "distance_lat":
            distance_lat,

        "distance_total":
            distance_total,

        "angle":
            angle,

        "velocity_long":
            velocity_long,

        "velocity_lat":
            velocity_lat,

        "velocity_total":
            velocity_total,

        "movement_status":
            movement_status,

        "object_class":
            object_class,

        "dynamic_property":
            dynamic_property,

        "rcs":
            rcs
    }


# ============================================================
# BUILD ROWS UNTUK SATU TICK
#
# Semua object dalam satu tick memakai timestamp tick yang sama.
# ============================================================

def build_tick_rows(
    tick_timestamp,
    delta_time_ms,
    tick_number,
    status,
    age_ms,
    snapshot
):

    common = {

        "timestamp":
            tick_timestamp
    }


    if (
        snapshot is not None
        and status != "NO_DATA"
    ):

        objects = snapshot["objects"]

    else:

        objects = []


    # Tidak ada object / tidak ada data -> satu baris penanda
    if not objects:

        row = dict(common)

        row["object_count"] = 0

        return [row]


    rows = []

    for obj in objects:

        row = dict(common)

        row.update({

            "object_count":
                len(objects),

            "object_id":
                obj["object_id"],

            "distance_long_m":
                f"{obj['distance_long']:.2f}",

            "distance_lat_m":
                f"{obj['distance_lat']:.2f}",

            "distance_total_m":
                f"{obj['distance_total']:.2f}",

            "angle_degree":
                f"{obj['angle']:.2f}",

            "velocity_long_mps":
                f"{obj['velocity_long']:.2f}",

            "velocity_lat_mps":
                f"{obj['velocity_lat']:.2f}",

            "velocity_total_mps":
                f"{obj['velocity_total']:.2f}",

            "movement_status":
                obj["movement_status"],

            "object_class":
                obj["object_class"],

            "dynamic_property":
                obj["dynamic_property"]
        })

        rows.append(row)


    return rows


# =============================================================
# === BAGIAN ZED: pembuat stereo (disalin apa adanya)        ===
# =============================================================

# =========================================================
# MEMBUAT STEREO
# =========================================================

def make_stereo(
    left_full,
    right_full,
    scale
):

    if scale >= 0.9999:

        return cv2.hconcat([
            left_full,
            right_full
        ])


    height, width = (
        left_full.shape[:2]
    )


    new_w = max(
        2,
        int(
            round(
                width
                * scale
                / 2
            )
        )
        * 2
    )


    new_h = max(
        2,
        int(
            round(
                height
                * scale
                / 2
            )
        )
        * 2
    )


    left_small = cv2.resize(
        left_full,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


    right_small = cv2.resize(
        right_full,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


    return cv2.hconcat([
        left_small,
        right_small
    ])


# =============================================================
# =============================================================
# PROSES SENSOR
# =============================================================
# =============================================================

import types


# -------------------------------------------------------------
# XSENS
# -------------------------------------------------------------

# =============================================================
# GNSS HOLD
#
# PVT hanya datang 4 Hz, sedangkan CSV ditulis 10 baris/detik.
# Nilai GNSS terakhir ditahan dan diisikan ke tiap baris, dengan
# gnss_age_ms sebagai umur data (ms sejak PVT terakhir).
#
# Kalau gnss_fix < GNSS_MIN_FIX (atau PVT belum pernah datang),
# gnss_valid = 0 dan field posisi/akurasi/kecepatan dikosongkan
# supaya nilai palsu (0,0 / akurasi raksasa) tidak masuk dataset.
# =============================================================

GNSS_MIN_FIX = 2

GNSS_HOLD_KEYS = [
    "gnss_fix",
    "gnss_num_sv",
    "gnss_h_acc",
    "gnss_v_acc",
    "gnss_speed",
    "gnss_lat",
    "gnss_lon",
    "gnss_alt",
    "gnss_utc"
]

GNSS_FIX_DEPENDENT_KEYS = [
    "latitude",
    "longitude",
    "altitude",
    "gnss_h_acc",
    "gnss_v_acc",
    "gnss_speed",
    "gnss_lat",
    "gnss_lon",
    "gnss_alt"
]

PER_PACKET_KEYS = {
    "packet_counter",
    "sample_time_fine",
    "utc_time",
    "status_word",
    "selftest_ok",
    "filter_valid",
    "gnss_fix_flag"
}


class GnssHold:

    def __init__(self):

        self.held = {}

        self.last_pvt_time = None

    def update(self, sensor):

        now = time.monotonic()

        got_pvt = "gnss_fix" in sensor

        if got_pvt:

            for key in GNSS_HOLD_KEYS:

                if key in sensor:

                    self.held[key] = sensor[key]

            self.last_pvt_time = now

            # Paket yang HANYA berisi PVT (tanpa data IMU/posisi
            # lain) tidak dijadikan baris CSV; nilainya cukup
            # ditahan untuk baris berikutnya.

            other_keys = (
                set(sensor)
                - PER_PACKET_KEYS
                - set(GNSS_HOLD_KEYS)
            )

            if not other_keys:

                return None

        # Isi nilai GNSS dari nilai terakhir yang ditahan.

        for key in GNSS_HOLD_KEYS:

            if key in self.held:

                sensor[key] = self.held[key]

        if self.last_pvt_time is not None:

            sensor["gnss_age_ms"] = int(
                (now - self.last_pvt_time) * 1000
            )

        fix = sensor.get("gnss_fix")

        valid = (
            1
            if isinstance(fix, int)
            and fix >= GNSS_MIN_FIX
            else 0
        )

        sensor["gnss_valid"] = valid

        if not valid:

            for key in GNSS_FIX_DEPENDENT_KEYS:

                sensor.pop(key, None)

        return sensor


def format_xsens_console(sensor):
    """
    Bagian konsol Xsens: v (kecepatan GNSS, satu nilai), gyro berlabel
    x/y/z, dan status GPS. Hanya tampilan; tidak mempengaruhi isi CSV.
    """

    speed = sensor.get("gnss_speed")

    speed_str = (
        f"{speed:.2f}m/s"
        if speed is not None
        else "N/A"
    )

    gyro_parts = []

    for axis in ("x", "y", "z"):

        value = sensor.get(f"gyro_{axis}")

        gyro_parts.append(
            f"{axis}={value: .4f}"
            if value is not None
            else f"{axis}=  N/A"
        )

    # Posisi: utamakan nilai langsung dari receiver (PVT),
    # cadangan: nilai hasil filter sensor.
    latitude = sensor.get("gnss_lat")
    longitude = sensor.get("gnss_lon")
    altitude = sensor.get("gnss_alt")

    if latitude is None:
        latitude = sensor.get("latitude")

    if longitude is None:
        longitude = sensor.get("longitude")

    if altitude is None:
        altitude = sensor.get("altitude")

    latitude_str = (
        f"{latitude:.7f}" if latitude is not None else "N/A"
    )

    longitude_str = (
        f"{longitude:.7f}" if longitude is not None else "N/A"
    )

    altitude_str = (
        f"{altitude:.3f}" if altitude is not None else "N/A"
    )

    fix = sensor.get("gnss_fix")
    num_sv = sensor.get("gnss_num_sv")

    fix_str = str(fix) if fix is not None else "-"
    sat_str = f"{num_sv:2d}" if num_sv is not None else " -"

    return (
        f"v={speed_str} | "
        f"GYR: {', '.join(gyro_parts)} | "
        f"GPS: fix={fix_str} sat={sat_str} "
        f"lat={latitude_str}, lon={longitude_str}, alt={altitude_str}"
    )


def create_csv_row(timestamp, sensor):
    """Satu baris CSV Xsens; kolom yang tidak ada di sensor dikosongkan."""

    row = {}

    for column in CSV_COLUMNS:

        if column == "timestamp":
            row[column] = timestamp
        else:
            row[column] = sensor.get(column, "")

    return row


def xsens_usb_reader(dev, buffer, ready, stop):

    parser = XBusParser()

    gnss_hold = GnssHold()

    last_data = time.monotonic()

    while not stop.is_set():

        raw = usb_read(dev, size=64, timeout=200)

        if raw:

            for mid, payload in parser.feed(raw):

                if mid != MID_MT_DATA2:
                    continue

                sensor = parse_mtdata2(payload)

                last_data = time.monotonic()

                # Tahan nilai GNSS terakhir + tandai validitas fix.
                # Paket yang HANYA berisi PVT tidak masuk buffer
                # (nilainya ditahan untuk paket IMU berikutnya).
                sensor = gnss_hold.update(sensor)

                if sensor is None:
                    continue

                buffer.add(sensor)

                ready.set()

        else:

            silence = time.monotonic() - last_data

            if silence > 5.0:

                print(f"[XSENS ] WARNING: tidak menerima MTData2 "
                      f"selama {silence:.1f} detik.")

                last_data = time.monotonic()


def xsens_sim_reader(buffer, ready, stop):

    rng = random.Random(1)

    counter = 0
    next_t = time.monotonic()

    while not stop.is_set():

        counter += 1

        sensor = {
            column: round(rng.uniform(-1.0, 1.0), 4)
            for column in CSV_COLUMNS
            if column != "timestamp"
        }

        sensor["packet_counter"] = counter % 65536
        sensor["sample_time_fine"] = counter * 1000
        sensor["utc_time"] = "sim"
        sensor["latitude"] = -6.9175 + rng.uniform(-1e-4, 1e-4)
        sensor["longitude"] = 107.6191 + rng.uniform(-1e-4, 1e-4)
        sensor["altitude"] = 768.0 + rng.uniform(-0.5, 0.5)
        sensor["status_word"] = 71
        sensor["selftest_ok"] = 1
        sensor["filter_valid"] = 1
        sensor["gnss_fix_flag"] = 1
        sensor["gnss_valid"] = 1
        sensor["gnss_fix"] = 3
        sensor["gnss_num_sv"] = 8
        sensor["gnss_h_acc"] = round(rng.uniform(1.0, 2.0), 3)
        sensor["gnss_v_acc"] = round(rng.uniform(1.5, 3.0), 3)
        sensor["gnss_speed"] = round(abs(rng.uniform(0.0, 0.2)), 3)
        sensor["gnss_lat"] = sensor["latitude"]
        sensor["gnss_lon"] = sensor["longitude"]
        sensor["gnss_alt"] = sensor["altitude"]
        sensor["gnss_utc"] = "sim"
        sensor["gnss_age_ms"] = int(rng.uniform(0, 250))

        buffer.add(sensor)
        ready.set()

        next_t += 0.1

        time.sleep(max(0.0, next_t - time.monotonic()
                       + rng.uniform(-0.005, 0.005)))


def xsens_process(shared, simulate):

    process_common_init()

    name = "XSENS"

    ready = shared["ready"][name]
    failed = shared["failed"][name]
    stop = shared["stop"]

    buffer = TimedBuffer()
    stats = new_stats()
    state = {"last": None}

    dev = None
    sink = None
    writer_thread = None

    try:

        # ---------------------------------------------------------
        # Buka perangkat
        # ---------------------------------------------------------

        if not simulate:

            if usb is None:
                raise RuntimeError("pyusb belum terpasang")

            dev = find_xsens()

            if dev is None:
                raise RuntimeError("Xsens tidak ditemukan")

            try:
                print_usb_info(dev)
            except Exception as error:
                print(f"[XSENS ] ERROR membaca USB configuration: {error}")

            prepare_usb(dev)

            configure_sensor(dev)

        sink = CsvSink(shared["paths"][name], CSV_COLUMNS)

        # ---------------------------------------------------------
        # Tick penulis CSV
        # ---------------------------------------------------------

        def tick(n, clock, late_ms):

            stamp = clock.stamp(n)

            item, offset = pick_sample(buffer, clock.target(n))

            if item is None:

                status = "NO_DATA"
                sensor = {}

            else:

                sensor = item[1]

                status = "REPEAT" if sensor is state["last"] else "NEW"

                state["last"] = sensor

            row = create_csv_row(stamp, sensor)

            result = sink.write_rows([row])

            note_status(stats, status, offset)

            if n % PRINT_EVERY_N_TICKS == 0:

                counter = sensor.get("packet_counter")

                print(
                    f"[XSENS ] {stamp} | tick {n:6d} | {status:7s} | "
                    f"pkt {counter if counter is not None else '-'} | "
                    f"write {result['write_time'] * 1000:.2f} ms | "
                    + format_xsens_console(sensor)
                )

        writer_thread = threading.Thread(
            target=run_tick_loop,
            args=(name, shared, tick, stats),
            daemon=False
        )

        writer_thread.start()

        ready_note = "simulasi" if simulate else "USB"
        print(f"[XSENS ] siap ({ready_note}), mengisi buffer...")

        # ---------------------------------------------------------
        # Baca sensor (thread utama proses ini)
        # ---------------------------------------------------------

        if simulate:
            xsens_sim_reader(buffer, ready, stop)
        else:
            xsens_usb_reader(dev, buffer, ready, stop)

    except Exception as error:

        print(f"[XSENS ] ERROR: {error!r}")

        if writer_thread is None:

            failed.value = 1

        else:

            # Pembaca mati di tengah jalan -> tick tetap ditulis
            # (NO_DATA) supaya timestamp CSV tetap seragam.
            stop.wait()

    finally:

        ready.set()

        if writer_thread is not None:
            writer_thread.join()

        if sink is not None:

            sink.close()

            print_summary(name, stats, sink)

            save_speed_graph(
                shared["paths"][name].replace(".csv", "_graph.png"),
                "Xsens MTi-G-710 - Writing Speed CSV",
                sink.times,
                sink.speeds,
                WRITING_SPEED_SMOOTHING
            )

        if dev is not None:
            release_usb(dev)


# -------------------------------------------------------------
# RADAR MR76
# -------------------------------------------------------------

RADAR_FIELDNAMES = [
    "timestamp",
    "object_count",
    "object_id",
    "distance_long_m",
    "distance_lat_m",
    "distance_total_m",
    "angle_degree",
    "velocity_long_mps",
    "velocity_lat_mps",
    "velocity_total_mps",
    "movement_status",
    "object_class",
    "dynamic_property"
]


class RadarCycleAssembler:
    """
    Radar mengirim satu siklus = 0x60A (header, berisi jumlah object)
    lalu 0x60B sebanyak jumlah object itu. Object dikumpulkan per
    siklus; begitu utuh, siklus masuk buffer sebagai SATU snapshot.
    Satu tick tidak pernah berisi campuran dua siklus.
    """

    def __init__(self, buffer, ready):

        self.buffer = buffer
        self.ready = ready

        self.building = {}
        self.expected = 0
        self.measurement = 0
        self.committed = True

        self.cycles_committed = 0
        self.cycles_incomplete = 0

    def _commit(self):

        objects = sorted(
            self.building.values(),
            key=lambda o: o["object_id"]
        )

        self.buffer.add({
            "objects": objects,
            "measurement_number": self.measurement
        })

        self.committed = True
        self.cycles_committed += 1

        self.ready.set()

    def start_cycle(self, measurement_number, object_count):

        # Siklus sebelumnya belum lengkap tapi header baru sudah datang
        if not self.committed:

            self.cycles_incomplete += 1
            self._commit()

        self.building = {}
        self.expected = object_count
        self.measurement = measurement_number
        self.committed = False

        if object_count == 0:
            self._commit()

    def add_object(self, obj):

        # Object tanpa header siklus (misal di awal program)
        if self.committed:
            return

        self.building[obj["object_id"]] = obj

        if len(self.building) >= self.expected:
            self._commit()


def radar_can_reader(bus, assembler, stop):

    measurement_number = 0

    while not stop.is_set():

        msg = bus.recv(timeout=0.5)

        if msg is None:
            continue

        if msg.arbitration_id == CAN_ID_MEASUREMENT:

            measurement_number += 1

            if len(msg.data) >= 1:
                assembler.start_cycle(measurement_number, msg.data[0])

        elif msg.arbitration_id == CAN_ID_OBJECT:

            if msg.dlc != 8:
                continue

            obj = decode_object(
                msg.data,
                measurement_number=measurement_number
            )

            if obj is not None:
                assembler.add_object(obj)


def radar_sim_reader(assembler, stop):

    rng = random.Random(2)

    measurement_number = 0
    next_t = time.monotonic()

    while not stop.is_set():

        measurement_number += 1

        count = rng.choice([0, 1, 1, 2, 2, 3])

        assembler.start_cycle(measurement_number, count)

        for i in range(count):

            data = bytes(
                [i * 3 + 1] + [rng.randrange(256) for _ in range(7)]
            )

            obj = decode_object(data, measurement_number=measurement_number)

            assembler.add_object(obj)

        next_t += 0.077

        time.sleep(max(0.0, next_t - time.monotonic()
                       + rng.uniform(-0.006, 0.006)))


def radar_process(shared, simulate):

    process_common_init()

    name = "RADAR"

    ready = shared["ready"][name]
    failed = shared["failed"][name]
    stop = shared["stop"]

    buffer = TimedBuffer()
    assembler = RadarCycleAssembler(buffer, ready)
    stats = new_stats()
    state = {"last": None}

    bus = None
    sink = None
    writer_thread = None

    try:

        if not simulate:

            if can is None:
                raise RuntimeError("python-can belum terpasang")

            bus = can.Bus(
                interface="slcan",
                channel=CAN_DEVICE,
                bitrate=BITRATE
            )

        sink = CsvSink(shared["paths"][name], RADAR_FIELDNAMES)

        def tick(n, clock, late_ms):

            stamp = clock.stamp(n)

            item, offset = pick_sample(buffer, clock.target(n))

            if item is None:

                snapshot = None
                status = "NO_DATA"

            else:

                snapshot = item[1]

                if not snapshot["objects"]:
                    status = "NO_OBJECT"
                elif snapshot is state["last"]:
                    status = "REPEAT"
                else:
                    status = "NEW"

                state["last"] = snapshot

            rows = build_tick_rows(
                stamp,
                None,
                n,
                status,
                None if offset is None else offset * 1000.0,
                snapshot
            )

            result = sink.write_rows(rows)

            note_status(stats, status, offset)

            if n % PRINT_EVERY_N_TICKS == 0:

                object_rows = (
                    0 if status in ("NO_OBJECT", "NO_DATA") else len(rows)
                )

                print(
                    f"[RADAR ] {stamp} | tick {n:6d} | {status:9s} | "
                    f"obj {object_rows:2d} | "
                    f"write {result['write_time'] * 1000:.2f} ms"
                )

        writer_thread = threading.Thread(
            target=run_tick_loop,
            args=(name, shared, tick, stats),
            daemon=False
        )

        writer_thread.start()

        print("[RADAR ] siap, mengisi buffer...")

        if simulate:
            radar_sim_reader(assembler, stop)
        else:
            radar_can_reader(bus, assembler, stop)

    except Exception as error:

        print(f"[RADAR ] ERROR: {error!r}")

        if writer_thread is None:
            failed.value = 1
        else:
            stop.wait()

    finally:

        ready.set()

        if writer_thread is not None:
            writer_thread.join()

        if sink is not None:

            sink.close()

            print_summary(name, stats, sink)

            save_speed_graph(
                shared["paths"][name].replace(".csv", "_graph.png"),
                "MR76 Radar - Writing Speed CSV",
                sink.times,
                sink.speeds,
                WRITING_SPEED_SMOOTHING
            )

        if bus is not None:

            try:
                bus.shutdown()
            except Exception:
                pass


# -------------------------------------------------------------
# PEAK CAN (CAN MOBIL)
# -------------------------------------------------------------

PEAKCAN_SIGNAL_FIELDS = [
    name
    for mapping in PEAKCAN_SIGNAL_BYTES.values()
    for name, _ in mapping
] + ["lampu_warna_warni_hex"]

PEAKCAN_FIELDNAMES = ["timestamp"] + PEAKCAN_SIGNAL_FIELDS


def peakcan_hex(value):
    """Satu byte menjadi teks heksa dua digit, mis. 163 -> 0xA3."""

    return f"0x{value:02X}"


def peakcan_decode(msg):
    """
    Nilai byte mentah dari satu frame, atau None kalau frame tidak
    dicatat (extended / error / remote / ID di luar daftar).
    """

    if msg.is_extended_id or msg.is_error_frame or msg.is_remote_frame:
        return None

    mapping = PEAKCAN_SIGNAL_BYTES.get(msg.arbitration_id)

    if mapping is None:
        return None

    data = bytes(msg.data)

    values = {
        name: (peakcan_hex(data[index]) if index < len(data) else "")
        for name, index in mapping
    }

    if msg.arbitration_id == 0x1A6:
        values["lampu_warna_warni_hex"] = data.hex(" ").upper()

    return values


class PeakCanState:
    """
    Frame datang per ID dengan laju berbeda-beda. Nilai terakhir tiap
    sinyal dipertahankan, dan setiap frame yang masuk menambah SATU
    snapshot (semua sinyal) ke buffer. Satu tick = satu snapshot
    terdekat ke waktu tick, sama seperti sensor lain.
    """

    def __init__(self, buffer, ready):

        self.buffer = buffer
        self.ready = ready

        self.values = {}
        self.frames = 0

    def update(self, signals):

        self.values.update(signals)

        self.buffer.add(dict(self.values))
        self.frames += 1

        self.ready.set()


def peakcan_reader(bus, state, stop):

    while not stop.is_set():

        msg = bus.recv(timeout=0.5)

        if msg is None:
            continue

        signals = peakcan_decode(msg)

        if signals is not None:
            state.update(signals)


def peakcan_sim_reader(state, stop):

    rng = random.Random(3)

    # periode kirim (detik) per ID, kira-kira seperti CAN mobil
    periods = {
        0x1AB: 0.010, 0x17C: 0.010, 0x1D0: 0.020,
        0x294: 0.100, 0x16A: 0.100, 0x1A6: 0.100
    }

    next_t = {can_id: time.monotonic() for can_id in periods}

    while not stop.is_set():

        now = time.monotonic()

        for can_id, period in periods.items():

            if now < next_t[can_id]:
                continue

            next_t[can_id] += period

            data = bytes(rng.randrange(256) for _ in range(8))

            mapping = PEAKCAN_SIGNAL_BYTES[can_id]

            values = {
                name: peakcan_hex(data[index]) for name, index in mapping
            }

            if can_id == 0x1A6:
                values["lampu_warna_warni_hex"] = data.hex(" ").upper()

            state.update(values)

        time.sleep(0.002)


def peakcan_process(shared, simulate):

    process_common_init()

    name = "PEAKCAN"

    ready = shared["ready"][name]
    failed = shared["failed"][name]
    stop = shared["stop"]

    buffer = TimedBuffer()
    state = PeakCanState(buffer, ready)
    stats = new_stats()
    last = {"snapshot": None}

    bus = None
    sink = None
    writer_thread = None

    try:

        if not simulate:

            if can is None:
                raise RuntimeError("python-can belum terpasang")

            # ignore_config: abaikan file/env konfigurasi python-can lain
            bus = can.Bus(
                interface="socketcan",
                channel=PEAKCAN_CHANNEL,
                ignore_config=True,
                receive_own_messages=False,
                can_filters=PEAKCAN_FILTERS
            )

        sink = CsvSink(shared["paths"][name], PEAKCAN_FIELDNAMES)

        def tick(n, clock, late_ms):

            stamp = clock.stamp(n)

            item, offset = pick_sample(buffer, clock.target(n))

            row = {"timestamp": stamp}

            if item is None:

                status = "NO_DATA"

            else:

                snapshot = item[1]

                status = (
                    "REPEAT" if snapshot is last["snapshot"] else "NEW"
                )

                last["snapshot"] = snapshot

                row.update(snapshot)

            result = sink.write_rows([row])

            note_status(stats, status, offset)

            if n % PRINT_EVERY_N_TICKS == 0:

                print(
                    f"[PEAKCAN] {stamp} | tick {n:6d} | {status:9s} | "
                    f"frame {state.frames:8d} | "
                    f"write {result['write_time'] * 1000:.2f} ms"
                )

        writer_thread = threading.Thread(
            target=run_tick_loop,
            args=(name, shared, tick, stats),
            daemon=False
        )

        writer_thread.start()

        print(f"[PEAKCAN] siap ({PEAKCAN_CHANNEL}), mengisi buffer...")

        if simulate:
            peakcan_sim_reader(state, stop)
        else:
            peakcan_reader(bus, state, stop)

    except Exception as error:

        print(f"[PEAKCAN] ERROR: {error!r}")

        if not simulate:
            print(f"[PEAKCAN] periksa antarmuka: "
                  f"ip -details link show {PEAKCAN_CHANNEL}")

        if writer_thread is None:
            failed.value = 1
        else:
            stop.wait()

    finally:

        ready.set()

        if writer_thread is not None:
            writer_thread.join()

        if sink is not None:

            sink.close()

            print_summary(name, stats, sink)

            save_speed_graph(
                shared["paths"][name].replace(".csv", "_graph.png"),
                "PEAK CAN - Writing Speed CSV",
                sink.times,
                sink.speeds,
                WRITING_SPEED_SMOOTHING
            )

        if bus is not None:

            try:
                bus.shutdown()
            except Exception:
                pass


# -------------------------------------------------------------
# KAMERA ZED
# -------------------------------------------------------------

def zed_encode(image):

    fmt = ZED_IMAGE_FORMAT.lower()

    if fmt == "jpg":

        return cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, int(ZED_JPEG_QUALITY)]
        )

    if fmt == "png":

        return cv2.imencode(".png", image)

    raise ValueError("ZED_IMAGE_FORMAT harus 'jpg' atau 'png'")


def zed_open():

    import pyzed.sl as sl

    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) belum terpasang")

    resolutions = {
        "VGA": sl.RESOLUTION.VGA,
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K
    }

    frame_period = 1.0 / ZED_FPS

    if ZED_CAPTURE_INTERVAL < frame_period:

        raise ValueError(
            f"ZED_CAPTURE_INTERVAL={ZED_CAPTURE_INTERVAL} lebih pendek dari "
            f"1 frame kamera ({frame_period:.4f} s) pada {ZED_FPS} FPS"
        )

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolutions[ZED_RESOLUTION]
    init_params.camera_fps = ZED_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_disable_self_calib = True

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Gagal membuka kamera ZED: {status}")

    return zed, sl


def zed_capture_loop(zed, sl, shared, buffer, ready, stop,
                     capture_gaps=None):

    image_left = sl.Mat()
    image_right = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    warmup_end = time.monotonic() + ZED_WARMUP
    next_capture = warmup_end

    image_number = 0
    last_capture_time = None
    last_preview = None
    preview_on = ZED_PREVIEW

    while not stop.is_set():

        if zed.grab(runtime_parameters) != sl.ERROR_CODE.SUCCESS:

            time.sleep(0.002)
            continue

        current = time.monotonic()

        # Capture dijadwalkan berdasarkan jam, bukan hitungan frame
        if current >= warmup_end and current >= next_capture - 1e-9:

            next_capture += ZED_CAPTURE_INTERVAL

            if current >= next_capture:
                next_capture = current + ZED_CAPTURE_INTERVAL

            image_number += 1

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            left_full = cv2.cvtColor(image_left.get_data(), cv2.COLOR_BGRA2BGR)
            right_full = cv2.cvtColor(image_right.get_data(), cv2.COLOR_BGRA2BGR)

            stereo = make_stereo(left_full, right_full, ZED_RESIZE_SCALE)

            success, encoded = zed_encode(stereo)

            if success:

                buffer.add({
                    "image_number": image_number,
                    "buffer": encoded,
                    "capture_width": stereo.shape[1],
                    "capture_height": stereo.shape[0],
                    "quality": ZED_JPEG_QUALITY,
                    "scale": ZED_RESIZE_SCALE
                }, mono=current)

                ready.set()

            else:

                print(f"[ZED   ] Gagal encode gambar No: {image_number:05d}")

            if capture_gaps is not None and last_capture_time is not None:

                capture_gaps.append((current - last_capture_time) * 1000.0)

            last_capture_time = current

            last_preview = stereo

            if stereo.shape[1] > ZED_PREVIEW_MAX_WIDTH:

                ratio = ZED_PREVIEW_MAX_WIDTH / stereo.shape[1]

                last_preview = cv2.resize(
                    stereo, None, fx=ratio, fy=ratio,
                    interpolation=cv2.INTER_AREA
                )

        if preview_on:

            try:

                if last_preview is not None:
                    cv2.imshow("ZED Stereo LEFT + RIGHT", last_preview)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):

                    print("[ZED   ] Capture dihentikan oleh user (Q).")

                    request_stop(shared)

                    break

            except Exception as error:

                print(f"[ZED   ] Preview dimatikan: {error}")

                preview_on = False


def zed_sim_reader(buffer, ready, stop, capture_gaps=None):

    rng = random.Random(3)

    fake_image = np.frombuffer(
        bytes(rng.getrandbits(8) for _ in range(350_000)),
        dtype=np.uint8
    )

    frame_period = 1.0 / ZED_FPS

    next_frame = time.monotonic()
    next_capture = next_frame

    image_number = 0
    last_capture_time = None

    while not stop.is_set():

        next_frame += frame_period

        time.sleep(max(0.0, next_frame - time.monotonic()))

        current = time.monotonic()

        if current >= next_capture - 1e-9:

            next_capture += ZED_CAPTURE_INTERVAL

            if current >= next_capture:
                next_capture = current + ZED_CAPTURE_INTERVAL

            image_number += 1

            buffer.add({
                "image_number": image_number,
                "buffer": fake_image,
                "capture_width": 2560,
                "capture_height": 720,
                "quality": ZED_JPEG_QUALITY,
                "scale": ZED_RESIZE_SCALE
            }, mono=current)

            ready.set()

            if capture_gaps is not None and last_capture_time is not None:

                capture_gaps.append((current - last_capture_time) * 1000.0)

            last_capture_time = current


def zed_process(shared, simulate):

    process_common_init()

    name = "ZED"

    ready = shared["ready"][name]
    failed = shared["failed"][name]
    stop = shared["stop"]

    buffer = TimedBuffer()
    stats = new_stats()
    state = {"last": None}

    image_speeds = types.SimpleNamespace(times=[], speeds=[])

    capture_gaps = []

    zed = None
    sl = None
    sink = None
    writer_thread = None

    try:

        if not simulate:
            zed, sl = zed_open()

        image_dir = shared["paths"]["ZED_DIR"]

        os.makedirs(image_dir, exist_ok=True)

        sink = CsvSink(
            shared["paths"][name],
            ["Timestamp", "Writing_Speed_MB_s", "JPEG_Quality", "Scale"]
        )

        speed_start = time.perf_counter()

        def tick(n, clock, late_ms):

            stamp = clock.stamp(n)

            status = "NO_DATA"
            offset = None
            speed = None
            size_mb = 0.0
            quality = ""
            scale = ""
            image_no = None

            try:

                item, offset = pick_sample(buffer, clock.target(n))

                # item None = tidak ada capture sama sekali / kamera
                # berhenti -> baris tetap ditulis tanpa gambar.
                if item is not None:

                    payload = item[1]

                    image_no = payload["image_number"]
                    quality = payload["quality"]
                    scale = f"{payload['scale']:.4f}"

                    status = "REPEAT" if payload is state["last"] else "NEW"

                    state["last"] = payload

                    filename = os.path.join(
                        image_dir,
                        f"{clock.file_stamp(n)}.{ZED_IMAGE_FORMAT.lower()}"
                    )

                    # Waktu mencakup open + write + close
                    write_start = time.perf_counter()

                    with open(filename, "wb") as image_file:
                        payload["buffer"].tofile(image_file)

                    write_time = time.perf_counter() - write_start

                    size_mb = os.path.getsize(filename) / 1_000_000

                    speed = size_mb / write_time if write_time > 0 else 0.0

                    image_speeds.times.append(time.perf_counter() - speed_start)
                    image_speeds.speeds.append(speed)

            except Exception as error:

                # Gambar gagal ditulis -> baris CSV TETAP dibuat di bawah
                status = "ERROR"
                speed = None

                print(f"[ZED   ] ERROR tick {n}: {error!r}", flush=True)

            # ---------------------------------------------------------
            # BARIS CSV SELALU DITULIS (satu baris per tick)
            # ---------------------------------------------------------

            sink.write_rows([{
                "Timestamp": stamp,
                "Writing_Speed_MB_s": "" if speed is None else f"{speed:.4f}",
                "JPEG_Quality": quality,
                "Scale": scale
            }])

            note_status(stats, status, offset)

            if n % PRINT_EVERY_N_TICKS == 0:

                print(
                    f"[ZED   ] {stamp} | tick {n:6d} | {status:7s} | "
                    + (f"img {image_no:05d}" if image_no is not None else "img -----")
                    + (f" | {size_mb:.2f} MB | {speed:.1f} MB/s"
                       if speed is not None else "")
                )

        writer_thread = threading.Thread(
            target=run_tick_loop,
            args=(name, shared, tick, stats),
            daemon=False
        )

        writer_thread.start()

        print(f"[ZED   ] siap ({ZED_RESOLUTION} {ZED_FPS} FPS), mengisi buffer...")

        if simulate:
            zed_sim_reader(buffer, ready, stop, capture_gaps)
        else:
            zed_capture_loop(zed, sl, shared, buffer, ready, stop,
                             capture_gaps)

    except Exception as error:

        print(f"[ZED   ] ERROR: {error!r}")

        if writer_thread is None:
            failed.value = 1
        else:
            stop.wait()

    finally:

        ready.set()

        # Pastikan main loop pembaca berhenti, lalu tunggu penulis
        # menghabiskan sisa tick sebelum kamera ditutup.
        if writer_thread is not None:
            writer_thread.join()

        if sink is not None:

            sink.close()

            print_summary(name, stats, image_speeds)

            save_speed_graph(
                shared["paths"][name].replace(".csv", "_graph.png"),
                "ZED Stereo Dataset - Kecepatan Tulis Per File",
                image_speeds.times,
                image_speeds.speeds,
                ZED_SMOOTHING
            )

        if zed is not None:

            try:
                zed.close()
            except Exception:
                pass

        if cv2 is not None and not simulate:

            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# =============================================================
# PEMERIKSAAN KESERAGAMAN TIMESTAMP CSV
# =============================================================

def read_timestamps(path):

    stamps = []

    with open(path, newline="", encoding="utf-8") as csv_file:

        reader = csv.reader(csv_file)

        next(reader, None)

        for row in reader:

            if not row or not row[0]:
                continue

            try:
                dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
                ms = int(round(dt.timestamp() * 1000.0))
            except ValueError:
                ms = None

            stamps.append((reader.line_num, ms, row[0]))

    return stamps


def verify_csv_uniformity(files):

    interval_ms = int(round(TICK_INTERVAL * 1000.0))

    print()
    print("=" * 70)
    print("PEMERIKSAAN TIMESTAMP CSV")
    print("=" * 70)

    all_ok = True

    series = {}

    for name, path in files.items():

        if not os.path.exists(path):

            print(f"{name}: file tidak ada, dilewati ({path})")
            continue

        unique = []

        for line, ms, raw in read_timestamps(path):

            if ms is None:

                print(f"  {path} baris {line}: timestamp tidak valid '{raw}'")
                all_ok = False
                continue

            # Radar menulis beberapa baris (object) per tick
            if unique and unique[-1][1] == ms:
                continue

            unique.append((line, ms))

        bad = 0

        for i in range(1, len(unique)):

            diff = unique[i][1] - unique[i - 1][1]

            if diff != interval_ms:

                bad += 1
                all_ok = False

                if bad <= 20:

                    print(f"  {path} baris {unique[i][0]}: selisih "
                          f"{diff} ms (harus {interval_ms} ms)")

        series[name] = unique

    names = list(series)

    for other in names[1:]:

        a = series[names[0]]
        b = series[other]

        if [ms for _, ms in a] == [ms for _, ms in b]:

            continue

        all_ok = False

        print(f"{names[0]} vs {other}: BERBEDA "
              f"({len(a)} vs {len(b)} tick)")

        for i in range(min(len(a), len(b))):

            if a[i][1] != b[i][1]:

                print(f"  pertama berbeda: {files[names[0]]} baris {a[i][0]}"
                      f" vs {files[other]} baris {b[i][0]}")

                break

    print("HASIL: SERAGAM" if all_ok else "HASIL: ADA YANG TIDAK SERAGAM")
    print("=" * 70)

    return all_ok


# =============================================================
# MAIN
# =============================================================

SENSOR_TARGETS = {
    "XSENS": xsens_process,
    "RADAR": radar_process,
    "ZED": zed_process,
    "PEAKCAN": peakcan_process
}


def parse_args(argv):

    options = {
        "simulate": False,
        "duration": None,
        "sensors": list(SENSOR_TARGETS)
    }

    i = 1

    while i < len(argv):

        arg = argv[i]

        if arg == "--simulate":

            options["simulate"] = True

            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                options["duration"] = float(argv[i + 1])
                i += 1

            elif options["duration"] is None:
                options["duration"] = 30.0

        elif arg == "--duration" and i + 1 < len(argv):

            options["duration"] = float(argv[i + 1])
            i += 1

        elif arg == "--sensors" and i + 1 < len(argv):

            options["sensors"] = [
                s.strip().upper() for s in argv[i + 1].split(",")
            ]

            i += 1

        else:

            print(f"Argumen tidak dikenal: {arg}")
            raise SystemExit(2)

        i += 1

    for sensor in options["sensors"]:

        if sensor not in SENSOR_TARGETS:

            print(f"Sensor tidak dikenal: {sensor} "
                  f"(pilihan: {', '.join(s.lower() for s in SENSOR_TARGETS)})")

            raise SystemExit(2)

    return options


def main():

    options = parse_args(sys.argv)

    simulate = options["simulate"]
    duration = options["duration"]
    names = options["sensors"]

    session = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = "sim_output" if simulate else "."

    zed_dir = os.path.join(root, ZED_FOLDER)

    paths = {
        "XSENS": os.path.join(root, XSENS_FOLDER,
                              f"xsens_mti_g710_{session}.csv"),
        "RADAR": os.path.join(root, RADAR_FOLDER,
                              f"mr76_radar_{session}.csv"),
        "ZED": os.path.join(zed_dir, f"writing_speed_{session}.csv"),
        "PEAKCAN": os.path.join(root, PEAKCAN_FOLDER,
                                f"peakcan_{session}.csv"),
        "ZED_DIR": zed_dir
    }

    ctx = mp.get_context("spawn")

    shared = {
        "ready": {n: ctx.Event() for n in names},
        "failed": {n: ctx.Value("b", 0) for n in names},
        "go": ctx.Event(),
        "stop": ctx.Event(),
        "stop_mono": ctx.Value("d", 0.0),
        "clock": ctx.Array("d", [0.0, 0.0, TICK_INTERVAL, BUFFER_DELAY]),
        "paths": paths
    }

    print()
    print("=" * 70)
    print("        SENSOR LOGGER TERPADU (XSENS + RADAR MR76 + ZED + PEAK CAN)")
    print("=" * 70)
    print(f"Mode            : {'SIMULASI' if simulate else 'SENSOR ASLI'}")
    print(f"Sensor          : {', '.join(names)}")
    print(f"Interval tick   : {TICK_INTERVAL * 1000:.0f} ms (semua sensor)")
    print(f"Jeda buffer     : {BUFFER_DELAY:.1f} detik (sama untuk semua sensor)")
    print(f"Durasi          : "
          f"{'sampai Ctrl+C / Q' if duration is None else f'{duration:.0f} detik'}")
    print("=" * 70)

    procs = {}

    for name in names:

        proc = ctx.Process(
            target=SENSOR_TARGETS[name],
            args=(shared, simulate),
            name=name
        )

        proc.start()

        procs[name] = proc

    stop = shared["stop"]

    try:

        # -----------------------------------------------------
        # Tunggu semua sensor siap (buffer sudah berisi data)
        # -----------------------------------------------------

        deadline = time.monotonic() + READY_TIMEOUT

        while time.monotonic() < deadline:

            if all(ev.is_set() for ev in shared["ready"].values()):
                break

            if not any(p.is_alive() for p in procs.values()):
                break

            time.sleep(0.05)

        active = [
            n for n in names
            if shared["failed"][n].value == 0 and procs[n].is_alive()
        ]

        for n in names:

            if n not in active:

                print(f"WARNING: {n} tidak aktif, dilewati.")

            elif not shared["ready"][n].is_set():

                print(f"WARNING: {n} belum mengirim data setelah "
                      f"{READY_TIMEOUT:.0f} detik; tick tetap ditulis "
                      f"(NO_DATA) sampai data masuk.")

        if not active:

            print("Tidak ada sensor yang aktif. Program berhenti.")

            request_stop(shared)

        else:

            # -------------------------------------------------
            # Tetapkan t0 SEKALI untuk semua sensor
            # -------------------------------------------------

            t0_mono = time.monotonic()
            wall0 = time.time()

            shared["clock"][0] = t0_mono
            shared["clock"][1] = int(wall0 * 1000)

            shared["go"].set()

            first = datetime.fromtimestamp(wall0).strftime("%H:%M:%S.%f")[:-3]

            print()
            print(f"Tick ke-0 (waktu data) : {first}")
            print(f"Penulisan CSV mulai    : {BUFFER_DELAY:.1f} detik lagi")
            print("Tekan Ctrl+C untuk berhenti"
                  + (" (atau Q di jendela ZED)." if "ZED" in names and ZED_PREVIEW else "."))
            print()

            while not stop.is_set():

                if (
                    duration is not None
                    and time.monotonic() - t0_mono >= duration
                ):

                    request_stop(shared)

                    break

                if not any(procs[n].is_alive() for n in active):

                    request_stop(shared)

                    break

                stop.wait(0.2)

    except KeyboardInterrupt:

        print()
        print("Ctrl+C diterima. Menghentikan semua sensor dengan aman...")

        request_stop(shared)

    # ---------------------------------------------------------
    # Tunggu semua proses menulis sisa tick lalu selesai
    # ---------------------------------------------------------

    for name, proc in procs.items():

        try:
            proc.join(timeout=60)
        except KeyboardInterrupt:
            request_stop(shared)
            proc.join(timeout=60)

        if proc.is_alive():

            print(f"WARNING: {name} tidak berhenti, dihentikan paksa.")
            proc.terminate()

    # ---------------------------------------------------------
    # Periksa keseragaman timestamp
    # ---------------------------------------------------------

    files = {
        n: paths[n] for n in names
        if shared["failed"][n].value == 0
    }

    if files:
        verify_csv_uniformity(files)

    print()
    print("=" * 70)
    print("PROGRAM SELESAI")
    print("=" * 70)

    for n, path in files.items():
        print(f"{n:6s}: {os.path.abspath(path)}")

    if "ZED" in files:
        print(f"Gambar : {os.path.abspath(zed_dir)}")

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
    #pengujian 10 menit telah dilakukan dan hasil konsisten setiap 0,1 detik 
    #untuk satelit saat pengujian dari jam 9 sampai jam 12 gagal sinyal satelit tidak terlihat namun saat dijalankan diatas jam 1 terlihat sinyalnya (ini didalam ruangan riset ai lantai 7)
