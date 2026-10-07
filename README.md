# Log Monitoring Brio

Logger terintegrasi berbasis Python untuk merekam data yang waktunya
tersinkron dari kendaraan dan sensor berikut:

- **Xsens MTi-G-710** melalui USB;
- **radar MR76** melalui adaptor serial-CAN (`/dev/ttyACM0`);
- **kamera ZED** sebagai gambar stereo gabungan kiri-kanan; dan
- **PCAN-USB / CAN kendaraan** melalui antarmuka SocketCAN (secara baku
  `can0`).

Setiap sensor ditulis pada jam tick bersama, secara baku setiap 100 ms.
Program memilih sampel sensor yang paling dekat dengan waktu tick dan
menyimpan data ke CSV; kamera ZED juga menyimpan gambar stereo. Grafik
kecepatan tulis CSV dibuat saat sesi berakhir.

## Isi proyek

| File/folder | Fungsi |
| --- | --- |
| `fusion.py` | Program utama untuk akuisisi, sinkronisasi, penyimpanan, dan simulasi. |
| `install_autostart.sh` | Memasang atau menghapus layanan systemd `fusion-logger`. |
| `data_xsens/` | CSV dan grafik hasil Xsens. |
| `data_radar/` | CSV dan grafik hasil radar MR76. |
| `data_peakcan/` | CSV dan grafik hasil CAN kendaraan. |
| `dataset/stereo/` | Gambar stereo ZED serta CSV kecepatan tulis ZED. |
| `sim_output/` | Keluaran ketika mode simulasi dipakai (dibuat saat diperlukan). |

## Perangkat dan sistem yang diperlukan

Gunakan Linux (ditujukan untuk NVIDIA Jetson) dengan Python 3. Siapkan
perangkat berikut sesuai sensor yang akan dijalankan:

- Xsens MTi-G-710 pada USB (VID:PID `2639:0017`);
- radar MR76 pada adaptor serial-CAN yang muncul sebagai `/dev/ttyACM0`;
- ZED Camera dan ZED SDK/Python API (`pyzed.sl`); dan/atau
- PCAN-USB yang telah tersedia sebagai antarmuka SocketCAN, misalnya `can0`.

Dependensi Python yang dipakai program:

```bash
python3 -m pip install numpy pyusb python-can opencv-python matplotlib
```

Instal ZED SDK beserta Python API dari Stereolabs sesuai versi Jetson dan
Python yang digunakan. Pastikan perangkat USB dapat diakses oleh pengguna
yang menjalankan logger; gunakan aturan `udev` atau jalankan layanan dengan
hak akses yang sesuai bila diperlukan.

Verifikasi dependensi sebelum memakai sensor asli:

```bash
python3 -c "import pyzed.sl, usb.core, can, cv2; print('semua ok')"
```

## Konfigurasi

Konstanta utama berada di bagian atas `fusion.py`:

- `TICK_INTERVAL = 0.1`: interval baris CSV, dalam detik.
- `BUFFER_DELAY = 1.0`: jeda sebelum tick ditulis agar buffer sensor terisi.
- `CAN_DEVICE = "/dev/ttyACM0"` dan `BITRATE = 500000`: koneksi radar MR76.
- `PEAKCAN_CHANNEL`: antarmuka CAN kendaraan; dapat dioverride dengan variabel
  lingkungan, misalnya `PEAKCAN_CHANNEL=can1`.
- Pengaturan ZED seperti resolusi, FPS, interval capture, format, dan kualitas
  JPEG: `ZED_RESOLUTION`, `ZED_FPS`, `ZED_CAPTURE_INTERVAL`,
  `ZED_IMAGE_FORMAT`, serta `ZED_JPEG_QUALITY`.

Nilai CAN dan pemetaan byte PCAN (`PEAKCAN_SIGNAL_BYTES`) harus disesuaikan
dengan jaringan kendaraan target sebelum pengambilan data produksi.

## Menjalankan logger

Masuk ke direktori proyek terlebih dahulu:

```bash
cd "/home/autonoma/Documents/logger otomatis sensor + PCAN"
```

### Aktifkan CAN kendaraan

Pastikan adaptor PCAN terdeteksi, kemudian aktifkan antarmuka dengan bitrate
yang sesuai jaringan kendaraan. Perintah berikut memakai mode *listen-only*
agar logger tidak mengirim frame CAN:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000 listen-only on
sudo ip link set can0 up
ip -details link show can0
```

### Uji sensor

Jalankan semua sensor selama 30 detik:

```bash
python3 fusion.py --duration 30
```

Jalankan hanya logger PCAN selama 20 detik:

```bash
PEAKCAN_CHANNEL=can0 python3 fusion.py --sensors peakcan --duration 20
```

Sensor yang tersedia untuk `--sensors` adalah `xsens`, `radar`, `zed`, dan
`peakcan`; gunakan daftar yang dipisahkan koma, misalnya:

```bash
python3 fusion.py --sensors xsens,radar --duration 60
```

Tanpa `--duration`, program berjalan sampai `Ctrl+C` ditekan. Saat preview
ZED aktif (ada variabel lingkungan `DISPLAY`), tombol `Q` pada jendela preview
juga menghentikan sesi.

### Mode simulasi

Mode simulasi tidak membutuhkan perangkat fisik dan menulis keluaran ke
`sim_output/`:

```bash
python3 fusion.py --simulate 10
```

Gunakan mode ini sebagai pemeriksaan awal instalasi dan struktur keluaran.

## Keluaran data

Nama file sesi berisi waktu mulai (`YYYYMMDD_HHMMSS`). Lokasi keluaran:

- `data_xsens/xsens_mti_g710_<sesi>.csv`
- `data_radar/mr76_radar_<sesi>.csv`
- `data_peakcan/peakcan_<sesi>.csv`
- `dataset/stereo/writing_speed_<sesi>.csv`
- `dataset/stereo/<timestamp>.jpg` (atau `.png` bila format diubah)

Setiap logger CSV juga menghasilkan grafik `<nama>_graph.png` yang memuat
kecepatan tulis. Saat selesai, program membandingkan timestamp antar CSV
aktif dan melaporkan apakah baris tick seragam.

Status yang dapat terlihat di konsol mencakup `NEW`, `REPEAT`, `NO_OBJECT`,
dan `NO_DATA`. `NO_DATA` berarti tidak ada sampel dalam batas waktu yang
diizinkan, tetapi baris tick tetap dipertahankan untuk menjaga keselarasan
waktu.

## Menjalankan otomatis saat boot

`install_autostart.sh` membuat layanan systemd bernama `fusion-logger`.
Skrip menunggu perangkat USB, mengaktifkan antarmuka CAN, lalu menjalankan
`fusion.py`. Jalankan dari direktori proyek:

```bash
# Instal, aktifkan saat boot, dan mulai sekarang
sudo bash install_autostart.sh

# Gunakan interpreter Python aktif, misalnya dari lingkungan conda
sudo PYTHON="$(which python3)" bash install_autostart.sh

# Atur antarmuka dan bitrate CAN saat boot
sudo PEAKCAN_CHANNEL=can0 PEAKCAN_BITRATE=500000 bash install_autostart.sh
```

Periksa layanan dan ikuti log:

```bash
systemctl status fusion-logger
journalctl -u fusion-logger -f
```

Untuk berhenti sementara:

```bash
sudo systemctl stop fusion-logger
```

Untuk menghapus autostart beserta unit service:

```bash
sudo bash install_autostart.sh remove
```

Untuk meninjau unit systemd tanpa menginstalnya:

```bash
./install_autostart.sh --print
```

## Troubleshooting

- **`python-can belum terpasang` atau `pyusb belum terpasang`**: instal
  dependensi Python dan ulangi pemeriksaan impor.
- **PCAN tidak menerima data**: periksa `ip -details link show can0`, bitrate,
  kabel, terminasi CAN, dan nama antarmuka. Gunakan
  `PEAKCAN_CHANNEL=<nama>` bila bukan `can0`.
- **Radar tidak terbaca**: pastikan `/dev/ttyACM0` adalah adaptor radar dan
  bitratenya sesuai (`500000` secara baku).
- **ZED gagal dibuka**: pastikan ZED SDK, `pyzed.sl`, kabel USB 3, dan izin
  perangkat tersedia. Preview akan dimatikan otomatis bila tidak ada display.
- **Sensor dicabut saat logger aktif**: hentikan layanan/program dengan aman,
  sambungkan kembali perangkat, lalu mulai ulang layanan. Apabila driver USB
  atau CAN tidak pulih, reboot Jetson.
- **Layanan gagal berjalan**: hapus unit, lalu instal ulang dengan interpreter
  yang benar:

  ```bash
  sudo bash install_autostart.sh remove
  sudo PYTHON="$(which python3)" bash install_autostart.sh
  ```

## Operasional aman

Sebelum menguji pada kendaraan, pastikan pemetaan frame CAN dan bitrate telah
divalidasi. Mode `listen-only` dianjurkan untuk aktivitas perekaman agar
logger tidak memengaruhi bus CAN kendaraan.
