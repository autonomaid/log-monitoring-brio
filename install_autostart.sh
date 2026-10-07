#!/usr/bin/env bash
#
# Pasang fusion.py supaya jalan otomatis saat Jetson menyala (systemd).
#
#   sudo ./install_autostart.sh            pasang + aktifkan + jalankan
#   sudo ./install_autostart.sh remove     lepas autostart
#   ./install_autostart.sh --print         hanya tampilkan isi service
#
# Opsi (environment variable):
#   RUN_AS=root        jalankan sebagai user lain (default: user yang memanggil sudo)
#   PYTHON=/path/python3   interpreter Python (default: python3 milik user itu)
#   PEAKCAN_CHANNEL=can0   antarmuka CAN mobil (default: can0)
#   PEAKCAN_BITRATE=500000 bitrate CAN mobil; can0 dinaikkan otomatis saat boot
#
set -euo pipefail

SERVICE="fusion-logger"
UNIT="/etc/systemd/system/${SERVICE}.service"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${DIR}/fusion.py"
RUN_USER="${RUN_AS:-${SUDO_USER:-$(id -un)}}"
PEAKCAN_CHANNEL="${PEAKCAN_CHANNEL:-can0}"
PEAKCAN_BITRATE="${PEAKCAN_BITRATE:-500000}"

find_python() {
    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"
    elif [ "$(id -u)" -eq 0 ] && [ "$RUN_USER" != "root" ]; then
        runuser -l "$RUN_USER" -c 'command -v python3'
    else
        command -v python3
    fi
}

make_unit() {
    local py
    py="$(find_python)"
    cat <<UNIT
[Unit]
Description=Fusion sensor logger (Xsens + Radar MR76 + ZED)
After=multi-user.target time-sync.target
Wants=time-sync.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${DIR}
Environment=PYTHONUNBUFFERED=1
Environment=PEAKCAN_CHANNEL=${PEAKCAN_CHANNEL}
# Beri waktu USB (Xsens, radar, ZED) selesai terdeteksi setelah boot
ExecStartPre=/bin/sleep 15
# Naikkan CAN mobil (tanda + = dijalankan sebagai root walau User bukan root)
ExecStartPre=+/bin/sh -c '/sbin/ip link set ${PEAKCAN_CHANNEL} down; /sbin/ip link set ${PEAKCAN_CHANNEL} up type can bitrate ${PEAKCAN_BITRATE} restart-ms 100'
ExecStart=${py} "${SCRIPT}"
# systemctl stop = Ctrl+C: sisa tick ditulis dan CSV ditutup dengan benar
KillSignal=SIGINT
KillMode=mixed
TimeoutStopSec=150
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
}

if [ "${1:-}" = "--print" ]; then
    make_unit
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "Jalankan dengan sudo: sudo $0 ${1:-}" >&2
    exit 1
fi

if [ "${1:-}" = "remove" ]; then
    systemctl disable --now "$SERVICE" 2>/dev/null || true
    rm -f "$UNIT"
    systemctl daemon-reload
    echo "Autostart dilepas."
    exit 0
fi

if [ ! -f "$SCRIPT" ]; then
    echo "fusion.py tidak ditemukan di ${DIR}" >&2
    exit 1
fi

make_unit > "$UNIT"
systemctl daemon-reload
systemctl enable --now "$SERVICE"

echo
echo "Terpasang. Program jalan sekarang dan otomatis setiap Jetson menyala."
echo "  Log      : journalctl -u ${SERVICE} -f"
echo "  Status   : systemctl status ${SERVICE}"
echo "  Berhenti : sudo systemctl stop ${SERVICE}"
echo "  Data     : ${DIR}/data_xsens, data_radar, dataset/stereo"
