"""
Session telemetry logger — writes bandwidth / latency / VAD / PTT stats to CSV.
"""

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


CSV_FIELDS = [
    "timestamp_iso",
    "elapsed_s",
    "role",
    "dh_state",
    "tx_packets",
    "tx_bytes",
    "tx_kbps",
    "rx_packets",
    "rx_ok",
    "rx_lost",
    "rx_tampered",
    "rx_kbps",
    "latency_ms",
    "jitter_ms",
    "underruns",
    "vad_enabled",
    "ptt_mode",
    "ptt_held",
    "transmitting",
    "is_speech",
    "energy_db",
    "zcr",
    "dtx_savings_pct",
]


class SessionCsvLogger:
    def __init__(self, path: Optional[str] = None, role: str = "station"):
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(logs_dir / f"session_{role}_{stamp}.csv")
        self.path = Path(path)
        self.role = role
        self._lock = threading.Lock()
        self._enabled = False
        self._start = None
        self._rows = 0
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def enable(self, enabled: bool = True):
        with self._lock:
            self._enabled = enabled
            if enabled and self._start is None:
                self._start = datetime.now()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def write(self, stats: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        now = datetime.now()
        with self._lock:
            start = self._start or now
            elapsed = (now - start).total_seconds()
            row = [
                now.isoformat(timespec="milliseconds"),
                f"{elapsed:.3f}",
                self.role,
                stats.get("dh_state", ""),
                stats.get("tx_packets", 0),
                stats.get("tx_bytes", 0),
                stats.get("tx_kbps", 0.0),
                stats.get("rx_packets", 0),
                stats.get("rx_ok", 0),
                stats.get("rx_lost", 0),
                stats.get("rx_tampered", 0),
                stats.get("rx_kbps", 0.0),
                stats.get("latency_ms", ""),
                stats.get("jitter_ms", 0.0),
                stats.get("underruns", 0),
                int(bool(stats.get("vad_enabled", False))),
                int(bool(stats.get("ptt_mode", False))),
                int(bool(stats.get("ptt_held", False))),
                int(bool(stats.get("transmitting", False))),
                int(bool(stats.get("is_speech", False))),
                stats.get("energy_db", ""),
                stats.get("zcr", ""),
                stats.get("dtx_savings_pct", 0.0),
            ]
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            self._rows += 1

    def row_count(self) -> int:
        return self._rows
