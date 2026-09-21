"""
Full-duplex encrypted call engine: mic → DSP → VAD/PTT → AES-GCM → UDP
and UDP → decrypt → jitter buffer → speakers, simultaneously.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from crypto_utils import AudioCrypto
from .dsp import (
    STANDARD_SAMPLE_RATE,
    DEFAULT_BLOCK_SIZE,
    MultiBandEqualizer,
    StandaloneFilter,
    VoiceActivityDetector,
)
from .generators import SignalGenerator, AudioFileReader, generate_synthetic_demo_track
from .network import JitterBuffer, UDPDuplexLink
from .stats_log import SessionCsvLogger


class SecureCallEngine:
    def __init__(
        self,
        local_port: int,
        peer_host: str,
        peer_port: int,
        crypto: AudioCrypto,
        role: str = "station",
        sample_rate: int = STANDARD_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        source_mode: str = "mic",
        enable_dh: bool = True,
        bind_ip: str = "0.0.0.0",
        csv_path: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.source_mode = source_mode
        self.role = role

        self.jitter_buffer = JitterBuffer(block_size=block_size, target_prebuffer_blocks=3, max_blocks=12)
        self.link = UDPDuplexLink(
            local_port=local_port,
            peer_host=peer_host,
            peer_port=peer_port,
            crypto=crypto,
            jitter_buffer=self.jitter_buffer,
            bind_ip=bind_ip,
            enable_dh=enable_dh,
        )

        self.equalizer = MultiBandEqualizer(sample_rate=sample_rate)
        self.filter = StandaloneFilter(sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(sample_rate=sample_rate, hangover_frames=5)
        self.vad_enabled = False
        self.ptt_mode = False
        self.ptt_held = False
        self.total_packets_evaluated = 0
        self.packets_suppressed = 0
        self.last_is_speech = False
        self.last_energy_db = -100.0
        self.last_zcr = 0.0
        self.last_transmitting = False

        self.signal_gen = SignalGenerator(sample_rate=sample_rate)
        self.file_reader = AudioFileReader(target_sample_rate=sample_rate)
        self.demo_reader = AudioFileReader(target_sample_rate=sample_rate)
        self._init_demo_track()

        self.stream: Optional[Any] = None
        self.is_running = False
        self.monitor_local_audio = False
        self.master_gain = 1.0
        self.mic_input_gain = 1.0
        self.rx_volume = 1.0
        self.viz_queue: queue.Queue = queue.Queue(maxsize=4)
        self._lock = threading.Lock()

        self.csv = SessionCsvLogger(path=csv_path, role=role)
        self._csv_thread: Optional[threading.Thread] = None
        self._last_viz: Dict[str, Any] = {}

    def _init_demo_track(self):
        demo_audio = generate_synthetic_demo_track(duration_sec=12.0, fs=self.sample_rate)
        self.demo_reader.audio_data = demo_audio
        self.demo_reader.file_sample_rate = self.sample_rate
        self.demo_reader.duration_seconds = len(demo_audio) / float(self.sample_rate)
        self.demo_reader.is_looping = True

    def set_vad_enabled(self, enabled: bool):
        with self._lock:
            self.vad_enabled = enabled
            if not enabled:
                self.vad.reset()

    def set_ptt_mode(self, enabled: bool):
        with self._lock:
            self.ptt_mode = enabled
            if not enabled:
                self.ptt_held = False

    def set_ptt_held(self, held: bool):
        self.ptt_held = bool(held)

    def set_filter_params(self, mode: str, cutoff: float, q: float = 1.0, order: int = 4):
        with self._lock:
            self.filter.set_params(mode=mode, cutoff=cutoff, q=q, order=order)

    def set_source(self, source: str):
        with self._lock:
            self.source_mode = source.lower()

    def set_master_gain(self, gain: float):
        self.master_gain = float(np.clip(gain, 0.0, 2.0))

    def set_mic_gain(self, gain: float):
        self.mic_input_gain = float(np.clip(gain, 0.0, 3.0))

    def set_rx_volume(self, vol: float):
        self.rx_volume = float(np.clip(vol, 0.0, 2.0))

    def set_csv_logging(self, enabled: bool):
        self.csv.enable(enabled)

    def start(self, input_device: Optional[int] = None, output_device: Optional[int] = None):
        if self.is_running:
            return
        self.link.start()
        self.is_running = True
        self.csv.enable(True)
        self._csv_thread = threading.Thread(target=self._csv_loop, name="CsvLogger", daemon=True)
        self._csv_thread.start()

        if sd is None:
            self._run_synthetic_thread()
            return
        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=(1, 1),
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception:
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=input_device,
                    channels=1,
                    dtype="float32",
                    callback=self._input_only_callback,
                )
                self.stream.start()
                self._run_rx_drain_thread()
            except Exception:
                self.stream = None
                self._run_synthetic_thread()

    def _run_synthetic_thread(self):
        def loop():
            interval = self.block_size / float(self.sample_rate)
            while self.is_running:
                start_t = time.time()
                in_chunk = np.zeros(self.block_size, dtype=np.float64)
                rx = self.jitter_buffer.pop(self.block_size)
                self._process_block(in_chunk, None, self.block_size, rx)
                elapsed = time.time() - start_t
                time.sleep(max(0.001, interval - elapsed))

        threading.Thread(target=loop, daemon=True).start()

    def _run_rx_drain_thread(self):
        def loop():
            interval = self.block_size / float(self.sample_rate)
            while self.is_running:
                start_t = time.time()
                self.jitter_buffer.pop(self.block_size)
                elapsed = time.time() - start_t
                time.sleep(max(0.001, interval - elapsed))

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.link.stop()

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        in_chunk = indata[:, 0].astype(np.float64) * self.mic_input_gain
        rx = self.jitter_buffer.pop(frames) * self.rx_volume
        self._process_block(in_chunk, outdata, frames, rx)

    def _input_only_callback(self, indata, frames, time_info, status):
        in_chunk = indata[:, 0].astype(np.float64) * self.mic_input_gain
        self._process_block(in_chunk, None, frames, None)

    def _process_block(
        self,
        mic_chunk: np.ndarray,
        outdata: Optional[np.ndarray],
        frames: int,
        rx_samples: Optional[np.ndarray],
    ):
        with self._lock:
            if self.source_mode == "mic":
                raw_input = mic_chunk
            elif self.source_mode == "generator":
                raw_input = self.signal_gen.generate_block(frames)
            elif self.source_mode == "file":
                raw_input = self.file_reader.read_block(frames)
            elif self.source_mode == "demo":
                raw_input = self.demo_reader.read_block(frames)
            else:
                raw_input = np.zeros(frames, dtype=np.float64)

            x_filt = self.filter.process(raw_input)
            x_eq = self.equalizer.process(x_filt)
            processed = np.tanh(x_eq * self.master_gain)

            is_speech, energy_db, zcr = self.vad.process_block(processed)
            self.total_packets_evaluated += 1

            should_transmit = True
            if self.ptt_mode and not self.ptt_held:
                should_transmit = False
            elif self.vad_enabled and not is_speech:
                should_transmit = False

            if not should_transmit:
                self.packets_suppressed += 1

            self.last_is_speech = is_speech
            self.last_energy_db = energy_db
            self.last_zcr = zcr
            self.last_transmitting = should_transmit

            if outdata is not None:
                out = np.zeros(frames, dtype=np.float32)
                if rx_samples is not None:
                    n = min(frames, len(rx_samples))
                    out[:n] = rx_samples[:n].astype(np.float32)
                if self.monitor_local_audio:
                    out += processed.astype(np.float32) * 0.35
                    out = np.clip(out, -1.0, 1.0)
                outdata[:, 0] = out

            if should_transmit:
                pcm_int16 = np.clip(processed * 32767.0, -32768.0, 32767.0).astype(np.int16)
                self.link.send_pcm_chunk(pcm_int16.tobytes())

            savings = (self.packets_suppressed / max(1, self.total_packets_evaluated)) * 100.0
            viz = {
                "input": raw_input.copy(),
                "output": processed.copy(),
                "rx": None if rx_samples is None else np.asarray(rx_samples).copy(),
                "is_speech": is_speech,
                "energy_db": energy_db,
                "zcr": zcr,
                "vad_enabled": self.vad_enabled,
                "ptt_mode": self.ptt_mode,
                "ptt_held": self.ptt_held,
                "transmitting": should_transmit,
                "savings_pct": savings,
                "timestamp": time.time(),
            }
            self._last_viz = viz
            if not self.viz_queue.full():
                try:
                    self.viz_queue.put_nowait(viz)
                except queue.Full:
                    pass

    def _csv_loop(self):
        while self.is_running:
            time.sleep(0.5)
            self.csv.write(self.collect_stats())

    def collect_stats(self) -> Dict[str, Any]:
        net = self.link.get_stats()
        viz = self._last_viz
        return {
            "dh_state": net.get("dh_state", ""),
            "tx_packets": net.get("packets_sent", 0),
            "tx_bytes": net.get("bytes_sent", 0),
            "tx_kbps": net.get("tx_kbps", 0.0),
            "rx_packets": net.get("packets_received", 0),
            "rx_ok": net.get("packets_ok", 0),
            "rx_lost": net.get("packets_lost", 0),
            "rx_tampered": net.get("packets_tampered", 0),
            "rx_kbps": net.get("rx_kbps", 0.0),
            "latency_ms": net.get("latency_ms", 0.0),
            "jitter_ms": net.get("jitter_ms", 0.0),
            "underruns": net.get("underruns", 0),
            "vad_enabled": self.vad_enabled,
            "ptt_mode": self.ptt_mode,
            "ptt_held": self.ptt_held,
            "transmitting": viz.get("transmitting", False),
            "is_speech": viz.get("is_speech", False),
            "energy_db": round(float(viz.get("energy_db", 0.0)), 2) if viz else "",
            "zcr": round(float(viz.get("zcr", 0.0)), 4) if viz else "",
            "dtx_savings_pct": round(float(viz.get("savings_pct", 0.0)), 2) if viz else 0.0,
        }

    def get_latest_visualization_data(self) -> Optional[Dict[str, Any]]:
        latest = None
        while not self.viz_queue.empty():
            try:
                latest = self.viz_queue.get_nowait()
            except queue.Empty:
                break
        return latest
