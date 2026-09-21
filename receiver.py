"""
Secure Audio Receiver (AES-256-GCM Decryption & Real-Time Playback).

Pipeline:
1. UDP Network Worker: Receives incoming UDP datagrams on a dedicated background thread.
2. Authenticated Decryption: AES-256-GCM verifies cryptographic tag and drops forged/corrupted packets.
3. Jitter Buffer: Elastic FIFO queue pre-buffers packets to absorb network delay variance.
4. Audio Playback: Real-time sounddevice output stream pulls decrypted PCM blocks and plays to speakers.
5. Real-Time Visualizer: Post-decryption waveform & FFT spectrum analyzer with live telemetry stats.
"""

import sys
import os
import time
import queue
import threading
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np

# Audio & DSP imports
try:
    import sounddevice as sd
except ImportError:
    sd = None

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from crypto_utils import AudioCrypto, load_key, generate_key, save_key, KEY_SIZE
from spequaliser.dsp import (
    STANDARD_SAMPLE_RATE,
    DEFAULT_BLOCK_SIZE,
    compute_spectrum
)
from spequaliser.network import JitterBuffer, UDPAudioReceiver


# UI Palette
COLOR_BG_DARK = "#090F1D"
COLOR_PANEL_BG = "#111A2E"
COLOR_PANEL_BORDER = "#1E2C48"
COLOR_TEXT_MAIN = "#F1F5F9"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_CYAN = "#00F2FE"
COLOR_BLUE = "#3B82F6"
COLOR_AMBER = "#F59E0B"
COLOR_GREEN = "#10B981"
COLOR_RED = "#EF4444"


class ReceiverEngine:
    """
    Core engine for receiving, decrypting, buffering, and playing audio.
    """
    def __init__(
        self,
        port: int,
        crypto: AudioCrypto,
        sample_rate: int = STANDARD_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        jitter_prebuffer_blocks: int = 3,
        bind_ip: str = "0.0.0.0"
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.volume = 1.0

        # Subsystems
        self.jitter_buffer = JitterBuffer(
            block_size=block_size,
            target_prebuffer_blocks=jitter_prebuffer_blocks,
            max_blocks=12
        )
        self.udp_receiver = UDPAudioReceiver(
            port=port,
            crypto=crypto,
            jitter_buffer=self.jitter_buffer,
            bind_ip=bind_ip
        )

        self.stream: Optional[Any] = None
        self.is_running = False
        self.viz_queue: queue.Queue = queue.Queue(maxsize=4)
        self._lock = threading.Lock()

    def set_volume(self, vol: float):
        self.volume = float(np.clip(vol, 0.0, 2.0))

    def start(self, output_device: Optional[int] = None):
        if self.is_running:
            return

        self.udp_receiver.start()
        self.is_running = True

        if sd is None:
            print("Warning: sounddevice not available. Running in headless drain mode.")
            self._run_drain_thread()
            return

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=output_device,
                channels=1,
                dtype='float32',
                callback=self._audio_playback_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Notice: Audio output device error ({e}). Draining jitter buffer internally.")
            self.stream = None
            self._run_drain_thread()

    def _run_drain_thread(self):
        """Fallback thread when speaker audio output cannot be initialized."""
        def drain():
            interval = self.block_size / float(self.sample_rate)
            while self.is_running:
                start_t = time.time()
                samples = self.jitter_buffer.pop(self.block_size)
                self._push_viz(samples)
                elapsed = time.time() - start_t
                sleep_t = max(0.001, interval - elapsed)
                time.sleep(sleep_t)
        t = threading.Thread(target=drain, daemon=True)
        t.start()

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.udp_receiver.stop()

    def _audio_playback_callback(self, outdata, frames, time_info, status):
        # Pull real-time decrypted audio from jitter buffer
        samples = self.jitter_buffer.pop(frames)
        scaled = samples * self.volume
        outdata[:, 0] = scaled

        # Push to visualizer
        self._push_viz(scaled)

    def _push_viz(self, samples: np.ndarray):
        if not self.viz_queue.full():
            try:
                self.viz_queue.put_nowait({
                    'samples': samples.copy(),
                    'timestamp': time.time()
                })
            except queue.Full:
                pass

    def get_latest_visualization_data(self) -> Optional[Dict[str, np.ndarray]]:
        latest = None
        while not self.viz_queue.empty():
            try:
                latest = self.viz_queue.get_nowait()
            except queue.Empty:
                break
        return latest


class ReceiverGUI:
    """
    Tkinter/Matplotlib GUI for the Secure Audio Receiver.
    """
    def __init__(self, root: tk.Tk, engine: ReceiverEngine):
        self.root = root
        self.engine = engine
        self.root.title("Secure Audio Receiver (AES-256-GCM + UDP) — SpEqualiser")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.configure(bg=COLOR_BG_DARK)

        self.vol_var = tk.DoubleVar(value=100.0)
        self.tamper_attack_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.engine.start()

        self.is_updating = True
        self.root.after(30, self._update_loop)

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=16, pady=10)
        header.pack(fill=tk.X)

        title_box = tk.Frame(header, bg=COLOR_BG_DARK)
        title_box.pack(side=tk.LEFT)

        lbl_title = tk.Label(
            title_box,
            text="SECURE AUDIO RECEIVER — AES-256-GCM AUTHENTICATED DECRYPTION",
            font=("Segoe UI", 13, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK
        )
        lbl_title.pack(anchor=tk.W)

        lbl_sub = tk.Label(
            title_box,
            text=f"Listening on UDP Port {self.engine.udp_receiver.port} • AES-GCM Integrity Verification • Jitter Buffer Audio Playback",
            font=("Segoe UI", 9),
            fg=COLOR_GREEN,
            bg=COLOR_BG_DARK
        )
        lbl_sub.pack(anchor=tk.W)

        # Right-side controls
        ctrl_box = tk.Frame(header, bg=COLOR_BG_DARK)
        ctrl_box.pack(side=tk.RIGHT)

        tk.Label(ctrl_box, text="Volume:", fg=COLOR_TEXT_MUTED, bg=COLOR_BG_DARK, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 4))
        vol_slider = tk.Scale(
            ctrl_box,
            from_=0.0,
            to=150.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=100,
            bg=COLOR_BG_DARK,
            troughcolor=COLOR_PANEL_BG,
            activebackground=COLOR_CYAN,
            highlightthickness=0,
            bd=0,
            variable=self.vol_var,
            command=lambda v: self.engine.set_volume(float(v) / 100.0)
        )
        vol_slider.pack(side=tk.LEFT, padx=(0, 14))

        # Tamper attack simulation button
        self.btn_tamper = tk.Checkbutton(
            ctrl_box,
            text="⚠️ Simulate Tamper Attack",
            variable=self.tamper_attack_var,
            command=self._on_toggle_tamper,
            fg=COLOR_RED,
            bg=COLOR_BG_DARK,
            selectcolor=COLOR_PANEL_BG,
            activebackground=COLOR_BG_DARK,
            activeforeground=COLOR_RED,
            font=("Segoe UI", 9, "bold")
        )
        self.btn_tamper.pack(side=tk.LEFT)

        # Main Layout
        content_frame = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=12, pady=4)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib Plots
        self._build_plots(content_frame)

        # Live Security & Performance Telemetry Cards
        self._build_telemetry_cards(content_frame)

        # Telemetry Status Bar
        self._build_statusbar()

    def _build_plots(self, parent):
        plot_frame = tk.Frame(parent, bg=COLOR_PANEL_BG, bd=1, relief=tk.SOLID)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.fig = Figure(figsize=(9, 4.2), facecolor=COLOR_PANEL_BG)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.14, hspace=0.38)

        # Waveform Subplot
        self.ax_wave = self.fig.add_subplot(2, 1, 1)
        self.ax_wave.set_facecolor(COLOR_BG_DARK)
        self.ax_wave.set_title("Post-Decryption Time-Domain Waveform (Decrypted Audio Playback)", fontsize=9, color=COLOR_GREEN, pad=4)
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlim(0, self.engine.block_size)
        self.ax_wave.tick_params(colors=COLOR_TEXT_MUTED, labelsize=7)
        self.ax_wave.grid(True, color="#1E293B", linestyle="--", linewidth=0.5)
        for spine in self.ax_wave.spines.values():
            spine.set_color(COLOR_PANEL_BORDER)

        t = np.arange(self.engine.block_size)
        self.line_wave, = self.ax_wave.plot(t, np.zeros(self.engine.block_size), color=COLOR_GREEN, lw=1.2)

        # Spectrum Subplot
        self.ax_spec = self.fig.add_subplot(2, 1, 2)
        self.ax_spec.set_facecolor(COLOR_BG_DARK)
        self.ax_spec.set_title("Post-Decryption Frequency Spectrum (Hann Windowed FFT)", fontsize=9, color=COLOR_GREEN, pad=4)
        self.ax_spec.set_xscale('log')
        self.ax_spec.set_xlim(20, self.engine.sample_rate / 2.0)
        self.ax_spec.set_ylim(-80, 10)
        self.ax_spec.set_xlabel("Frequency (Hz)", fontsize=7, color=COLOR_TEXT_MUTED)
        self.ax_spec.set_ylabel("Magnitude (dB)", fontsize=7, color=COLOR_TEXT_MUTED)
        self.ax_spec.tick_params(colors=COLOR_TEXT_MUTED, labelsize=7)
        self.ax_spec.grid(True, which='both', color="#1E293B", linestyle="--", linewidth=0.5)
        for spine in self.ax_spec.spines.values():
            spine.set_color(COLOR_PANEL_BORDER)

        dummy_freqs = np.fft.rfftfreq(self.engine.block_size, d=1.0 / self.engine.sample_rate)
        self.line_spec, = self.ax_spec.plot(dummy_freqs, np.full_like(dummy_freqs, -80.0), color=COLOR_CYAN, lw=1.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_telemetry_cards(self, parent):
        card_row = tk.Frame(parent, bg=COLOR_BG_DARK)
        card_row.pack(fill=tk.X, pady=(0, 4))

        # Helper to create telemetry stat tile
        def make_tile(parent, title, val_init, val_color):
            frame = tk.Frame(parent, bg=COLOR_PANEL_BG, padx=12, pady=8, bd=1, relief=tk.GROOVE)
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            lbl_t = tk.Label(frame, text=title, font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG)
            lbl_t.pack(anchor=tk.W)
            lbl_v = tk.Label(frame, text=val_init, font=("Segoe UI", 12, "bold"), fg=val_color, bg=COLOR_PANEL_BG)
            lbl_v.pack(anchor=tk.W)
            return lbl_v

        self.card_ok = make_tile(card_row, "VERIFIED & OK (AES-GCM)", "0", COLOR_GREEN)
        self.card_tampered = make_tile(card_row, "TAMPERED / DROPPED", "0", COLOR_RED)
        self.card_lost = make_tile(card_row, "PACKETS LOST", "0", COLOR_AMBER)
        self.card_jitter = make_tile(card_row, "JITTER BUFFER", "0 blocks (0 ms)", COLOR_CYAN)
        self.card_bitrate = make_tile(card_row, "NETWORK BITRATE", "0.0 kbps", COLOR_TEXT_MAIN)

    def _build_statusbar(self):
        status_bar = tk.Frame(self.root, bg=COLOR_PANEL_BG, padx=14, pady=6, bd=1, relief=tk.SOLID)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_sec_badge = tk.Label(
            status_bar,
            text="● RECEIVER READY — AES-256-GCM SECURE PIPELINE",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_GREEN,
            bg=COLOR_PANEL_BG
        )
        self.lbl_sec_badge.pack(side=tk.LEFT, padx=(0, 16))

        self.lbl_summary = tk.Label(
            status_bar,
            text="Waiting for encrypted audio stream...",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL_BG
        )
        self.lbl_summary.pack(side=tk.LEFT)

        self.lbl_listen = tk.Label(
            status_bar,
            text=f"UDP Port: {self.engine.udp_receiver.port} (0.0.0.0)",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_CYAN,
            bg=COLOR_PANEL_BG
        )
        self.lbl_listen.pack(side=tk.RIGHT)

    def _on_toggle_tamper(self):
        active = self.tamper_attack_var.get()
        self.engine.udp_receiver.simulate_tampering = active
        if active:
            self.lbl_sec_badge.config(text="⚠️ SIMULATING PACKET TAMPERING ATTACK", fg=COLOR_RED)
        else:
            self.lbl_sec_badge.config(text="● AES-256-GCM SECURE INTEGRITY VERIFIED", fg=COLOR_GREEN)

    def _update_loop(self):
        if not self.is_updating:
            return

        viz = self.engine.get_latest_visualization_data()
        if viz is not None:
            samples = viz['samples']
            self.line_wave.set_ydata(samples)

            freqs, mag_db, _, _ = compute_spectrum(
                samples,
                sample_rate=self.engine.sample_rate,
                window_type="hann"
            )
            self.line_spec.set_data(freqs, mag_db)
            self.canvas.draw_idle()

        # Update telemetry stats
        stats = self.engine.udp_receiver.get_stats()
        self.card_ok.config(text=f"{stats['packets_ok']:,}")
        self.card_tampered.config(text=f"{stats['packets_tampered']:,}")
        self.card_lost.config(text=f"{stats['packets_lost']:,}")
        self.card_jitter.config(text=f"{stats['jitter_blocks']} blk ({stats['jitter_ms']} ms)")
        self.card_bitrate.config(text=f"{stats['bitrate_kbps']:.1f} kbps")

        if stats['packets_received'] > 0:
            self.lbl_summary.config(
                text=f"Packets: {stats['packets_received']:,} received | Verified OK: {stats['packets_ok']:,} | Tampered: {stats['packets_tampered']:,} | Lost: {stats['packets_lost']:,}"
            )
        else:
            self.lbl_summary.config(text="Listening for UDP stream... Speak into the sender's microphone!")

        self.root.after(35, self._update_loop)

    def on_closing(self):
        self.is_updating = False
        self.engine.stop()
        self.root.destroy()


def run_receiver(
    port: int = 5005,
    bind_ip: str = "0.0.0.0",
    key_path: str = "shared_secret.key",
    jitter_blocks: int = 3,
    no_gui: bool = False
):
    key_file = Path(key_path)
    if not key_file.exists():
        print(f"Secret key not found at '{key_path}'. Auto-generating 256-bit key for instant session...")
        k = generate_key()
        save_key(k, key_file)
        print(f"Generated and saved key to '{key_path}'.")

    crypto = AudioCrypto(key_file)
    engine = ReceiverEngine(
        port=port,
        crypto=crypto,
        jitter_prebuffer_blocks=jitter_blocks,
        bind_ip=bind_ip
    )

    if no_gui:
        print("=" * 65)
        print(" SpEqualiser - Secure Audio Receiver (Headless Mode)")
        print(f" Listening on: {bind_ip}:{port} via UDP")
        print(" AES-256-GCM Integrity Verification Active | Jitter Buffer Enabled")
        print(" Press Ctrl+C to stop.")
        print("=" * 65)
        engine.start()
        try:
            while True:
                time.sleep(1.0)
                s = engine.udp_receiver.get_stats()
                print(
                    f"  [RX] Total: {s['packets_received']:5d} | OK: {s['packets_ok']:5d} | "
                    f"Tampered: {s['packets_tampered']:3d} | Lost: {s['packets_lost']:3d} | "
                    f"Jitter: {s['jitter_blocks']} blk | Bitrate: {s['bitrate_kbps']:5.1f} kbps\r",
                    end="",
                    flush=True
                )
        except KeyboardInterrupt:
            print("\nStopping receiver...")
        finally:
            engine.stop()
    else:
        root = tk.Tk()
        app = ReceiverGUI(root, engine)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="SpEqualiser Encrypted Audio Receiver (AES-256-GCM + UDP)")
    parser.add_argument("--port", type=int, default=5005, help="UDP listening port (default: 5005)")
    parser.add_argument("--bind", default="0.0.0.0", help="Network bind address (default: 0.0.0.0)")
    parser.add_argument("--key", default="shared_secret.key", help="Path to shared secret key (default: shared_secret.key)")
    parser.add_argument("--jitter-blocks", type=int, default=3, help="Jitter buffer pre-buffer blocks (default: 3)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless console mode without GUI")
    args = parser.parse_args()

    run_receiver(port=args.port, bind_ip=args.bind, key_path=args.key, jitter_blocks=args.jitter_blocks, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
