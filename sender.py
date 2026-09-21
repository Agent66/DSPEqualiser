"""
Secure Audio Sender (AES-256-GCM Encrypted UDP Transport).

Pipeline:
1. Capture Audio: Microphone, Synthetic Demo Track, Signal Generator, or Audio File.
2. DSP Chain: Hann Windowing, FFT Spectrum Analysis, 8-Band Equalizer, Filter, Limiter.
3. Quantization: Float32 audio -> 16-bit signed PCM (int16).
4. Encryption & UDP Transport: AES-256-GCM encryption sent via background thread.
5. Real-Time Visualizer: Pre-encryption waveform & spectrum with interactive EQ sliders.
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
    EQ_CENTER_FREQUENCIES,
    EQ_BAND_NAMES,
    EQ_PRESETS,
    MultiBandEqualizer,
    StandaloneFilter,
    compute_spectrum,
    detect_vad,
    VoiceActivityDetector
)
from spequaliser.generators import SignalGenerator, AudioFileReader, generate_synthetic_demo_track
from spequaliser.network import UDPAudioSender


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


class SenderEngine:
    """
    Audio and DSP engine for the Sender.
    Performs capture, real-time DSP, quantization, and hands off to UDP worker.
    """
    def __init__(
        self,
        host: str,
        port: int,
        crypto: AudioCrypto,
        sample_rate: int = STANDARD_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        source_mode: str = "mic"
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.source_mode = source_mode

        # Network Sender
        self.udp_sender = UDPAudioSender(host=host, port=port, crypto=crypto)

        # DSP Components
        self.equalizer = MultiBandEqualizer(sample_rate=sample_rate)
        self.filter = StandaloneFilter(sample_rate=sample_rate)

        # Voice Activity Detection (VAD) for Telecom DTX
        self.vad = VoiceActivityDetector(sample_rate=sample_rate, hangover_frames=5)
        self.vad_enabled = False
        self.total_packets_evaluated = 0
        self.packets_suppressed = 0

        # Sources
        self.signal_gen = SignalGenerator(sample_rate=sample_rate)
        self.file_reader = AudioFileReader(target_sample_rate=sample_rate)
        self.demo_reader = AudioFileReader(target_sample_rate=sample_rate)
        self._init_demo_track()

        self.stream: Optional[Any] = None
        self.is_running = False
        self.monitor_local_audio = False  # Default muted locally to avoid feedback
        self.master_gain = 1.0
        self.mic_input_gain = 1.0
        self.viz_queue: queue.Queue = queue.Queue(maxsize=4)
        self._lock = threading.Lock()

    def set_vad_enabled(self, enabled: bool):
        with self._lock:
            self.vad_enabled = enabled
            if not enabled:
                self.vad.reset()

    def set_filter_params(self, mode: str, cutoff: float, q: float = 1.0, order: int = 4):
        with self._lock:
            self.filter.set_params(mode=mode, cutoff=cutoff, q=q, order=order)

    def _init_demo_track(self):
        demo_audio = generate_synthetic_demo_track(duration_sec=12.0, fs=self.sample_rate)
        self.demo_reader.audio_data = demo_audio
        self.demo_reader.file_sample_rate = self.sample_rate
        self.demo_reader.duration_seconds = len(demo_audio) / float(self.sample_rate)
        self.demo_reader.is_looping = True

    def set_source(self, source: str):
        with self._lock:
            self.source_mode = source.lower()

    def set_master_gain(self, gain: float):
        self.master_gain = float(np.clip(gain, 0.0, 2.0))

    def set_mic_gain(self, gain: float):
        self.mic_input_gain = float(np.clip(gain, 0.0, 3.0))

    def start(self, input_device: Optional[int] = None, output_device: Optional[int] = None):
        if self.is_running:
            return

        self.udp_sender.start()
        self.is_running = True

        if sd is None:
            print("Warning: sounddevice not available. Running in mock/generator mode.")
            return

        try:
            # Try full duplex first
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=(1, 1),
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception:
            # Fallback to output-only or input-only depending on availability
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=input_device,
                    channels=1,
                    dtype='float32',
                    callback=self._input_only_callback
                )
                self.stream.start()
            except Exception as e:
                # If mic is unavailable, run generator loop in background thread
                print(f"Notice: Audio input device unavailable ({e}). Running synthetic audio source.")
                self.stream = None
                self._run_synthetic_thread()

    def _run_synthetic_thread(self):
        def loop():
            interval = self.block_size / float(self.sample_rate)
            while self.is_running:
                start_t = time.time()
                in_chunk = np.zeros(self.block_size, dtype=np.float64)
                self._process_and_transmit(in_chunk, None, self.block_size)
                elapsed = time.time() - start_t
                sleep_t = max(0.001, interval - elapsed)
                time.sleep(sleep_t)
        t = threading.Thread(target=loop, daemon=True)
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
        self.udp_sender.stop()

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        in_chunk = indata[:, 0].astype(np.float64) * self.mic_input_gain
        self._process_and_transmit(in_chunk, outdata, frames)

    def _input_only_callback(self, indata, frames, time_info, status):
        in_chunk = indata[:, 0].astype(np.float64) * self.mic_input_gain
        self._process_and_transmit(in_chunk, None, frames)

    def _process_and_transmit(self, mic_chunk: np.ndarray, outdata: Optional[np.ndarray], frames: int):
        with self._lock:
            # 1. Capture Source
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

            # 2. DSP Pipeline: Standalone Filter -> 8-Band Equalizer -> Master Gain -> Limiter
            x_filt = self.filter.process(raw_input)
            x_eq = self.equalizer.process(x_filt)
            processed = x_eq * self.master_gain
            processed = np.tanh(processed)

            # 3. Voice Activity Detection (VAD) Evaluation
            is_speech, energy_db, zcr = self.vad.process_block(processed)
            self.total_packets_evaluated += 1

            should_transmit = True
            if self.vad_enabled:
                should_transmit = is_speech
                if not should_transmit:
                    self.packets_suppressed += 1

            # 4. Local Audio Monitor (optional)
            if outdata is not None:
                if self.monitor_local_audio:
                    outdata[:, 0] = processed.astype(np.float32)
                else:
                    outdata[:, 0] = 0.0

            # 5. Quantize to 16-bit PCM and send if speech active / DTX allows
            if should_transmit:
                pcm_int16 = np.clip(processed * 32767.0, -32768.0, 32767.0).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()
                # Hand off to UDP network worker (non-blocking)
                self.udp_sender.send_pcm_chunk(pcm_bytes)

            # 6. Queue for UI visualizer
            if not self.viz_queue.full():
                try:
                    savings = (self.packets_suppressed / max(1, self.total_packets_evaluated)) * 100.0
                    self.viz_queue.put_nowait({
                        'input': raw_input.copy(),
                        'output': processed.copy(),
                        'is_speech': is_speech,
                        'energy_db': energy_db,
                        'zcr': zcr,
                        'vad_enabled': self.vad_enabled,
                        'transmitting': should_transmit,
                        'savings_pct': savings,
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


class SenderGUI:
    """
    Tkinter/Matplotlib GUI for the Secure Audio Sender.
    """
    def __init__(self, root: tk.Tk, engine: SenderEngine):
        self.root = root
        self.engine = engine
        self.root.title("Secure Audio Sender (AES-256-GCM + UDP) — SpEqualiser")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.configure(bg=COLOR_BG_DARK)

        self.selected_source = tk.StringVar(value=self.engine.source_mode)
        self.selected_preset = tk.StringVar(value="Flat")
        self.monitor_var = tk.BooleanVar(value=self.engine.monitor_local_audio)
        self.vad_var = tk.BooleanVar(value=self.engine.vad_enabled)
        self.filter_mode_var = tk.StringVar(value="bypass")
        self.filter_cutoff_var = tk.DoubleVar(value=1000.0)
        self.gain_var = tk.DoubleVar(value=100.0)

        self.f_eval = np.logspace(np.log10(20), np.log10(self.engine.sample_rate / 2.0 - 50.0), 200)

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
            text="SECURE AUDIO SENDER — AES-256-GCM ENCRYPTED UDP",
            font=("Segoe UI", 13, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK
        )
        lbl_title.pack(anchor=tk.W)

        lbl_sub = tk.Label(
            title_box,
            text=f"Target: {self.engine.udp_sender.host}:{self.engine.udp_sender.port} • Pipeline: Capture → VAD/DTX → Filter → 8-Band EQ → 16-bit PCM → AES-GCM",
            font=("Segoe UI", 9),
            fg=COLOR_CYAN,
            bg=COLOR_BG_DARK
        )
        lbl_sub.pack(anchor=tk.W)

        # Source selector & Mute toggle & Smart VAD toggle
        ctrl_box = tk.Frame(header, bg=COLOR_BG_DARK)
        ctrl_box.pack(side=tk.RIGHT)

        tk.Label(ctrl_box, text="Source:", fg=COLOR_TEXT_MUTED, bg=COLOR_BG_DARK, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 4))
        src_combo = ttk.Combobox(
            ctrl_box,
            textvariable=self.selected_source,
            values=["mic", "demo", "generator"],
            state="readonly",
            width=10
        )
        src_combo.pack(side=tk.LEFT, padx=(0, 8))
        src_combo.bind("<<ComboboxSelected>>", lambda e: self.engine.set_source(self.selected_source.get()))

        chk_vad = tk.Checkbutton(
            ctrl_box,
            text="Smart VAD / DTX",
            variable=self.vad_var,
            command=self._on_vad_toggle,
            fg=COLOR_CYAN,
            bg=COLOR_BG_DARK,
            selectcolor=COLOR_PANEL_BG,
            activebackground=COLOR_BG_DARK,
            activeforeground=COLOR_CYAN,
            font=("Segoe UI", 9, "bold")
        )
        chk_vad.pack(side=tk.LEFT, padx=(0, 8))

        chk_mon = tk.Checkbutton(
            ctrl_box,
            text="Monitor Locally",
            variable=self.monitor_var,
            command=self._on_monitor_toggle,
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK,
            selectcolor=COLOR_PANEL_BG,
            activebackground=COLOR_BG_DARK,
            activeforeground=COLOR_TEXT_MAIN,
            font=("Segoe UI", 9)
        )
        chk_mon.pack(side=tk.LEFT, padx=(0, 4))

        # Main Layout: Visualizer on top/center, EQ on bottom
        content_frame = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=12, pady=4)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib Plots
        self._build_plots(content_frame)

        # 8-Band Equalizer Controls
        self._build_eq_panel(content_frame)

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
        self.ax_wave.set_title("Pre-Encryption Time-Domain Waveform (Oscilloscope)", fontsize=9, color=COLOR_CYAN, pad=4)
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlim(0, self.engine.block_size)
        self.ax_wave.tick_params(colors=COLOR_TEXT_MUTED, labelsize=7)
        self.ax_wave.grid(True, color="#1E293B", linestyle="--", linewidth=0.5)
        for spine in self.ax_wave.spines.values():
            spine.set_color(COLOR_PANEL_BORDER)

        t = np.arange(self.engine.block_size)
        self.line_wave, = self.ax_wave.plot(t, np.zeros(self.engine.block_size), color=COLOR_BLUE, lw=1.2)

        # Spectrum Subplot
        self.ax_spec = self.fig.add_subplot(2, 1, 2)
        self.ax_spec.set_facecolor(COLOR_BG_DARK)
        self.ax_spec.set_title("Pre-Encryption Frequency Spectrum (Hann Windowed FFT)", fontsize=9, color=COLOR_CYAN, pad=4)
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
        self.line_eq_curve, = self.ax_spec.plot(self.f_eval, np.zeros_like(self.f_eval), color=COLOR_AMBER, lw=1.0, ls="--", alpha=0.8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_eq_panel(self, parent):
        eq_card = tk.LabelFrame(
            parent,
            text=" DSP CONTROLS: STANDALONE FILTER & 8-BAND GRAPHIC EQUALIZER ",
            bg=COLOR_PANEL_BG,
            fg=COLOR_CYAN,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
            bd=1,
            relief=tk.GROOVE
        )
        eq_card.pack(fill=tk.X, pady=(0, 4))

        # Top Control Row: Standalone Filters (LP/HP/BP/Notch) & EQ Preset
        top_ctrl_bar = tk.Frame(eq_card, bg=COLOR_PANEL_BG)
        top_ctrl_bar.pack(fill=tk.X, pady=(0, 6))

        # Filter mode
        tk.Label(top_ctrl_bar, text="Filter:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        filter_combo = ttk.Combobox(
            top_ctrl_bar,
            textvariable=self.filter_mode_var,
            values=["bypass", "lowpass", "highpass", "bandpass", "notch"],
            state="readonly",
            width=10
        )
        filter_combo.pack(side=tk.LEFT, padx=(0, 8))
        filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        self.lbl_cutoff = tk.Label(top_ctrl_bar, text="Cutoff: 1000 Hz", bg=COLOR_PANEL_BG, fg=COLOR_AMBER, font=("Segoe UI", 8))
        self.lbl_cutoff.pack(side=tk.LEFT, padx=(0, 4))

        cutoff_scale = tk.Scale(
            top_ctrl_bar,
            from_=60.0,
            to=12000.0,
            resolution=50.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.filter_cutoff_var,
            length=110,
            bg=COLOR_PANEL_BG,
            fg=COLOR_TEXT_MAIN,
            troughcolor=COLOR_BG_DARK,
            activebackground=COLOR_CYAN,
            highlightthickness=0,
            bd=0,
            command=self._on_filter_cutoff_change
        )
        cutoff_scale.pack(side=tk.LEFT, padx=(0, 16))

        # Preset selection
        tk.Label(top_ctrl_bar, text="EQ Preset:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 6))
        preset_combo = ttk.Combobox(
            top_ctrl_bar,
            textvariable=self.selected_preset,
            values=list(EQ_PRESETS.keys()),
            state="readonly",
            width=14
        )
        preset_combo.pack(side=tk.LEFT, padx=(0, 8))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        btn_reset = tk.Button(
            top_ctrl_bar,
            text="Reset (0 dB)",
            command=self._reset_eq,
            bg="#1E293B",
            fg=COLOR_TEXT_MAIN,
            font=("Segoe UI", 8),
            bd=1
        )
        btn_reset.pack(side=tk.LEFT)

        # 8 Faders
        fader_container = tk.Frame(eq_card, bg=COLOR_PANEL_BG)
        fader_container.pack(fill=tk.X)

        self.eq_sliders = []
        self.eq_labels = []

        for i, (freq_hz, name) in enumerate(zip(EQ_CENTER_FREQUENCIES, EQ_BAND_NAMES)):
            col = tk.Frame(fader_container, bg=COLOR_PANEL_BG)
            col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4)

            val_lbl = tk.Label(col, text="0.0 dB", font=("Segoe UI", 7, "bold"), fg=COLOR_AMBER, bg=COLOR_PANEL_BG)
            val_lbl.pack()
            self.eq_labels.append(val_lbl)

            slider = tk.Scale(
                col,
                from_=12.0,
                to=-12.0,
                resolution=0.5,
                orient=tk.VERTICAL,
                showvalue=False,
                length=75,
                bg=COLOR_BG_DARK,
                fg=COLOR_TEXT_MAIN,
                troughcolor=COLOR_PANEL_BG,
                activebackground=COLOR_CYAN,
                highlightthickness=0,
                bd=0,
                command=lambda val, idx=i: self._on_slider_change(idx, val)
            )
            slider.set(0.0)
            slider.pack(pady=2)
            self.eq_sliders.append(slider)

            freq_lbl = tk.Label(col, text=name, font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG)
            freq_lbl.pack()

    def _build_statusbar(self):
        status_bar = tk.Frame(self.root, bg=COLOR_PANEL_BG, padx=14, pady=6, bd=1, relief=tk.SOLID)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_sec_badge = tk.Label(
            status_bar,
            text="● AES-256-GCM SECURE STREAM ACTIVE",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_GREEN,
            bg=COLOR_PANEL_BG
        )
        self.lbl_sec_badge.pack(side=tk.LEFT, padx=(0, 14))

        self.lbl_vad_badge = tk.Label(
            status_bar,
            text="● STREAMING: CONTINUOUS",
            font=("Segoe UI", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG
        )
        self.lbl_vad_badge.pack(side=tk.LEFT, padx=(0, 14))

        self.lbl_stats = tk.Label(
            status_bar,
            text="Packets Sent: 0 | Total Data: 0.0 KB | Bitrate: 0.0 kbps",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL_BG
        )
        self.lbl_stats.pack(side=tk.LEFT)

        self.lbl_dest = tk.Label(
            status_bar,
            text=f"UDP Dest: {self.engine.udp_sender.host}:{self.engine.udp_sender.port}",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_CYAN,
            bg=COLOR_PANEL_BG
        )
        self.lbl_dest.pack(side=tk.RIGHT)

    def _on_vad_toggle(self):
        self.engine.set_vad_enabled(self.vad_var.get())

    def _on_filter_change(self, event=None):
        mode = self.filter_mode_var.get()
        cutoff = float(self.filter_cutoff_var.get())
        self.engine.set_filter_params(mode=mode, cutoff=cutoff)

    def _on_filter_cutoff_change(self, val):
        cutoff = float(val)
        self.lbl_cutoff.config(text=f"Cutoff: {cutoff:.0f} Hz")
        mode = self.filter_mode_var.get()
        self.engine.set_filter_params(mode=mode, cutoff=cutoff)

    def _on_monitor_toggle(self):
        self.engine.monitor_local_audio = self.monitor_var.get()

    def _on_slider_change(self, band_idx: int, val: str):
        gain_db = float(val)
        self.engine.equalizer.set_band_gain(band_idx, gain_db)
        self.eq_labels[band_idx].config(text=f"{gain_db:+.1f} dB")
        # Update theoretical EQ response
        h_db = self.engine.equalizer.get_frequency_response(self.f_eval)
        self.line_eq_curve.set_ydata(h_db)

    def _on_preset_change(self, event=None):
        preset_name = self.selected_preset.get()
        if preset_name in EQ_PRESETS:
            gains = EQ_PRESETS[preset_name]
            for i, g in enumerate(gains):
                self.eq_sliders[i].set(g)
                self.engine.equalizer.set_band_gain(i, g)
                self.eq_labels[i].config(text=f"{g:+.1f} dB")
            h_db = self.engine.equalizer.get_frequency_response(self.f_eval)
            self.line_eq_curve.set_ydata(h_db)

    def _reset_eq(self):
        self.selected_preset.set("Flat")
        self._on_preset_change()

    def _update_loop(self):
        if not self.is_updating:
            return

        viz = self.engine.get_latest_visualization_data()
        if viz is not None:
            out_signal = viz['output']
            self.line_wave.set_ydata(out_signal)

            freqs, mag_db, _, _ = compute_spectrum(
                out_signal,
                sample_rate=self.engine.sample_rate,
                window_type="hann"
            )
            self.line_spec.set_data(freqs, mag_db)
            self.canvas.draw_idle()

            # Update VAD status badge
            if viz.get('vad_enabled', False):
                if viz.get('transmitting', True):
                    self.lbl_vad_badge.config(
                        text=f"🟢 VAD: SPEECH ({viz['energy_db']:.0f} dB | ZCR: {viz['zcr']:.2f})",
                        fg=COLOR_GREEN
                    )
                else:
                    savings = viz.get('savings_pct', 0.0)
                    self.lbl_vad_badge.config(
                        text=f"⏸️ VAD: SILENCE (Bandwidth Saved: {savings:.1f}%)",
                        fg=COLOR_AMBER
                    )
            else:
                self.lbl_vad_badge.config(
                    text="● STREAMING: CONTINUOUS (DTX OFF)",
                    fg=COLOR_TEXT_MUTED
                )

        # Update telemetry
        stats = self.engine.udp_sender.get_stats()
        kb_sent = stats['bytes_sent'] / 1024.0
        self.lbl_stats.config(
            text=f"Packets Sent: {stats['packets_sent']:,} | Total: {kb_sent:,.1f} KB | Bitrate: {stats['bitrate_kbps']:.1f} kbps"
        )

        self.root.after(35, self._update_loop)

    def on_closing(self):
        self.is_updating = False
        self.engine.stop()
        self.root.destroy()


def run_sender(
    host: str = "127.0.0.1",
    port: int = 5005,
    key_path: str = "shared_secret.key",
    source: str = "mic",
    vad: bool = False,
    no_gui: bool = False
):
    # Ensure key exists
    key_file = Path(key_path)
    if not key_file.exists():
        print(f"Secret key not found at '{key_path}'. Auto-generating 256-bit key for instant session...")
        k = generate_key()
        save_key(k, key_file)
        print(f"Generated and saved key to '{key_path}'.")

    crypto = AudioCrypto(key_file)
    engine = SenderEngine(host=host, port=port, crypto=crypto, source_mode=source)
    if vad:
        engine.set_vad_enabled(True)

    if no_gui:
        print("=" * 65)
        print(" SpEqualiser - Secure Audio Sender (Headless Mode)")
        print(f" Streaming to: {host}:{port} via UDP")
        print(f" Source: {source} | AES-256-GCM Encryption Active")
        print(f" Smart VAD / DTX: {'ENABLED' if engine.vad_enabled else 'DISABLED'}")
        print(" Press Ctrl+C to stop.")
        print("=" * 65)
        engine.start()
        try:
            while True:
                time.sleep(1.0)
                stats = engine.udp_sender.get_stats()
                kb = stats['bytes_sent'] / 1024.0
                print(f"  [TX] Packets: {stats['packets_sent']:6d} | Data: {kb:7.1f} KB | Bitrate: {stats['bitrate_kbps']:5.1f} kbps\r", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopping sender...")
        finally:
            engine.stop()
    else:
        root = tk.Tk()
        app = SenderGUI(root, engine)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="SpEqualiser Encrypted Audio Sender (AES-256-GCM + UDP)")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5005, help="Receiver UDP port (default: 5005)")
    parser.add_argument("--key", default="shared_secret.key", help="Path to shared secret key (default: shared_secret.key)")
    parser.add_argument("--source", choices=["mic", "demo", "generator", "file"], default="mic", help="Audio source (default: mic)")
    parser.add_argument("--vad", action="store_true", help="Enable Smart Voice Activity Detection (DTX bandwidth saver)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless console mode without GUI")
    args = parser.parse_args()

    run_sender(host=args.host, port=args.port, key_path=args.key, source=args.source, vad=args.vad, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
