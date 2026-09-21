"""
Graphical User Interface (GUI) for Real-Time Audio Equalizer & Spectrum Analyzer.

Built with Tkinter and Matplotlib (FigureCanvasTkAgg).
Features:
- Sleek dark aesthetic tailored to the DSP presentation theme
- Real-Time Oscilloscope Waveform Visualizer
- Real-Time Logarithmic Frequency Spectrum Analyzer (FFT) with peak detection and theoretical EQ curve
- 8-Band Graphic Equalizer with vertical faders and audio presets
- Standalone Digital Filter panel (Low-Pass, High-Pass, Band-Pass, Notch)
- DSP Extensions: Pitch Detection with Musical Note display & Spectral Subtraction Noise Reduction
- Audio Source Router (Microphone, Test Signal Generator, WAV file player, Synthetic demo track)
"""

import sys
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from .dsp import (
    STANDARD_SAMPLE_RATE,
    DEFAULT_BLOCK_SIZE,
    EQ_CENTER_FREQUENCIES,
    EQ_BAND_NAMES,
    EQ_PRESETS,
    compute_spectrum,
    detect_pitch
)
from .audio_engine import AudioEngine


# Color Palette (Deep Space DSP Theme)
COLOR_BG_DARK = "#090F1D"        # Deepest dark navy
COLOR_PANEL_BG = "#111A2E"       # Panel background
COLOR_PANEL_BORDER = "#1E2C48"   # Border lines
COLOR_TEXT_MAIN = "#F1F5F9"      # Primary white-ish text
COLOR_TEXT_MUTED = "#94A3B8"     # Secondary gray text
COLOR_CYAN = "#00F2FE"           # Cyan accent for spectrum
COLOR_BLUE = "#3B82F6"           # Electric blue for waveform
COLOR_AMBER = "#F59E0B"          # Amber for EQ curve / notes
COLOR_GREEN = "#10B981"          # Emerald green for status & pitch
COLOR_RED = "#EF4444"            # Crimson for warnings & notch


class EqualizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real-Time Audio Equalizer & Spectrum Analyzer — DSP Suite")
        self.root.geometry("1260x860")
        self.root.minsize(1050, 750)
        self.root.configure(bg=COLOR_BG_DARK)

        # Initialize Audio Engine
        self.engine = AudioEngine(sample_rate=STANDARD_SAMPLE_RATE, block_size=DEFAULT_BLOCK_SIZE)

        # GUI State Variables
        self.selected_window = tk.StringVar(value="hann")
        self.selected_source = tk.StringVar(value="demo")
        self.selected_preset = tk.StringVar(value="Flat")
        self.filter_mode = tk.StringVar(value="bypass")
        self.gen_mode = tk.StringVar(value="sine")
        self.gen_freq_var = tk.DoubleVar(value=440.0)
        self.filter_cutoff_var = tk.DoubleVar(value=1000.0)
        self.filter_q_var = tk.DoubleVar(value=1.0)
        self.master_vol_var = tk.DoubleVar(value=85.0)
        self.nr_enabled_var = tk.BooleanVar(value=False)
        self.show_eq_curve_var = tk.BooleanVar(value=True)

        # Matplotlib frequency grid for theoretical curves (log-spaced)
        self.f_eval = np.logspace(np.log10(20), np.log10(STANDARD_SAMPLE_RATE / 2.0 - 50.0), 300)

        # Build UI Layout
        self._setup_styles()
        self._build_header()
        self._build_main_content()
        self._build_statusbar()

        # Start animation loop
        self.is_updating = True
        self.root.after(30, self._update_visualizer)

        # Start audio automatically with demo track
        self._start_audio()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(".", background=COLOR_BG_DARK, foreground=COLOR_TEXT_MAIN)
        style.configure("TFrame", background=COLOR_BG_DARK)
        style.configure("Card.TFrame", background=COLOR_PANEL_BG)

        style.configure("TLabel", background=COLOR_PANEL_BG, foreground=COLOR_TEXT_MAIN, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=COLOR_BG_DARK, foreground=COLOR_TEXT_MAIN, font=("Segoe UI", 15, "bold"))
        style.configure("SubHeader.TLabel", background=COLOR_BG_DARK, foreground=COLOR_CYAN, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=COLOR_PANEL_BG, foreground=COLOR_CYAN, font=("Segoe UI", 10, "bold"))
        style.configure("Value.TLabel", background=COLOR_PANEL_BG, foreground=COLOR_AMBER, font=("Segoe UI", 8, "bold"))

        style.configure("TButton", background="#1E293B", foreground=COLOR_TEXT_MAIN, borderwidth=1, font=("Segoe UI", 9))
        style.map("TButton",
            background=[('active', '#334155'), ('pressed', '#0F172A')],
            foreground=[('active', '#FFFFFF')]
        )

        style.configure("Accent.TButton", background="#2563EB", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",
            background=[('active', '#1D4ED8'), ('pressed', '#1E40AF')]
        )

        style.configure("Stop.TButton", background="#DC2626", foreground="#FFFFFF", font=("Segoe UI", 9, "bold"))
        style.map("Stop.TButton",
            background=[('active', '#B91C1C'), ('pressed', '#991B1B')]
        )

        style.configure("TCombobox", fieldbackground=COLOR_BG_DARK, background=COLOR_PANEL_BG, foreground=COLOR_TEXT_MAIN)
        style.map("TCombobox",
            fieldbackground=[('readonly', COLOR_BG_DARK)],
            foreground=[('readonly', COLOR_TEXT_MAIN)]
        )

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=16, pady=8)
        header_frame.pack(fill=tk.X)

        # Title & Subtitle
        text_frame = tk.Frame(header_frame, bg=COLOR_BG_DARK)
        text_frame.pack(side=tk.LEFT)

        title = tk.Label(
            text_frame,
            text="REAL-TIME AUDIO EQUALIZER & SPECTRUM ANALYZER",
            font=("Segoe UI", 14, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK
        )
        title.pack(anchor=tk.W)

        subtitle = tk.Label(
            text_frame,
            text="Digital Signal Processing Project • 44.1 kHz Sampling Rate • FFT & Biquad Filter Engine",
            font=("Segoe UI", 9),
            fg=COLOR_CYAN,
            bg=COLOR_BG_DARK
        )
        subtitle.pack(anchor=tk.W)

        # Master Controls
        controls_frame = tk.Frame(header_frame, bg=COLOR_BG_DARK)
        controls_frame.pack(side=tk.RIGHT)

        self.btn_start = tk.Button(
            controls_frame,
            text="▶ Start Audio",
            command=self._start_audio,
            bg="#16A34A",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_stop = tk.Button(
            controls_frame,
            text="⏹ Stop",
            command=self._stop_audio,
            bg="#DC2626",
            fg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=4,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        # Master Volume
        tk.Label(controls_frame, text="Vol:", bg=COLOR_BG_DARK, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.slider_vol = tk.Scale(
            controls_frame,
            from_=0,
            to=120,
            orient=tk.HORIZONTAL,
            variable=self.master_vol_var,
            command=self._on_master_vol_change,
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_MAIN,
            highlightthickness=0,
            length=100,
            showvalue=False
        )
        self.slider_vol.pack(side=tk.LEFT, padx=2)

        self.lbl_vol_val = tk.Label(controls_frame, text="85%", bg=COLOR_BG_DARK, fg=COLOR_AMBER, font=("Segoe UI", 9, "bold"), width=4)
        self.lbl_vol_val.pack(side=tk.LEFT, padx=2)

        # Encrypted UDP Streaming Launchers
        btn_sender = tk.Button(
            controls_frame,
            text="🔒 Sender",
            command=self._launch_sender,
            bg="#1E293B",
            fg=COLOR_CYAN,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=3,
            relief=tk.GROOVE,
            cursor="hand2"
        )
        btn_sender.pack(side=tk.LEFT, padx=(10, 2))

        btn_receiver = tk.Button(
            controls_frame,
            text="📡 Receiver",
            command=self._launch_receiver,
            bg="#1E293B",
            fg=COLOR_GREEN,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=3,
            relief=tk.GROOVE,
            cursor="hand2"
        )
        btn_receiver.pack(side=tk.LEFT, padx=(2, 2))

    def _build_main_content(self):
        # Vertical split: Top = Visualizers (Oscilloscope + FFT), Bottom = Equalizer & Controls
        main_paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, bg=COLOR_PANEL_BORDER, bd=0, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # -------------------------------------------------------------
        # 1. VISUALIZER PANEL (MATPLOTLIB EMBEDDED)
        # -------------------------------------------------------------
        viz_container = tk.Frame(main_paned, bg=COLOR_PANEL_BG)
        main_paned.add(viz_container, minsize=320)

        self._build_matplotlib_plots(viz_container)

        # -------------------------------------------------------------
        # 2. CONTROLS PANEL (EQUALIZER + FILTERS + EXTENSIONS)
        # -------------------------------------------------------------
        controls_container = tk.Frame(main_paned, bg=COLOR_BG_DARK)
        main_paned.add(controls_container, minsize=290)

        # Grid with 3 columns:
        # Col 0: 8-Band Equalizer (Expand)
        # Col 1: Standalone Digital Filters & Audio Routing
        # Col 2: DSP Extensions (Pitch Detection & Noise Reduction)
        controls_container.columnconfigure(0, weight=5)
        controls_container.columnconfigure(1, weight=3)
        controls_container.columnconfigure(2, weight=3)
        controls_container.rowconfigure(0, weight=1)

        self._build_eq_panel(controls_container)
        self._build_filters_panel(controls_container)
        self._build_extensions_panel(controls_container)

    def _build_matplotlib_plots(self, parent):
        # Matplotlib Figure with 2 subplots (Waveform on top, Spectrum on bottom)
        self.fig = Figure(figsize=(10, 4.2), dpi=100, facecolor=COLOR_PANEL_BG)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.4], hspace=0.35, top=0.92, bottom=0.12, left=0.06, right=0.97)

        # Subplot 1: Waveform (Oscilloscope)
        self.ax_wave = self.fig.add_subplot(gs[0, 0])
        self.ax_wave.set_facecolor(COLOR_BG_DARK)
        self.ax_wave.set_title("TIME-DOMAIN WAVEFORM (OSCILLOSCOPE)", fontsize=9, color=COLOR_BLUE, fontweight="bold", pad=4, loc="left")
        self.ax_wave.set_xlim(0, DEFAULT_BLOCK_SIZE / STANDARD_SAMPLE_RATE * 1000.0)  # ms
        self.ax_wave.set_ylim(-1.05, 1.05)
        self.ax_wave.set_ylabel("Amplitude", fontsize=8, color=COLOR_TEXT_MUTED)
        self.ax_wave.tick_params(colors=COLOR_TEXT_MUTED, labelsize=7)
        self.ax_wave.grid(True, color="#1B283E", linestyle="--", linewidth=0.5)

        time_axis_ms = np.arange(DEFAULT_BLOCK_SIZE) / STANDARD_SAMPLE_RATE * 1000.0
        self.line_wave_in, = self.ax_wave.plot(time_axis_ms, np.zeros(DEFAULT_BLOCK_SIZE), color="#475569", lw=1.0, label="Input", alpha=0.6)
        self.line_wave_out, = self.ax_wave.plot(time_axis_ms, np.zeros(DEFAULT_BLOCK_SIZE), color=COLOR_BLUE, lw=1.3, label="Output")
        self.ax_wave.legend(loc="upper right", fontsize=7, facecolor=COLOR_PANEL_BG, edgecolor=COLOR_PANEL_BORDER, labelcolor=COLOR_TEXT_MAIN)

        # Subplot 2: Spectrum Analyzer (FFT)
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.ax_spec.set_facecolor(COLOR_BG_DARK)
        self.ax_spec.set_title("FREQUENCY-DOMAIN SPECTRUM (FFT & FILTER RESPONSE)", fontsize=9, color=COLOR_CYAN, fontweight="bold", pad=4, loc="left")
        self.ax_spec.set_xscale('log')
        self.ax_spec.set_xlim(20, 20000)
        self.ax_spec.set_ylim(-80, 15)
        self.ax_spec.set_xlabel("Frequency (Hz)", fontsize=8, color=COLOR_TEXT_MUTED)
        self.ax_spec.set_ylabel("Magnitude (dB)", fontsize=8, color=COLOR_TEXT_MUTED)
        self.ax_spec.set_xticks([20, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000])
        self.ax_spec.set_xticklabels(["20", "50", "100", "250", "500", "1k", "2.5k", "5k", "10k", "20k"])
        self.ax_spec.tick_params(colors=COLOR_TEXT_MUTED, labelsize=7)
        self.ax_spec.grid(True, which="both", color="#1B283E", linestyle="--", linewidth=0.5)

        # Lines for Spectrum: Input FFT, Output FFT, Theoretical EQ curve, Peak marker
        init_freqs = np.fft.rfftfreq(DEFAULT_BLOCK_SIZE, 1.0 / STANDARD_SAMPLE_RATE)
        self.line_spec_in, = self.ax_spec.plot(init_freqs, np.full_like(init_freqs, -80.0), color="#64748B", lw=0.9, ls=":", label="Input FFT", alpha=0.6)
        self.line_spec_out, = self.ax_spec.plot(init_freqs, np.full_like(init_freqs, -80.0), color=COLOR_CYAN, lw=1.5, label="Output FFT")
        self.line_eq_curve, = self.ax_spec.plot(self.f_eval, np.zeros_like(self.f_eval), color=COLOR_AMBER, lw=1.2, ls="--", label="EQ Target Curve", alpha=0.85)

        self.peak_point, = self.ax_spec.plot([1000], [-80], marker="o", markersize=5, color=COLOR_RED, linestyle='None')
        self.peak_text = self.ax_spec.text(1000, 0, "", color=COLOR_RED, fontsize=8, fontweight="bold")
        self.ax_spec.legend(loc="upper right", fontsize=7, facecolor=COLOR_PANEL_BG, edgecolor=COLOR_PANEL_BORDER, labelcolor=COLOR_TEXT_MAIN)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_eq_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" 8-BAND GRAPHIC EQUALIZER ",
            bg=COLOR_PANEL_BG,
            fg=COLOR_CYAN,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
            bd=1,
            relief=tk.SOLID
        )
        frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Top row: Presets dropdown & Quick Buttons
        top_bar = tk.Frame(frame, bg=COLOR_PANEL_BG)
        top_bar.pack(fill=tk.X, pady=(0, 6))

        tk.Label(top_bar, text="Preset:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)
        self.combo_presets = ttk.Combobox(
            top_bar,
            values=list(EQ_PRESETS.keys()),
            textvariable=self.selected_preset,
            state="readonly",
            width=14,
            font=("Segoe UI", 8)
        )
        self.combo_presets.pack(side=tk.LEFT, padx=4)
        self.combo_presets.bind("<<ComboboxSelected>>", self._on_preset_selected)

        btn_flat = tk.Button(
            top_bar,
            text="Reset Flat",
            command=self._reset_eq_flat,
            bg="#334155",
            fg=COLOR_TEXT_MAIN,
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            padx=6,
            pady=1,
            cursor="hand2"
        )
        btn_flat.pack(side=tk.LEFT, padx=4)

        self.btn_bypass_eq = tk.Button(
            top_bar,
            text="Bypass EQ: OFF",
            command=self._toggle_eq_bypass,
            bg="#1E293B",
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            padx=6,
            pady=1,
            cursor="hand2"
        )
        self.btn_bypass_eq.pack(side=tk.LEFT, padx=4)

        # 8 Vertical Sliders Container
        sliders_frame = tk.Frame(frame, bg=COLOR_PANEL_BG)
        sliders_frame.pack(fill=tk.BOTH, expand=True)

        self.eq_sliders = []
        self.eq_val_labels = []

        for i, (freq_hz, name) in enumerate(zip(EQ_CENTER_FREQUENCIES, EQ_BAND_NAMES)):
            col_frame = tk.Frame(sliders_frame, bg=COLOR_PANEL_BG)
            col_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

            # Gain value label on top
            val_lbl = tk.Label(col_frame, text="0.0 dB", bg=COLOR_PANEL_BG, fg=COLOR_AMBER, font=("Segoe UI", 7, "bold"))
            val_lbl.pack(side=tk.TOP)
            self.eq_val_labels.append(val_lbl)

            # Vertical slider (-12 dB to +12 dB)
            slider = tk.Scale(
                col_frame,
                from_=12.0,
                to=-12.0,
                resolution=0.5,
                orient=tk.VERTICAL,
                bg=COLOR_PANEL_BG,
                fg=COLOR_TEXT_MAIN,
                activebackground=COLOR_CYAN,
                troughcolor=COLOR_BG_DARK,
                highlightthickness=0,
                showvalue=False,
                command=lambda val, idx=i: self._on_eq_slider_change(idx, val)
            )
            slider.set(0.0)
            slider.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
            self.eq_sliders.append(slider)

            # Band frequency name on bottom
            tk.Label(col_frame, text=name, bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MAIN, font=("Segoe UI", 8, "bold")).pack(side=tk.BOTTOM)

    def _build_filters_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" DIGITAL FILTER & AUDIO ROUTING ",
            bg=COLOR_PANEL_BG,
            fg=COLOR_CYAN,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
            bd=1,
            relief=tk.SOLID
        )
        frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        # 1. Filter Type Selector
        tk.Label(frame, text="Filter Type:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)
        filter_box = tk.Frame(frame, bg=COLOR_PANEL_BG)
        filter_box.pack(fill=tk.X, pady=2)

        modes = [("Bypass", "bypass"), ("Low-Pass", "lowpass"), ("High-Pass", "highpass"), ("Band-Pass", "bandpass"), ("Notch", "notch")]
        for text, val in modes:
            rb = tk.Radiobutton(
                filter_box,
                text=text,
                value=val,
                variable=self.filter_mode,
                command=self._on_filter_changed,
                bg=COLOR_PANEL_BG,
                fg=COLOR_TEXT_MAIN,
                selectcolor=COLOR_BG_DARK,
                activebackground=COLOR_PANEL_BG,
                activeforeground=COLOR_CYAN,
                font=("Segoe UI", 7)
            )
            rb.pack(side=tk.LEFT, padx=1)

        # Cutoff Frequency Slider
        fc_frame = tk.Frame(frame, bg=COLOR_PANEL_BG)
        fc_frame.pack(fill=tk.X, pady=4)

        tk.Label(fc_frame, text="Cutoff / Center Freq:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MAIN, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.lbl_cutoff_val = tk.Label(fc_frame, text="1000 Hz", bg=COLOR_PANEL_BG, fg=COLOR_AMBER, font=("Segoe UI", 8, "bold"))
        self.lbl_cutoff_val.pack(side=tk.RIGHT)

        # Slider mapped logarithmically (20 Hz - 18 kHz)
        self.slider_cutoff = tk.Scale(
            frame,
            from_=np.log10(20),
            to=np.log10(18000),
            resolution=0.01,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=COLOR_PANEL_BG,
            highlightthickness=0,
            troughcolor=COLOR_BG_DARK,
            command=self._on_cutoff_slider_change
        )
        self.slider_cutoff.set(np.log10(1000.0))
        self.slider_cutoff.pack(fill=tk.X, pady=(0, 4))

        # Q factor / Resonance
        q_frame = tk.Frame(frame, bg=COLOR_PANEL_BG)
        q_frame.pack(fill=tk.X, pady=2)
        tk.Label(q_frame, text="Q / Resonance:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MAIN, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.lbl_q_val = tk.Label(q_frame, text="1.00", bg=COLOR_PANEL_BG, fg=COLOR_AMBER, font=("Segoe UI", 8, "bold"))
        self.lbl_q_val.pack(side=tk.RIGHT)

        self.slider_q = tk.Scale(
            frame,
            from_=0.2,
            to=8.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=COLOR_PANEL_BG,
            highlightthickness=0,
            troughcolor=COLOR_BG_DARK,
            command=self._on_q_slider_change
        )
        self.slider_q.set(1.0)
        self.slider_q.pack(fill=tk.X, pady=(0, 6))

        # Divider line
        tk.Frame(frame, height=1, bg=COLOR_PANEL_BORDER).pack(fill=tk.X, pady=4)

        # Audio Source Selector
        tk.Label(frame, text="Audio Source Input:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(anchor=tk.W)
        source_frame = tk.Frame(frame, bg=COLOR_PANEL_BG)
        source_frame.pack(fill=tk.X, pady=2)

        sources = [("Demo Music", "demo"), ("Test Gen", "generator"), ("WAV File", "file"), ("Live Mic", "mic")]
        for text, val in sources:
            rb = tk.Radiobutton(
                source_frame,
                text=text,
                value=val,
                variable=self.selected_source,
                command=self._on_source_changed,
                bg=COLOR_PANEL_BG,
                fg=COLOR_TEXT_MAIN,
                selectcolor=COLOR_BG_DARK,
                activebackground=COLOR_PANEL_BG,
                font=("Segoe UI", 7)
            )
            rb.pack(side=tk.LEFT, padx=1)

        # Tone Generator Sub-controls
        self.gen_ctrls_frame = tk.Frame(frame, bg=COLOR_PANEL_BG)
        self.gen_ctrls_frame.pack(fill=tk.X, pady=2)

        tk.Label(self.gen_ctrls_frame, text="Tone:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 7)).pack(side=tk.LEFT)
        tones = ["Sine", "Sweep", "White Noise", "Pink Noise", "Chord"]
        self.combo_tones = ttk.Combobox(
            self.gen_ctrls_frame,
            values=tones,
            state="readonly",
            width=9,
            font=("Segoe UI", 7)
        )
        self.combo_tones.set("Sine")
        self.combo_tones.pack(side=tk.LEFT, padx=2)
        self.combo_tones.bind("<<ComboboxSelected>>", self._on_tone_selected)

        self.btn_load_wav = tk.Button(
            self.gen_ctrls_frame,
            text="Open WAV...",
            command=self._open_audio_file,
            bg="#334155",
            fg=COLOR_TEXT_MAIN,
            font=("Segoe UI", 7),
            relief=tk.FLAT,
            padx=4,
            pady=1,
            cursor="hand2"
        )
        self.btn_load_wav.pack(side=tk.RIGHT, padx=2)

    def _build_extensions_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text=" DSP EXTENSIONS & ANALYSIS ",
            bg=COLOR_PANEL_BG,
            fg=COLOR_CYAN,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=6,
            bd=1,
            relief=tk.SOLID
        )
        frame.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)

        # 1. Pitch Detector Card
        pitch_box = tk.Frame(frame, bg=COLOR_BG_DARK, padx=8, pady=6, bd=1, relief=tk.SOLID)
        pitch_box.pack(fill=tk.X, pady=4)

        tk.Label(pitch_box, text="REAL-TIME PITCH DETECTOR (YIN/AUTOCORR)", bg=COLOR_BG_DARK, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 7, "bold")).pack(anchor=tk.W)

        pitch_row = tk.Frame(pitch_box, bg=COLOR_BG_DARK)
        pitch_row.pack(fill=tk.X, pady=2)

        self.lbl_note = tk.Label(pitch_row, text="--", bg=COLOR_BG_DARK, fg=COLOR_GREEN, font=("Segoe UI", 18, "bold"), width=4)
        self.lbl_note.pack(side=tk.LEFT)

        pitch_info = tk.Frame(pitch_row, bg=COLOR_BG_DARK)
        pitch_info.pack(side=tk.LEFT, padx=8)

        self.lbl_pitch_hz = tk.Label(pitch_info, text="Freq: 0.0 Hz", bg=COLOR_BG_DARK, fg=COLOR_TEXT_MAIN, font=("Segoe UI", 8))
        self.lbl_pitch_hz.pack(anchor=tk.W)

        self.lbl_cents = tk.Label(pitch_info, text="Deviation: 0 cents", bg=COLOR_BG_DARK, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8))
        self.lbl_cents.pack(anchor=tk.W)

        # 2. Windowing Function (Slide 6 concept: Hann vs Hamming vs Rectangular)
        tk.Label(frame, text="FFT Windowing Function:", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, pady=(6, 2))
        win_frame = tk.Frame(frame, bg=COLOR_PANEL_BG)
        win_frame.pack(fill=tk.X)

        windows = [("Hann (Tapered)", "hann"), ("Hamming", "hamming"), ("Rectangular (Leakage)", "rectangular")]
        for text, val in windows:
            rb = tk.Radiobutton(
                win_frame,
                text=text,
                value=val,
                variable=self.selected_window,
                bg=COLOR_PANEL_BG,
                fg=COLOR_TEXT_MAIN,
                selectcolor=COLOR_BG_DARK,
                activebackground=COLOR_PANEL_BG,
                font=("Segoe UI", 7)
            )
            rb.pack(anchor=tk.W)

        # 3. Spectral Subtraction Noise Reduction (Slide 10)
        tk.Label(frame, text="Noise Reduction (Spectral Subtraction):", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, pady=(6, 2))

        nr_row = tk.Frame(frame, bg=COLOR_PANEL_BG)
        nr_row.pack(fill=tk.X)

        chk_nr = tk.Checkbutton(
            nr_row,
            text="Enable Denoiser",
            variable=self.nr_enabled_var,
            command=self._on_nr_toggle,
            bg=COLOR_PANEL_BG,
            fg=COLOR_TEXT_MAIN,
            selectcolor=COLOR_BG_DARK,
            activebackground=COLOR_PANEL_BG,
            font=("Segoe UI", 8)
        )
        chk_nr.pack(side=tk.LEFT)

        self.btn_calibrate_nr = tk.Button(
            nr_row,
            text="Calibrate Profile",
            command=self._calibrate_noise,
            bg="#334155",
            fg=COLOR_TEXT_MAIN,
            font=("Segoe UI", 7),
            relief=tk.FLAT,
            padx=4,
            pady=1,
            cursor="hand2"
        )
        self.btn_calibrate_nr.pack(side=tk.RIGHT)

    def _build_statusbar(self):
        status_frame = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=12, pady=4)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = tk.Label(
            status_frame,
            text="● STREAMING ACTIVE",
            bg=COLOR_BG_DARK,
            fg=COLOR_GREEN,
            font=("Segoe UI", 8, "bold")
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_device_info = tk.Label(
            status_frame,
            text=f"Sample Rate: {STANDARD_SAMPLE_RATE} Hz | Buffer: {DEFAULT_BLOCK_SIZE} samples (~46ms)",
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 8)
        )
        self.lbl_device_info.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------------

    def _start_audio(self):
        try:
            self.engine.start()
            self.lbl_status.config(text="● STREAMING ACTIVE", fg=COLOR_GREEN)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showwarning("Audio Notice", f"Audio device initialized with warning: {e}")

    def _stop_audio(self):
        self.engine.stop()
        self.lbl_status.config(text="■ AUDIO STOPPED", fg=COLOR_RED)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def _on_master_vol_change(self, val):
        vol_pct = float(val)
        self.lbl_vol_val.config(text=f"{int(vol_pct)}%")
        self.engine.set_master_gain(vol_pct / 100.0)

    def _on_eq_slider_change(self, band_idx: int, val: str):
        gain_db = float(val)
        sign = "+" if gain_db > 0 else ""
        self.eq_val_labels[band_idx].config(text=f"{sign}{gain_db:.1f} dB")
        self.engine.equalizer.set_band_gain(band_idx, gain_db)
        self._update_eq_theoretical_curve()

    def _on_preset_selected(self, event=None):
        preset_name = self.selected_preset.get()
        if preset_name in EQ_PRESETS:
            gains = EQ_PRESETS[preset_name]
            for i, g in enumerate(gains):
                self.eq_sliders[i].set(g)
                sign = "+" if g > 0 else ""
                self.eq_val_labels[i].config(text=f"{sign}{g:.1f} dB")
                self.engine.equalizer.set_band_gain(i, g)
            self._update_eq_theoretical_curve()

    def _reset_eq_flat(self):
        self.selected_preset.set("Flat")
        self._on_preset_selected()

    def _toggle_eq_bypass(self):
        self.engine.equalizer.enabled = not self.engine.equalizer.enabled
        if self.engine.equalizer.enabled:
            self.btn_bypass_eq.config(text="Bypass EQ: OFF", fg=COLOR_TEXT_MUTED)
        else:
            self.btn_bypass_eq.config(text="Bypass EQ: ON", fg=COLOR_RED)
        self._update_eq_theoretical_curve()

    def _on_filter_changed(self):
        mode = self.filter_mode.get()
        cutoff = float(self.filter_cutoff_var.get())
        q = float(self.filter_q_var.get())
        self.engine.filter.set_params(mode=mode, cutoff=cutoff, q=q)
        self._update_eq_theoretical_curve()

    def _on_cutoff_slider_change(self, val):
        fc = 10.0 ** float(val)
        self.filter_cutoff_var.set(fc)
        if fc >= 1000:
            self.lbl_cutoff_val.config(text=f"{fc/1000.0:.2f} kHz")
        else:
            self.lbl_cutoff_val.config(text=f"{int(fc)} Hz")
        self._on_filter_changed()

    def _on_q_slider_change(self, val):
        q = float(val)
        self.filter_q_var.set(q)
        self.lbl_q_val.config(text=f"{q:.2f}")
        self._on_filter_changed()

    def _on_source_changed(self):
        src = self.selected_source.get()
        self.engine.set_source(src)

    def _on_tone_selected(self, event=None):
        tone = self.combo_tones.get().lower().replace(" ", "_")
        self.engine.signal_gen.set_mode(tone)
        self.selected_source.set("generator")
        self.engine.set_source("generator")

    def _open_audio_file(self):
        path = filedialog.askopenfilename(
            title="Open Audio File",
            filetypes=[("Audio Files", "*.wav *.flac *.ogg"), ("All Files", "*.*")]
        )
        if path:
            success = self.engine.load_audio_file(path)
            if success:
                self.selected_source.set("file")
            else:
                messagebox.showerror("Error", "Could not load audio file. Ensure it is a valid WAV/FLAC/OGG.")

    def _on_nr_toggle(self):
        self.engine.noise_reducer.enabled = self.nr_enabled_var.get()

    def _calibrate_noise(self):
        self.engine.noise_reducer.start_calibration()
        self.lbl_status.config(text="● CALIBRATING NOISE PROFILE (0.5s)...", fg=COLOR_AMBER)
        self.root.after(600, lambda: self.lbl_status.config(text="● STREAMING ACTIVE", fg=COLOR_GREEN))

    def _update_eq_theoretical_curve(self):
        """Re-computes total combined frequency response of EQ + Filter for visual overlay."""
        eq_resp = self.engine.equalizer.get_frequency_response(self.f_eval)
        filt_resp = self.engine.filter.get_frequency_response(self.f_eval)
        total_resp = eq_resp + filt_resp
        self.line_eq_curve.set_ydata(total_resp)

    # ------------------------------------------------------------------------
    # REAL-TIME VISUALIZATION UPDATE LOOP (30-40 FPS)
    # ------------------------------------------------------------------------

    def _update_visualizer(self):
        if not self.is_updating:
            return

        data = self.engine.get_latest_visualization_data()
        if data is not None:
            raw_in = data['input']
            raw_out = data['output']
            win_type = self.selected_window.get()

            # 1. Update Waveform Oscilloscope
            self.line_wave_in.set_ydata(raw_in)
            self.line_wave_out.set_ydata(raw_out)

            # 2. Update Frequency Spectrum (FFT)
            freqs_in, mag_in, _, _ = compute_spectrum(raw_in, sample_rate=STANDARD_SAMPLE_RATE, window_type=win_type)
            freqs_out, mag_out, peak_f, peak_db = compute_spectrum(raw_out, sample_rate=STANDARD_SAMPLE_RATE, window_type=win_type)

            # Filter positive frequencies >= 20 Hz
            valid_mask = freqs_out >= 20.0
            f_plot = freqs_out[valid_mask]
            m_in_plot = mag_in[valid_mask]
            m_out_plot = mag_out[valid_mask]

            self.line_spec_in.set_data(f_plot, m_in_plot)
            self.line_spec_out.set_data(f_plot, m_out_plot)

            # Update Peak Frequency Marker
            if peak_db > -55.0 and peak_f >= 20.0:
                self.peak_point.set_data([peak_f], [peak_db])
                if peak_f >= 1000.0:
                    text_str = f" {peak_f/1000.0:.2f} kHz ({peak_db:.0f} dB)"
                else:
                    text_str = f" {peak_f:.0f} Hz ({peak_db:.0f} dB)"
                self.peak_text.set_position((peak_f, peak_db + 3.0))
                self.peak_text.set_text(text_str)
            else:
                self.peak_point.set_data([], [])
                self.peak_text.set_text("")

            # 3. Update Pitch Detection
            pitch_hz, note_name, cents = detect_pitch(raw_out, sample_rate=STANDARD_SAMPLE_RATE)
            if pitch_hz > 40.0:
                self.lbl_note.config(text=note_name, fg=COLOR_GREEN)
                self.lbl_pitch_hz.config(text=f"Freq: {pitch_hz:.1f} Hz")
                sign = "+" if cents >= 0 else ""
                self.lbl_cents.config(text=f"Dev: {sign}{cents:.1f} cents")
            else:
                self.lbl_note.config(text="--", fg=COLOR_TEXT_MUTED)
                self.lbl_pitch_hz.config(text="Freq: -- Hz")
                self.lbl_cents.config(text="Dev: --")

            # Redraw canvas
            self.canvas.draw_idle()

        # Schedule next frame (approx 33 ms -> 30 FPS)
        self.root.after(33, self._update_visualizer)

    def _launch_sender(self):
        """Launches Encrypted Sender application."""
        try:
            subprocess.Popen([sys.executable, "sender.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch sender: {e}")

    def _launch_receiver(self):
        """Launches Encrypted Receiver application."""
        try:
            subprocess.Popen([sys.executable, "receiver.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch receiver: {e}")

    def on_closing(self):
        self.is_updating = False
        self.engine.stop()
        self.root.destroy()
