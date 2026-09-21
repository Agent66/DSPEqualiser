"""
Full-duplex secure call UI (sender and receiver share this layout).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .call_engine import SecureCallEngine
from .dsp import EQ_CENTER_FREQUENCIES, EQ_BAND_NAMES, EQ_PRESETS, compute_spectrum
from .theme import (
    COLOR_BG_DARK,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_PANEL_ELEV,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_CYAN,
    COLOR_BLUE,
    COLOR_AMBER,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_PTT,
    COLOR_PTT_ACTIVE,
    apply_ttk_theme,
    card,
    pill_button,
    stat_tile,
    LevelMeter,
)


class SecureCallGUI:
    def __init__(self, root: tk.Tk, engine: SecureCallEngine, title: str, accent: str = COLOR_CYAN):
        self.root = root
        self.engine = engine
        self.accent = accent
        self.root.title(title)
        self.root.geometry("1240x860")
        self.root.minsize(1040, 720)
        self.root.configure(bg=COLOR_BG_DARK)
        apply_ttk_theme(root)

        self.selected_source = tk.StringVar(value=self.engine.source_mode)
        self.selected_preset = tk.StringVar(value="Flat")
        self.monitor_var = tk.BooleanVar(value=self.engine.monitor_local_audio)
        self.vad_var = tk.BooleanVar(value=self.engine.vad_enabled)
        self.ptt_var = tk.BooleanVar(value=self.engine.ptt_mode)
        self.csv_var = tk.BooleanVar(value=True)
        self.filter_mode_var = tk.StringVar(value="bypass")
        self.filter_cutoff_var = tk.DoubleVar(value=1000.0)
        self.rx_vol_var = tk.DoubleVar(value=100.0)
        self.tamper_attack_var = tk.BooleanVar(value=False)
        self.ptt_latched = False

        self.f_eval = np.logspace(np.log10(20), np.log10(self.engine.sample_rate / 2.0 - 50.0), 200)

        self._build_ui()
        self._bind_keys()
        self.engine.start()

        self.is_updating = True
        self.root.after(30, self._update_loop)

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=14, pady=4)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=COLOR_BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_plots(left)
        self._build_eq_panel(left)

        right = tk.Frame(body, bg=COLOR_BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_side_panel(right)

        self._build_statusbar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=18, pady=12)
        header.pack(fill=tk.X)

        title_box = tk.Frame(header, bg=COLOR_BG_DARK)
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            title_box,
            text="SPEQUALISER  ·  SECURE FULL-DUPLEX CALL",
            font=("Segoe UI", 15, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK,
        ).pack(anchor=tk.W)

        peer = f"{self.engine.link.peer_host}:{self.engine.link.peer_port}"
        local = self.engine.link.local_port
        self.lbl_sub = tk.Label(
            title_box,
            text=f"Listen :{local}  →  Peer {peer}   ·   X25519 → AES-256-GCM   ·   VAD / DTX / PTT",
            font=("Segoe UI", 9),
            fg=self.accent,
            bg=COLOR_BG_DARK,
        )
        self.lbl_sub.pack(anchor=tk.W, pady=(2, 0))

        ctrl = tk.Frame(header, bg=COLOR_BG_DARK)
        ctrl.pack(side=tk.RIGHT)

        tk.Label(ctrl, text="Source", fg=COLOR_TEXT_MUTED, bg=COLOR_BG_DARK, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 6))
        src = ttk.Combobox(
            ctrl,
            textvariable=self.selected_source,
            values=["mic", "demo", "generator"],
            state="readonly",
            width=11,
        )
        src.pack(side=tk.LEFT, padx=(0, 10))
        src.bind("<<ComboboxSelected>>", lambda e: self.engine.set_source(self.selected_source.get()))

        self._toggle(ctrl, "CSV log", self.csv_var, self._on_csv_toggle)
        self._toggle(ctrl, "Monitor", self.monitor_var, self._on_monitor_toggle)

    def _toggle(self, parent, text, var, cmd):
        chk = tk.Checkbutton(
            parent,
            text=text,
            variable=var,
            command=cmd,
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG_DARK,
            selectcolor=COLOR_PANEL_ELEV,
            activebackground=COLOR_BG_DARK,
            activeforeground=COLOR_CYAN,
            font=("Segoe UI", 8, "bold"),
            bd=0,
            highlightthickness=0,
        )
        chk.pack(side=tk.LEFT, padx=4)

    def _build_plots(self, parent):
        plot_card = tk.Frame(parent, bg=COLOR_PANEL_BG, highlightbackground=COLOR_PANEL_BORDER, highlightthickness=1)
        plot_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.fig = Figure(figsize=(8.4, 4.0), facecolor=COLOR_PANEL_BG)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.14, hspace=0.42)

        self.ax_wave = self.fig.add_subplot(2, 1, 1)
        self._style_axis(self.ax_wave, "TX / RX waveform", self.accent)
        self.ax_wave.set_ylim(-1.1, 1.1)
        self.ax_wave.set_xlim(0, self.engine.block_size)
        t = np.arange(self.engine.block_size)
        self.line_tx, = self.ax_wave.plot(t, np.zeros(self.engine.block_size), color=COLOR_BLUE, lw=1.15, label="TX")
        self.line_rx, = self.ax_wave.plot(t, np.zeros(self.engine.block_size), color=COLOR_GREEN, lw=1.0, alpha=0.85, label="RX")
        self.ax_wave.legend(loc="upper right", fontsize=7, facecolor=COLOR_PANEL_BG, edgecolor=COLOR_PANEL_BORDER, labelcolor=COLOR_TEXT_MAIN)

        self.ax_spec = self.fig.add_subplot(2, 1, 2)
        self._style_axis(self.ax_spec, "Pre-encrypt spectrum (Hann FFT)", COLOR_CYAN)
        self.ax_spec.set_xscale("log")
        self.ax_spec.set_xlim(20, self.engine.sample_rate / 2.0)
        self.ax_spec.set_ylim(-80, 10)
        dummy = np.fft.rfftfreq(self.engine.block_size, d=1.0 / self.engine.sample_rate)
        self.line_spec, = self.ax_spec.plot(dummy, np.full_like(dummy, -80.0), color=COLOR_CYAN, lw=1.2)
        self.line_eq_curve, = self.ax_spec.plot(self.f_eval, np.zeros_like(self.f_eval), color=COLOR_AMBER, lw=1.0, ls="--", alpha=0.8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_axis(self, ax, title, color):
        ax.set_facecolor(COLOR_BG_DARK)
        ax.set_title(title, fontsize=9, color=color, pad=4, loc="left", fontweight="bold")
        ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=7)
        ax.grid(True, color="#1E293B", linestyle="--", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(COLOR_PANEL_BORDER)

    def _build_eq_panel(self, parent):
        eq = card(parent, "DSP  ·  FILTER & 8-BAND EQUALIZER")
        eq.pack(fill=tk.X)
        inner = eq.inner

        top = tk.Frame(inner, bg=COLOR_PANEL_BG)
        top.pack(fill=tk.X, pady=(0, 4))

        tk.Label(top, text="Filter", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        filt = ttk.Combobox(
            top,
            textvariable=self.filter_mode_var,
            values=["bypass", "lowpass", "highpass", "bandpass", "notch"],
            state="readonly",
            width=10,
        )
        filt.pack(side=tk.LEFT, padx=(6, 10))
        filt.bind("<<ComboboxSelected>>", self._on_filter_change)

        self.lbl_cutoff = tk.Label(top, text="1000 Hz", bg=COLOR_PANEL_BG, fg=COLOR_AMBER, font=("Segoe UI", 8, "bold"))
        self.lbl_cutoff.pack(side=tk.LEFT)
        tk.Scale(
            top,
            from_=60.0,
            to=12000.0,
            resolution=50.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.filter_cutoff_var,
            length=120,
            bg=COLOR_PANEL_BG,
            fg=COLOR_TEXT_MAIN,
            troughcolor=COLOR_BG_DARK,
            activebackground=COLOR_CYAN,
            highlightthickness=0,
            bd=0,
            command=self._on_filter_cutoff_change,
        ).pack(side=tk.LEFT, padx=8)

        tk.Label(top, text="Preset", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 4))
        preset = ttk.Combobox(top, textvariable=self.selected_preset, values=list(EQ_PRESETS.keys()), state="readonly", width=12)
        preset.pack(side=tk.LEFT)
        preset.bind("<<ComboboxSelected>>", self._on_preset_change)

        pill_button(top, "Reset EQ", self._reset_eq, COLOR_PANEL_ELEV, COLOR_TEXT_MAIN, padx=8, pady=3, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=8)

        faders = tk.Frame(inner, bg=COLOR_PANEL_BG)
        faders.pack(fill=tk.X)
        self.eq_sliders = []
        self.eq_labels = []
        for i, name in enumerate(EQ_BAND_NAMES):
            col = tk.Frame(faders, bg=COLOR_PANEL_BG)
            col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            val = tk.Label(col, text="0.0 dB", font=("Segoe UI", 7, "bold"), fg=COLOR_AMBER, bg=COLOR_PANEL_BG)
            val.pack()
            self.eq_labels.append(val)
            slider = tk.Scale(
                col,
                from_=12.0,
                to=-12.0,
                resolution=0.5,
                orient=tk.VERTICAL,
                showvalue=False,
                length=78,
                bg=COLOR_BG_DARK,
                fg=COLOR_TEXT_MAIN,
                troughcolor=COLOR_PANEL_ELEV,
                activebackground=COLOR_CYAN,
                highlightthickness=0,
                bd=0,
                command=lambda v, idx=i: self._on_slider_change(idx, v),
            )
            slider.set(0.0)
            slider.pack(pady=2)
            self.eq_sliders.append(slider)
            tk.Label(col, text=name, font=("Segoe UI", 8), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack()

    def _build_side_panel(self, parent):
        dh_card = card(parent, "SESSION KEY")
        dh_card.pack(fill=tk.X, pady=(0, 8))
        self.lbl_dh = tk.Label(
            dh_card.inner,
            text="Waiting for X25519 handshake…",
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_AMBER,
            bg=COLOR_PANEL_BG,
            wraplength=240,
            justify=tk.LEFT,
        )
        self.lbl_dh.pack(anchor=tk.W)
        self.lbl_dh_detail = tk.Label(
            dh_card.inner,
            text="Public keys are exchanged over UDP. AES-256-GCM starts after HKDF.",
            font=("Segoe UI", 8),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            wraplength=240,
            justify=tk.LEFT,
        )
        self.lbl_dh_detail.pack(anchor=tk.W, pady=(4, 0))

        tx_card = card(parent, "TRANSMIT MODE")
        tx_card.pack(fill=tk.X, pady=(0, 8))

        modes = tk.Frame(tx_card.inner, bg=COLOR_PANEL_BG)
        modes.pack(fill=tk.X)
        self._side_check(modes, "Smart VAD / DTX", self.vad_var, self._on_vad_toggle)
        self._side_check(modes, "Push-to-talk", self.ptt_var, self._on_ptt_toggle)

        self.btn_ptt = tk.Button(
            tx_card.inner,
            text="HOLD TO TALK\n(SPACE)",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_PTT,
            fg="#FFFFFF",
            activebackground=COLOR_PTT_ACTIVE,
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            bd=0,
            height=3,
            cursor="hand2",
        )
        self.btn_ptt.pack(fill=tk.X, pady=(10, 4))
        self.btn_ptt.bind("<ButtonPress-1>", lambda e: self._ptt_down())
        self.btn_ptt.bind("<ButtonRelease-1>", lambda e: self._ptt_up())

        self.lbl_tx_state = tk.Label(
            tx_card.inner,
            text="Continuous transmit",
            font=("Segoe UI", 8),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
        )
        self.lbl_tx_state.pack(anchor=tk.W)

        meters = tk.Frame(tx_card.inner, bg=COLOR_PANEL_BG)
        meters.pack(fill=tk.X, pady=(8, 0))
        tk.Label(meters, text="TX", bg=COLOR_PANEL_BG, fg=COLOR_BLUE, font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)
        self.meter_tx = LevelMeter(meters, width=16, height=56)
        self.meter_tx.pack(side=tk.LEFT, padx=6)
        tk.Label(meters, text="RX", bg=COLOR_PANEL_BG, fg=COLOR_GREEN, font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)
        self.meter_rx = LevelMeter(meters, width=16, height=56)
        self.meter_rx.pack(side=tk.LEFT, padx=6)

        vol = tk.Frame(tx_card.inner, bg=COLOR_PANEL_BG)
        vol.pack(fill=tk.X, pady=(8, 0))
        tk.Label(vol, text="RX volume", bg=COLOR_PANEL_BG, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Scale(
            vol,
            from_=0.0,
            to=150.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=110,
            bg=COLOR_PANEL_BG,
            troughcolor=COLOR_BG_DARK,
            highlightthickness=0,
            bd=0,
            variable=self.rx_vol_var,
            command=lambda v: self.engine.set_rx_volume(float(v) / 100.0),
        ).pack(side=tk.LEFT, padx=6)

        stats = card(parent, "LINK TELEMETRY")
        stats.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        grid = tk.Frame(stats.inner, bg=COLOR_PANEL_BG)
        grid.pack(fill=tk.BOTH, expand=True)
        tiles = [
            ("TX RATE", "0.0 kbps", COLOR_CYAN),
            ("RX RATE", "0.0 kbps", COLOR_GREEN),
            ("LATENCY", "— ms", COLOR_AMBER),
            ("RX OK", "0", COLOR_GREEN),
            ("LOST", "0", COLOR_AMBER),
            ("TAMPER", "0", COLOR_RED),
        ]
        self.stat_labels = {}
        for i, (cap, init, col) in enumerate(tiles):
            fr, lbl = stat_tile(grid, cap, init, col)
            fr.grid(row=i // 2, column=i % 2, sticky="nsew", padx=3, pady=3)
            self.stat_labels[cap] = lbl
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.lbl_csv = tk.Label(
            stats.inner,
            text=f"CSV → {self.engine.csv.path}",
            font=("Segoe UI", 7),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            wraplength=240,
            justify=tk.LEFT,
        )
        self.lbl_csv.pack(anchor=tk.W, pady=(6, 0))

        attack = tk.Frame(parent, bg=COLOR_BG_DARK)
        attack.pack(fill=tk.X)
        tk.Checkbutton(
            attack,
            text="Simulate tamper attack",
            variable=self.tamper_attack_var,
            command=self._on_toggle_tamper,
            fg=COLOR_RED,
            bg=COLOR_BG_DARK,
            selectcolor=COLOR_PANEL_ELEV,
            activebackground=COLOR_BG_DARK,
            activeforeground=COLOR_RED,
            font=("Segoe UI", 8, "bold"),
            bd=0,
            highlightthickness=0,
        ).pack(anchor=tk.W)

    def _side_check(self, parent, text, var, cmd):
        tk.Checkbutton(
            parent,
            text=text,
            variable=var,
            command=cmd,
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL_BG,
            selectcolor=COLOR_PANEL_ELEV,
            activebackground=COLOR_PANEL_BG,
            activeforeground=COLOR_CYAN,
            font=("Segoe UI", 9),
            bd=0,
            highlightthickness=0,
        ).pack(anchor=tk.W)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=COLOR_PANEL_ELEV, padx=16, pady=8)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_sec = tk.Label(bar, text="●  CONNECTING", font=("Segoe UI", 9, "bold"), fg=COLOR_AMBER, bg=COLOR_PANEL_ELEV)
        self.lbl_sec.pack(side=tk.LEFT)
        self.lbl_vad = tk.Label(bar, text="DTX off", font=("Segoe UI", 8, "bold"), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_ELEV)
        self.lbl_vad.pack(side=tk.LEFT, padx=16)
        self.lbl_stats = tk.Label(bar, text="", font=("Segoe UI", 8), fg=COLOR_TEXT_MAIN, bg=COLOR_PANEL_ELEV)
        self.lbl_stats.pack(side=tk.LEFT)
        self.lbl_dest = tk.Label(
            bar,
            text=f":{self.engine.link.local_port}  →  {self.engine.link.peer_host}:{self.engine.link.peer_port}",
            font=("Segoe UI", 8, "bold"),
            fg=self.accent,
            bg=COLOR_PANEL_ELEV,
        )
        self.lbl_dest.pack(side=tk.RIGHT)

    def _bind_keys(self):
        self.root.bind("<KeyPress-space>", self._on_space_down)
        self.root.bind("<KeyRelease-space>", self._on_space_up)
        self.root.bind("<space>", lambda e: "break")

    def _on_space_down(self, event):
        if self.engine.ptt_mode:
            self._ptt_down()
        return "break"

    def _on_space_up(self, event):
        if self.engine.ptt_mode:
            self._ptt_up()
        return "break"

    def _ptt_down(self):
        self.engine.set_ptt_held(True)
        self.btn_ptt.config(bg=COLOR_PTT_ACTIVE, text="TRANSMITTING")

    def _ptt_up(self):
        self.engine.set_ptt_held(False)
        self.btn_ptt.config(bg=COLOR_PTT, text="HOLD TO TALK\n(SPACE)")

    def _on_vad_toggle(self):
        self.engine.set_vad_enabled(self.vad_var.get())

    def _on_ptt_toggle(self):
        self.engine.set_ptt_mode(self.ptt_var.get())
        if not self.ptt_var.get():
            self._ptt_up()

    def _on_csv_toggle(self):
        self.engine.set_csv_logging(self.csv_var.get())

    def _on_monitor_toggle(self):
        self.engine.monitor_local_audio = self.monitor_var.get()

    def _on_toggle_tamper(self):
        self.engine.link.simulate_tampering = self.tamper_attack_var.get()

    def _on_filter_change(self, event=None):
        self.engine.set_filter_params(mode=self.filter_mode_var.get(), cutoff=float(self.filter_cutoff_var.get()))

    def _on_filter_cutoff_change(self, val):
        cutoff = float(val)
        self.lbl_cutoff.config(text=f"{cutoff:.0f} Hz")
        self.engine.set_filter_params(mode=self.filter_mode_var.get(), cutoff=cutoff)

    def _on_slider_change(self, band_idx: int, val: str):
        gain_db = float(val)
        self.engine.equalizer.set_band_gain(band_idx, gain_db)
        self.eq_labels[band_idx].config(text=f"{gain_db:+.1f} dB")
        self.line_eq_curve.set_ydata(self.engine.equalizer.get_frequency_response(self.f_eval))

    def _on_preset_change(self, event=None):
        gains = EQ_PRESETS.get(self.selected_preset.get())
        if not gains:
            return
        for i, g in enumerate(gains):
            self.eq_sliders[i].set(g)
            self.engine.equalizer.set_band_gain(i, g)
            self.eq_labels[i].config(text=f"{g:+.1f} dB")
        self.line_eq_curve.set_ydata(self.engine.equalizer.get_frequency_response(self.f_eval))

    def _reset_eq(self):
        self.selected_preset.set("Flat")
        self._on_preset_change()

    def _update_loop(self):
        if not self.is_updating:
            return

        viz = self.engine.get_latest_visualization_data()
        if viz is not None:
            tx = viz["output"]
            self.line_tx.set_ydata(tx)
            rx = viz.get("rx")
            if rx is not None and len(rx) == len(tx):
                self.line_rx.set_ydata(rx)
            freqs, mag_db, _, _ = compute_spectrum(tx, sample_rate=self.engine.sample_rate, window_type="hann")
            self.line_spec.set_data(freqs, mag_db)
            self.canvas.draw_idle()

            self.meter_tx.set_level(float(np.sqrt(np.mean(np.square(tx)))) * 3.5)
            if rx is not None and len(rx):
                self.meter_rx.set_level(float(np.sqrt(np.mean(np.square(rx)))) * 3.5)

            if viz.get("ptt_mode"):
                if viz.get("ptt_held"):
                    self.lbl_tx_state.config(text="PTT held — gate open", fg=COLOR_GREEN)
                else:
                    self.lbl_tx_state.config(text="PTT idle — not sending", fg=COLOR_AMBER)
            elif viz.get("vad_enabled"):
                if viz.get("transmitting"):
                    self.lbl_tx_state.config(
                        text=f"VAD speech  {viz['energy_db']:.0f} dB  ZCR {viz['zcr']:.2f}",
                        fg=COLOR_GREEN,
                    )
                else:
                    self.lbl_tx_state.config(
                        text=f"VAD silence — saved {viz.get('savings_pct', 0):.1f}%",
                        fg=COLOR_AMBER,
                    )
            else:
                self.lbl_tx_state.config(text="Always sending (no DTX)", fg=COLOR_TEXT_MUTED)

            if viz.get("vad_enabled"):
                if viz.get("transmitting"):
                    self.lbl_vad.config(text="VAD  SPEECH", fg=COLOR_GREEN)
                else:
                    self.lbl_vad.config(text=f"VAD  SILENCE  −{viz.get('savings_pct', 0):.0f}% BW", fg=COLOR_AMBER)
            else:
                self.lbl_vad.config(text="DTX off", fg=COLOR_TEXT_MUTED)

        net = self.engine.link.get_stats()
        self.stat_labels["TX RATE"].config(text=f"{net['tx_kbps']:.1f} kbps")
        self.stat_labels["RX RATE"].config(text=f"{net['rx_kbps']:.1f} kbps")
        lat = net.get("latency_ms", 0.0)
        self.stat_labels["LATENCY"].config(text="— ms" if lat <= 0 else f"{lat:.1f} ms")
        self.stat_labels["RX OK"].config(text=f"{net['packets_ok']:,}")
        self.stat_labels["LOST"].config(text=f"{net['packets_lost']:,}")
        self.stat_labels["TAMPER"].config(text=f"{net['packets_tampered']:,}")

        dh = net.get("dh_state", "")
        if dh == "established":
            self.lbl_dh.config(text="Session key established", fg=COLOR_GREEN)
            self.lbl_dh_detail.config(text="X25519 ECDH + HKDF-SHA256 → AES-256-GCM")
            self.lbl_sec.config(text="●  SECURE CALL ACTIVE", fg=COLOR_GREEN)
        elif dh == "psk":
            self.lbl_dh.config(text="Pre-shared key mode", fg=COLOR_CYAN)
            self.lbl_sec.config(text="●  PSK STREAM ACTIVE", fg=COLOR_CYAN)
        else:
            self.lbl_dh.config(text="Waiting for X25519 handshake…", fg=COLOR_AMBER)
            self.lbl_sec.config(text="●  EXCHANGING KEYS", fg=COLOR_AMBER)

        self.lbl_stats.config(
            text=f"TX {net['packets_sent']:,} pkts  ·  RX {net['packets_received']:,}  ·  jitter {net['jitter_ms']:.0f} ms"
        )
        self.root.after(35, self._update_loop)

    def on_closing(self):
        self.is_updating = False
        self.engine.stop()
        self.root.destroy()
