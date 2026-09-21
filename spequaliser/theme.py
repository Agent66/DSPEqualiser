"""
Shared visual theme helpers for SpEqualiser Tkinter windows.
"""

import tkinter as tk
from tkinter import ttk

COLOR_BG_DARK = "#070B16"
COLOR_PANEL_BG = "#10192C"
COLOR_PANEL_ELEV = "#162038"
COLOR_PANEL_BORDER = "#243353"
COLOR_TEXT_MAIN = "#F1F5F9"
COLOR_TEXT_MUTED = "#8BA0B8"
COLOR_CYAN = "#22D3EE"
COLOR_BLUE = "#60A5FA"
COLOR_AMBER = "#FBBF24"
COLOR_GREEN = "#34D399"
COLOR_RED = "#F87171"
COLOR_PURPLE = "#A78BFA"
COLOR_PTT = "#EF4444"
COLOR_PTT_ACTIVE = "#22C55E"


def apply_ttk_theme(root: tk.Tk) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=COLOR_BG_DARK, foreground=COLOR_TEXT_MAIN)
    style.configure("TFrame", background=COLOR_BG_DARK)
    style.configure("Card.TFrame", background=COLOR_PANEL_BG)
    style.configure("TLabel", background=COLOR_PANEL_BG, foreground=COLOR_TEXT_MAIN, font=("Segoe UI", 9))
    style.configure(
        "TCombobox",
        fieldbackground=COLOR_PANEL_ELEV,
        background=COLOR_PANEL_ELEV,
        foreground=COLOR_TEXT_MAIN,
        arrowcolor=COLOR_CYAN,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", COLOR_PANEL_ELEV)],
        foreground=[("readonly", COLOR_TEXT_MAIN)],
        selectbackground=[("readonly", COLOR_PANEL_BORDER)],
    )
    return style


def card(parent, title: str | None = None) -> tk.Frame:
    wrap = tk.Frame(parent, bg=COLOR_PANEL_BG, highlightbackground=COLOR_PANEL_BORDER, highlightthickness=1, bd=0)
    inner = tk.Frame(wrap, bg=COLOR_PANEL_BG, padx=12, pady=10)
    inner.pack(fill=tk.BOTH, expand=True)
    if title:
        tk.Label(
            inner,
            text=title,
            bg=COLOR_PANEL_BG,
            fg=COLOR_CYAN,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor=tk.W, pady=(0, 6))
    wrap.inner = inner
    return wrap


def pill_button(parent, text, command, bg, fg="#FFFFFF", padx=14, pady=8, font=("Segoe UI", 9, "bold")):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=bg,
        activeforeground=fg,
        font=font,
        padx=padx,
        pady=pady,
        relief=tk.FLAT,
        bd=0,
        cursor="hand2",
        highlightthickness=0,
    )
    return btn


def stat_tile(parent, caption: str, initial: str, color: str):
    frame = tk.Frame(parent, bg=COLOR_PANEL_ELEV, padx=10, pady=8)
    tk.Label(frame, text=caption, font=("Segoe UI", 7, "bold"), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_ELEV).pack(anchor=tk.W)
    value = tk.Label(frame, text=initial, font=("Segoe UI", 13, "bold"), fg=color, bg=COLOR_PANEL_ELEV)
    value.pack(anchor=tk.W, pady=(2, 0))
    return frame, value


class LevelMeter(tk.Canvas):
    def __init__(self, parent, width=14, height=72, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=COLOR_BG_DARK,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._w = width
        self._h = height

    def set_level(self, rms: float):
        rms = max(0.0, min(1.0, float(rms)))
        self.delete("all")
        self.create_rectangle(0, 0, self._w, self._h, fill=COLOR_BG_DARK, outline=COLOR_PANEL_BORDER)
        fill_h = int(self._h * rms)
        y0 = self._h - fill_h
        color = COLOR_GREEN if rms < 0.75 else (COLOR_AMBER if rms < 0.92 else COLOR_RED)
        if fill_h > 0:
            self.create_rectangle(2, y0, self._w - 2, self._h - 1, fill=color, outline="")
