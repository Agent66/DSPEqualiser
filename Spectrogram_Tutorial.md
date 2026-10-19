# How to Read an Audio Spectrogram: A Complete Visual Guide
### Time, Frequency, and Amplitude in Real-Time Digital Signal Processing

---

## 1. What is a Spectrogram?

A **spectrogram** is a visual representation of the spectrum of frequencies of a signal as it varies with time. While a standard waveform oscilloscope shows **amplitude vs. time**, and a standard FFT spectrum shows **amplitude vs. frequency**, a spectrogram combines all three dimensions into a single continuous **time-frequency-energy map**:

```
Y-Axis: Frequency (Hz) ──► Pitch / Tone (Bass at bottom, Treble at top)
X-Axis: Time (Seconds) ──► Progression (Past on left, Present on right)
Color:  Magnitude (dB) ──► Energy / Loudness (Dark = Silence, Bright = Loud)
```

In the **DSP Equalizer Suite**, the scrolling spectrogram is a real-time **waterfall display** running at 30–40 FPS. The rightmost edge represents the **current instant ($t = 0\text{ s}$)**, and audio continuously scrolls left into the past (**up to $-4.0\text{ s}$ ago**).

---

## 2. Understanding the Three Dimensions

```
Frequency (Hz)
 ▲
 │  ┌─────────────────────────────────────────────────────────────┐ 0 dB (Loudest: Yellow/White)
12k│  │                                 ░░░░░░░ [Hi-Hat / Fricatives]│
 │  │                                                             │
4k │  │                   ═════════════════════ [Vocal Formant F2]│
 │  │      ══════════════════════════════════ [Vocal Formant F1]│
1k │  │                                                             │
 │  │ ─── ─── ─── ─── ─── ─── ─── ─── ─── ─── ── [50 Hz Mains Hum]│
20 │  └─────────────────────────────────────────────────────────────┘ -80 dB (Noise Floor: Black/Purple)
 0 └──┴─────────────────────────────────────────────────────────────►
     -4.0s                      -2.0s                        0.0s (Live)
                              Time (Seconds)
```

### 1. The Horizontal Axis (X-Axis): Time
- **Direction**: Audio scrolls from **right to left**.
- **Live Edge ($0.0\text{ s}$)**: What is happening right now in your microphone or demo track.
- **History ($-4.0\text{ s}$)**: What happened four seconds ago.
- **Reading Duration**: The width of a visual block tells you how long a sound lasted. A brief drum hit appears as a thin vertical line, while a held vowel forms a sustained horizontal band.

### 2. The Vertical Axis (Y-Axis): Frequency
- **Units**: Hertz (Hz) and kilohertz (kHz), corresponding to musical pitch.
- **Low Frequencies ($20\text{ Hz} - 250\text{ Hz}$)**: Located at the bottom. This contains kick drums, bass guitar, acoustic room rumble, and fundamental mains hum ($50\text{ Hz}$).
- **Mid Frequencies ($250\text{ Hz} - 4\text{ kHz}$)**: The human vocal range, instrument fundamentals, and speech formants ($F_1, F_2, F_3$).
- **High Frequencies ($4\text{ kHz} - 16\text{ kHz}$)**: Cymbals, vocal sibilance ("s", "sh", "f"), breath, and brightness harmonics.

### 3. The Color Scale (Z-Axis): Magnitude / Energy (dBFS)
Colors represent signal power in decibels relative to full scale (**dBFS**):

| Color (Magma Palette) | Intensity (dBFS) | Meaning in Audio |
|---|---|---|
| **Deep Purple / Black** | $-80\text{ dB}$ to $-60\text{ dB}$ | Complete silence or background acoustic noise floor |
| **Dark Red / Magenta** | $-60\text{ dB}$ to $-35\text{ dB}$ | Quiet ambient sound, reverberant decay tails |
| **Orange / Amber** | $-35\text{ dB}$ to $-15\text{ dB}$ | Moderate level speech, instrument accompaniment |
| **Bright Yellow / White** | $-15\text{ dB}$ to $0\text{ dB}$ | Peak signal energy, vowel formants, sharp transients |

---

## 3. How to Identify Common Audio Signatures

### 1. Pure Sine Wave
- **Visual Pattern**: A single, razor-sharp horizontal line.
- **Interpretation**: A single isolated frequency without any overtones. If you switch the audio source to **Signal Generator** with a $440\text{ Hz}$ tone, you will see a solitary glowing amber line at exactly $440\text{ Hz}$.

```
Frequency
  ▲
  │
440 Hz ────────────────────────────────────────── [Single Constant Line]
  │
  0 ─────────────────────────────────────────────► Time
```

---

### 2. Mains Hum (50 Hz Indian/European & Harmonics)
- **Visual Pattern**: Steady, unwavering parallel horizontal lines at exact integer multiples:
  - Fundamental: **$50\text{ Hz}$** (faint or prominent depending on pickup)
  - 2nd Harmonic: **$100\text{ Hz}$** (often the strongest hum component)
  - 3rd Harmonic: **$150\text{ Hz}$**
- **In This App**: Click **⚡ Inject 50 Hz Hum** to immediately see these persistent parallel bars appear. Toggle **Auto Hum Removal** and watch the notch filter carve them out in real time!

```
Frequency
  ▲
150 Hz ------------------------------------------ [3rd Harmonic]
100 Hz ========================================== [2nd Harmonic: Strongest]
 50 Hz ------------------------------------------ [Fundamental Hum]
  0 ─────────────────────────────────────────────► Time
```

---

### 3. The Human Voice (Voiced Speech vs. Unvoiced Speech)

Human speech is split into two distinct acoustic categories:

#### A. Voiced Sounds (Vowels: "Ah", "Ee", "Oo", "Mm")
- **Visual Pattern**: A **harmonic stack** (like horizontal ladder rungs) with dense bright energy concentrations called **formants**.
- **Fundamental Frequency ($F_0$)**: The bottom-most bar ($100 - 150\text{ Hz}$ for adult males, $180 - 250\text{ Hz}$ for adult females).
- **Harmonics ($2F_0, 3F_0, 4F_0, \dots$)**: Parallel bars stacked evenly above the fundamental.
- **Formants ($F_1, F_2, F_3$)**: The thicker, brighter bands where the mouth and throat resonate (e.g., $F_1 \approx 700\text{ Hz}, F_2 \approx 1200\text{ Hz}$ for "Ah").

```
Frequency
  ▲
2400 Hz ──── ──── ────                            [Formant F3]
1800 Hz ══════════════                            [Formant F2: Resonated]
1200 Hz ──── ──── ────
 600 Hz ══════════════                            [Formant F1: Strong Energy]
 200 Hz ──────────────                            [Pitch Fundamental F0]
  0 ─────────────────────────────────────────────► Time
```

#### B. Unvoiced Sounds (Consonants: "S", "Sh", "T", "Ch", "F")
- **Visual Pattern**: A cloudy, diffuse vertical patch of noise concentrated in the high frequencies ($3\text{ kHz} - 8\text{ kHz}$), with **no horizontal harmonic lines**.
- **Why**: These sounds are produced by turbulent air flowing through teeth and lips without vocal cord vibration.

```
Frequency
  ▲
 8 kHz      ░░░░░░░░░░░░░                         [Fricative "S" Noise Cloud]
 4 kHz      ░░░░░░░░░░░░░                         [No horizontal bars]
  0 ─────────────────────────────────────────────► Time
```

#### C. Vocal Pitch Glide / Vibrato
- When a singer modulates pitch or slides between notes, the horizontal harmonic lines curve upward or downward like waves.

---

### 4. Percussion, Clicks, and Attacks (Transients)
- **Visual Pattern**: A **solid vertical streak** spanning across all frequencies simultaneously.
- **Kick Drum**: A vertical line with intense bright yellow energy concentrated below $120\text{ Hz}$.
- **Snare Drum**: A sharp vertical stripe followed by a wideband cloud of noise (the snare wires) between $1\text{ kHz}$ and $6\text{ kHz}$.
- **Finger Snap / Mic Tap / Glitch**: An ultra-thin vertical spike spanning from $20\text{ Hz}$ to $12\text{ kHz}$.

```
Frequency
  ▲
12 kHz    │     │                                [Wideband Clicks / Snaps]
 4 kHz    │     │
 1 kHz    │     │
100 Hz    █     │                                [Kick Drum Low-End Weight]
  0 ──────┴─────┴────────────────────────────────► Time
```

---

### 5. Ambient Background Noise
- **Visual Pattern**: A continuous, grainy "fog" or speckled texture across the entire background, especially between $100\text{ Hz}$ and $4\text{ kHz}$.
- **In This App**: Click **⚡ Inject Noise** to simulate ambient room hiss. Then click **🎙️ Record 1s Noise Sample** to activate **Spectral Subtraction** — you will immediately see the dark background clear up, leaving only clean voice signals!

---

## 4. How to Use the Spectrogram Controls in This App

The GUI in `spequaliser/gui.py` provides dedicated controls to make spectrogram inspection intuitive:

### 1. Focus Frequency Modes
Located in the plot toolbar at the top of the window:
- **🎙️ Vocal Formants (20 Hz – 3.6 kHz)** *(Default)*:
  - Best for speech analysis, singers, and instrument tuning.
  - Expands the vertical axis to highlight vowels, pitch fundamentals, and formants.
- **🎵 Musical Log (20 Hz – 8 kHz)**:
  - Best for analyzing full musical tracks, piano chords, and acoustic guitars.
- **⚡ Wideband (20 Hz – 12 kHz)**:
  - Captures full treble air, cymbals, breath sibilance, and high-frequency noise.

### 2. Palette Selector (Colormaps)
- **magma** *(Default)*: High perceptual contrast; black noise floor with vibrant rose/amber peaks.
- **viridis**: High-contrast blue-to-green-to-yellow (ideal for colorblind accessibility).
- **plasma**: Vibrant violet-to-orange-to-yellow palette.
- **inferno**: Dark charcoal with high-intensity fiery yellow bursts.

### 3. Sensitivity Slider
- Adjusts visual dynamic range by $\pm 15\text{ dB}$.
- If your microphone is quiet, increase sensitivity ($+10\text{ dB}$ to $+20\text{ dB}$) to bring faint harmonics out of the dark noise floor.
- If signals are blowing out into solid yellow blocks, decrease sensitivity.

### 4. Interactive Crosshair Hover & Time-Frequency Inspection
- **Hover** anywhere on the spectrogram canvas with your cursor.
- The interactive HUD displays:
  - Exact time offset in seconds (e.g., $t = -1.45\text{ s}$).
  - Exact frequency in Hz (e.g., $f = 440.0\text{ Hz}$).
  - The closest musical note and cents deviation (e.g., `A4 (+0.0¢)`).

### 5. Interactive Zoom & Drag-to-Pan
- **Mouse Scroll Wheel**: Zoom in and out centered directly at your mouse cursor. Adaptive axis tick marks automatically recompute down to $10\text{ Hz}$ and $0.05\text{ s}$ increments.
- **Click and Drag**: Pan across time and frequency to examine fine detail.
- **Double Click or [Esc]**: Instantly reset zoom back to the full $4.0\text{ s}$ window.

### 6. Freeze Waterfall (`[F]` or `⏸ Freeze`)
- Pauses the real-time scroll so the waterfall stops moving.
- Audio continues playing through speakers, allowing you to calmly hover over harmonics, measure note intervals, or screenshot patterns for documentation.
- Press `[F]` again to resume live streaming.

---

## 5. Quick Pattern Recognition Cheatsheet

| If You See This Pattern... | What It Actually Is | How to Test in the Studio |
|---|---|---|
| **Single sharp horizontal line** | Steady pure tone (Sine wave) | Set Source to `Generator` $\to$ `Sine` |
| **Grid of steady horizontal lines at 50, 100, 150 Hz** | Mains hum & transformer interference | Click `⚡ Inject 50 Hz Hum` |
| **Parallel horizontal bands that move together** | Musical voice, singing, or string instrument | Speak into `Live Mic` or play `Demo Music` |
| **Cloudy diffuse noise above 3 kHz with no lines** | Consonant sounds ("S", "Sh", "F") | Make an "SSSS" sound into the microphone |
| **Sharp vertical line cutting across all frequencies** | Drum hit, finger snap, or click transient | Tap your desk or snap fingers near mic |
| **Diagonal line sloping upwards or downwards** | Frequency chirp / siren sweep | Set Source to `Generator` $\to$ `Chirp` |
| **Uniform grainy background speckling** | Ambient room noise / thermal circuit hiss | Click `⚡ Inject Noise` $\to$ test `Record 1s Noise` |

---

## 6. How the Spectrogram is Computed (DSP Pipeline)

For students and grading evaluators interested in the mathematics:

1. **Windowed Frame Extraction**: Incoming continuous audio blocks ($N = 2048$ samples at $f_s = 44.1\text{ kHz}$) are tapered with a Hann window $w[n]$ to suppress spectral sidelobe leakage by $>57\text{ dB}$.
2. **Short-Time Fourier Transform (STFT)**: A Discrete Fourier Transform is computed on each frame:
   $$X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j 2\pi k n / N}$$
3. **Logarithmic Decibel Conversion**: Magnitude values are mapped to decibels:
   $$M_{\text{dB}}[k] = 20 \log_{10}\left(\max(|X[k]|, 10^{-4})\right)$$
4. **Frequency Interpolation**: The slice is resampled onto $120$ frequency bins across the active focus band ($20\text{ Hz}$ to $3.6\text{ kHz}$).
5. **FIFO Waterfall Rolling**: The 2D matrix ($120 \times 100$) is shifted left using `np.roll(buffer, -1, axis=1)`, and the latest frequency slice is inserted into the rightmost column ($t = 0\text{ s}$).
6. **Blitted Render**: The updated image is rasterized via Matplotlib `imshow` using hardware-accelerated Tkinter blitting at silky-smooth 30–40 FPS.
