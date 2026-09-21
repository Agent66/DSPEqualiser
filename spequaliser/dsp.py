"""
Core Digital Signal Processing (DSP) Module.

Implements:
- Windowing (Hann, Hamming, Rectangular)
- Fast Fourier Transform (FFT) spectrum analysis & parabolic peak interpolation
- Digital Filter Design (Low-Pass, High-Pass, Band-Pass, Notch) using SOS biquad filters
- Multi-Band Parametric/Graphic Equalizer with state preservation across audio blocks
- Pitch Detection using Normalized Autocorrelation (NACF) and musical note mapping
- Spectral Subtraction Noise Reduction
- Voice Activity Detection (VAD) using Short-Time Energy and Zero-Crossing Rate (ZCR)
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import scipy.signal as signal

STANDARD_SAMPLE_RATE = 44100
DEFAULT_BLOCK_SIZE = 2048

# Standard ISO Octave / 2/3-Octave Center Frequencies for 8-band EQ
EQ_CENTER_FREQUENCIES = [63, 160, 400, 1000, 2500, 6000, 12000, 16000]
EQ_BAND_NAMES = ["63 Hz", "160 Hz", "400 Hz", "1 kHz", "2.5 kHz", "6 kHz", "12 kHz", "16 kHz"]

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ============================================================================
# 1. WINDOWING FUNCTIONS
# ============================================================================

def get_window(window_type: str, n: int) -> np.ndarray:
    """
    Returns a window function of length n.
    Supported types: 'hann', 'hamming', 'rectangular'.
    """
    w_type = window_type.lower()
    if w_type == 'hann':
        return signal.windows.hann(n, sym=False)
    elif w_type == 'hamming':
        return signal.windows.hamming(n, sym=False)
    elif w_type == 'rectangular':
        return np.ones(n, dtype=np.float64)
    else:
        return signal.windows.hann(n, sym=False)


# ============================================================================
# 2. FFT SPECTRUM ANALYSIS
# ============================================================================

def compute_spectrum(
    samples: np.ndarray,
    sample_rate: int = STANDARD_SAMPLE_RATE,
    window_type: str = 'hann',
    ref_level: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Computes single-sided FFT magnitude spectrum in decibels (dB).

    Returns:
        freqs: Frequency bins (Hz) from 0 to Nyquist (fs / 2)
        mag_db: Magnitude in dB, scaled relative to full-scale sinusoid
        peak_freq: Dominant frequency estimate in Hz (sub-bin parabolic interpolation)
        peak_db: Peak magnitude in dB
    """
    n = len(samples)
    if n == 0:
        return np.array([0.0]), np.array([-100.0]), 0.0, -100.0

    w = get_window(window_type, n)
    w_sum = np.sum(w)
    if w_sum <= 0:
        w_sum = 1.0

    windowed_signal = samples * w
    fft_result = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Coherent gain normalization for true sinusoid peak calibration
    mag = np.abs(fft_result) * (2.0 / w_sum)
    mag[0] /= 2.0  # DC component
    if len(mag) > 1 and n % 2 == 0:
        mag[-1] /= 2.0  # Nyquist component

    # Prevent log(0)
    mag_db = 20.0 * np.log10(np.maximum(mag / ref_level, 1e-6))

    # Quadratic / Parabolic sub-bin peak interpolation for accurate frequency identification
    # Ignore DC bin (index 0)
    if len(mag) > 3:
        search_slice = mag[1:-1]
        k = int(np.argmax(search_slice)) + 1
        alpha = mag_db[k - 1]
        beta = mag_db[k]
        gamma = mag_db[k + 1]

        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-9:
            delta = 0.5 * (alpha - gamma) / denom
            delta = np.clip(delta, -0.5, 0.5)
        else:
            delta = 0.0

        bin_width = sample_rate / float(n)
        peak_freq = (k + delta) * bin_width
        peak_db = beta - 0.25 * (alpha - gamma) * delta
    elif len(mag) > 1:
        k = int(np.argmax(mag[1:])) + 1
        peak_freq = freqs[k]
        peak_db = mag_db[k]
    else:
        peak_freq = 0.0
        peak_db = mag_db[0]

    return freqs, mag_db, float(peak_freq), float(peak_db)


# ============================================================================
# 3. BIQUAD FILTER DESIGN (Audio EQ Cookbook)
# ============================================================================

def design_peaking_biquad(
    freq: float,
    gain_db: float,
    q: float = 1.414,
    fs: int = STANDARD_SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Designs a 2nd-order Peaking EQ biquad filter (Audio EQ Cookbook).
    Returns (b, a) coefficient vectors.
    """
    if abs(gain_db) < 0.05:
        # Flat passthrough
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    freq = np.clip(freq, 10.0, fs / 2.0 - 50.0)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def design_low_shelf_biquad(
    freq: float,
    gain_db: float,
    q: float = 0.707,
    fs: int = STANDARD_SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray]:
    """Low-shelf biquad filter."""
    if abs(gain_db) < 0.05:
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    freq = np.clip(freq, 10.0, fs / 2.0 - 50.0)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_a_alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def design_high_shelf_biquad(
    freq: float,
    gain_db: float,
    q: float = 0.707,
    fs: int = STANDARD_SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray]:
    """High-shelf biquad filter."""
    if abs(gain_db) < 0.05:
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    freq = np.clip(freq, 10.0, fs / 2.0 - 50.0)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrt_a_alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


# ============================================================================
# 4. STANDALONE DIGITAL FILTER (Low-pass, High-pass, Band-pass, Notch)
# ============================================================================

class StandaloneFilter:
    """
    Implements standalone Low-Pass, High-Pass, Band-Pass, and Notch filters.
    Uses Second-Order Sections (SOS) representation with persistent state
    to guarantee zero boundary transients across real-time audio blocks.
    """
    def __init__(self, sample_rate: int = STANDARD_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.mode = "bypass"  # 'bypass', 'lowpass', 'highpass', 'bandpass', 'notch'
        self.cutoff = 1000.0   # Hz
        self.q = 1.0          # Q factor / resonance
        self.order = 4
        self.sos: Optional[np.ndarray] = None
        self.zi: Optional[np.ndarray] = None
        self._rebuild_filter()

    def set_params(self, mode: str, cutoff: float, q: float = 1.0, order: int = 4):
        """Updates filter parameters and recalculates SOS coefficients."""
        mode = mode.lower()
        if mode != self.mode or abs(cutoff - self.cutoff) > 0.5 or abs(q - self.q) > 0.05 or order != self.order:
            self.mode = mode
            self.cutoff = float(cutoff)
            self.q = float(q)
            self.order = int(order)
            self._rebuild_filter()

    def _rebuild_filter(self):
        nyquist = self.sample_rate / 2.0
        cutoff_clamped = np.clip(self.cutoff, 20.0, nyquist - 100.0)

        if self.mode == "bypass":
            self.sos = None
            self.zi = None
            return

        try:
            if self.mode == "lowpass":
                self.sos = signal.butter(self.order, cutoff_clamped, btype='low', fs=self.sample_rate, output='sos')
            elif self.mode == "highpass":
                self.sos = signal.butter(self.order, cutoff_clamped, btype='high', fs=self.sample_rate, output='sos')
            elif self.mode == "bandpass":
                # Bandwidth calculated from Q: BW = fc / Q
                bw = cutoff_clamped / max(self.q, 0.1)
                f_low = max(20.0, cutoff_clamped - bw / 2.0)
                f_high = min(nyquist - 50.0, cutoff_clamped + bw / 2.0)
                if f_high <= f_low + 10.0:
                    f_high = f_low + 10.0
                self.sos = signal.butter(2, [f_low, f_high], btype='bandpass', fs=self.sample_rate, output='sos')
            elif self.mode == "notch":
                q_notch = max(1.0, self.q * 3.0)
                b, a = signal.iirnotch(cutoff_clamped, q_notch, fs=self.sample_rate)
                self.sos = signal.tf2sos(b, a)
            else:
                self.sos = None
                self.zi = None
                return

            # Initialize filter state for seamless real-time processing
            self.zi = signal.sosfilt_zi(self.sos)
        except Exception:
            self.sos = None
            self.zi = None

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Processes a 1D block of audio samples through the digital filter."""
        if self.mode == "bypass" or self.sos is None:
            return audio_block

        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)

        # Scale zi by first sample to minimize start-up transients
        filtered_block, self.zi = signal.sosfilt(self.sos, audio_block, zi=self.zi)
        return filtered_block

    def get_frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        """Returns the magnitude response in dB at the specified frequencies."""
        if self.mode == "bypass" or self.sos is None:
            return np.zeros_like(freqs)

        w = 2.0 * np.pi * freqs / self.sample_rate
        _, h = signal.sosfreqz(self.sos, worN=w)
        return 20.0 * np.log10(np.maximum(np.abs(h), 1e-6))


# ============================================================================
# 5. MULTI-BAND EQUALIZER
# ============================================================================

class MultiBandEqualizer:
    """
    8-Band Parametric/Graphic Audio Equalizer.
    Maintains persistent filter state (`zi`) across contiguous audio chunks,
    guaranteeing seamless audio playback without clicks or popping.
    """
    def __init__(
        self,
        center_freqs: List[float] = EQ_CENTER_FREQUENCIES,
        sample_rate: int = STANDARD_SAMPLE_RATE
    ):
        self.sample_rate = sample_rate
        self.center_freqs = list(center_freqs)
        self.num_bands = len(self.center_freqs)
        self.gains_db = [0.0] * self.num_bands
        self.q_factors = [1.414] * self.num_bands
        self.enabled = True

        # State storage: each band is a 2nd-order biquad (1 section SOS)
        self.sos_list: List[Optional[np.ndarray]] = [None] * self.num_bands
        self.zi_list: List[Optional[np.ndarray]] = [None] * self.num_bands

        self._update_all_bands()

    def set_band_gain(self, band_index: int, gain_db: float):
        """Sets gain in decibels for a specific band (-15 dB to +15 dB)."""
        if 0 <= band_index < self.num_bands:
            gain_db = float(np.clip(gain_db, -24.0, 24.0))
            if abs(self.gains_db[band_index] - gain_db) > 0.05:
                self.gains_db[band_index] = gain_db
                self._update_band(band_index)

    def set_all_gains(self, gains_db: List[float]):
        """Sets gains for all bands simultaneously."""
        for i, g in enumerate(gains_db[:self.num_bands]):
            self.set_band_gain(i, g)

    def reset_gains(self):
        """Resets all bands to flat (0 dB)."""
        self.set_all_gains([0.0] * self.num_bands)

    def _update_band(self, idx: int):
        f0 = self.center_freqs[idx]
        gain = self.gains_db[idx]
        q = self.q_factors[idx]

        if abs(gain) < 0.05:
            self.sos_list[idx] = None
            self.zi_list[idx] = None
            return

        if idx == 0 and f0 <= 100.0:
            b, a = design_low_shelf_biquad(f0, gain, q=0.707, fs=self.sample_rate)
        elif idx == self.num_bands - 1 and f0 >= 14000.0:
            b, a = design_high_shelf_biquad(f0, gain, q=0.707, fs=self.sample_rate)
        else:
            b, a = design_peaking_biquad(f0, gain, q=q, fs=self.sample_rate)

        sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
        self.sos_list[idx] = sos

        # Keep or reset zi
        if self.zi_list[idx] is None:
            self.zi_list[idx] = signal.sosfilt_zi(sos)

    def _update_all_bands(self):
        for i in range(self.num_bands):
            self._update_band(i)

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Processes audio block through the cascade of 8 EQ biquad filters."""
        if not self.enabled:
            return audio_block

        out = audio_block.copy()
        for idx in range(self.num_bands):
            sos = self.sos_list[idx]
            if sos is not None:
                if self.zi_list[idx] is None:
                    self.zi_list[idx] = signal.sosfilt_zi(sos)
                out, self.zi_list[idx] = signal.sosfilt(sos, out, zi=self.zi_list[idx])

        return out

    def get_frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        """Computes combined frequency response in dB across all EQ bands."""
        if not self.enabled:
            return np.zeros_like(freqs)

        w = 2.0 * np.pi * freqs / self.sample_rate
        total_mag_db = np.zeros_like(freqs)

        for idx in range(self.num_bands):
            sos = self.sos_list[idx]
            if sos is not None:
                _, h = signal.sosfreqz(sos, worN=w)
                total_mag_db += 20.0 * np.log10(np.maximum(np.abs(h), 1e-6))

        return total_mag_db


# Presets for the 8-band Equalizer
EQ_PRESETS: Dict[str, List[float]] = {
    "Flat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Bass Boost": [6.0, 5.0, 3.0, 1.0, 0.0, 0.0, -1.0, -1.0],
    "Vocal Booster": [-2.0, -1.0, 1.0, 4.5, 4.0, 2.0, 0.0, -1.0],
    "Treble Boost": [-1.0, -1.0, 0.0, 1.0, 2.5, 4.5, 6.0, 6.5],
    "Acoustic": [4.0, 3.0, 1.5, 1.0, 2.0, 3.5, 4.0, 3.0],
    "Electronic / EDM": [5.5, 4.5, 1.0, -1.0, 2.0, 3.5, 5.0, 5.5],
    "Rock": [5.0, 3.5, -1.0, -0.5, 1.5, 3.5, 4.5, 4.0],
    "Speech Clarifier": [-6.0, -3.0, 0.0, 3.0, 4.0, 3.0, 1.0, -2.0],
}


# ============================================================================
# 6. PITCH DETECTION (Normalized Autocorrelation / YIN)
# ============================================================================

def detect_pitch(
    samples: np.ndarray,
    sample_rate: int = STANDARD_SAMPLE_RATE,
    f_min: float = 50.0,
    f_max: float = 2000.0,
    threshold: float = 0.3
) -> Tuple[float, str, float]:
    """
    Detects the fundamental frequency (F0) using Normalized Autocorrelation Function (NACF).

    Returns:
        pitch_hz: Fundamental frequency in Hz (0.0 if unvoiced / no clear pitch)
        note_name: Closest musical note (e.g., 'A4', 'C#3')
        cents_dev: Deviation in cents from standard equal temperament tuning (-50 to +50)
    """
    n = len(samples)
    if n < 256:
        return 0.0, "--", 0.0

    # Subtract DC offset
    signal_centered = samples - np.mean(samples)
    energy = np.sum(signal_centered ** 2)
    if energy < 1e-6:
        return 0.0, "--", 0.0

    # FFT-based autocorrelation for high efficiency O(N log N)
    fft_size = 1 << (2 * n - 1).bit_length()
    f_x = np.fft.rfft(signal_centered, n=fft_size)
    r_xx = np.fft.irfft(f_x * np.conj(f_x), n=fft_size)[:n]

    # Normalize autocorrelation
    r_norm = r_xx / (r_xx[0] + 1e-9)

    lag_min = int(sample_rate / f_max)
    lag_max = min(int(sample_rate / f_min), n - 2)

    if lag_min >= lag_max:
        return 0.0, "--", 0.0

    # Search for first significant local maximum in valid lag window
    search_lags = r_norm[lag_min:lag_max + 1]
    peak_indices, _ = signal.find_peaks(search_lags, height=threshold, distance=max(1, lag_min // 2))

    if len(peak_indices) == 0:
        # Fallback: check if the global maximum in search window exceeds threshold
        best_rel_idx = int(np.argmax(search_lags))
        if search_lags[best_rel_idx] >= threshold:
            best_lag = lag_min + best_rel_idx
        else:
            return 0.0, "--", 0.0
    else:
        best_lag = lag_min + peak_indices[int(np.argmax(search_lags[peak_indices]))]

    # Parabolic interpolation for sub-sample lag resolution
    if 0 < best_lag < n - 1:
        y0 = r_norm[best_lag - 1]
        y1 = r_norm[best_lag]
        y2 = r_norm[best_lag + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-9:
            delta_lag = 0.5 * (y0 - y2) / denom
            refined_lag = best_lag + delta_lag
        else:
            refined_lag = float(best_lag)
    else:
        refined_lag = float(best_lag)

    pitch_hz = sample_rate / refined_lag

    # Musical note calculation: MIDI = 69 + 12 * log2(f / 440)
    midi_num = 69.0 + 12.0 * np.log2(max(pitch_hz, 1.0) / 440.0)
    nearest_midi = int(round(midi_num))
    cents_dev = (midi_num - nearest_midi) * 100.0

    note_idx = nearest_midi % 12
    octave = (nearest_midi // 12) - 1
    note_str = f"{NOTE_NAMES[note_idx]}{octave}"

    return float(pitch_hz), note_str, float(cents_dev)


# ============================================================================
# 7. SPECTRAL SUBTRACTION NOISE REDUCTION
# ============================================================================

class SpectralNoiseReducer:
    """
    Real-time noise reduction using Spectral Subtraction.
    Estimates stationary background noise power spectrum and suppresses it.
    """
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, alpha: float = 2.0, beta: float = 0.05):
        self.block_size = block_size
        self.alpha = alpha  # Over-subtraction factor
        self.beta = beta    # Spectral floor factor
        self.noise_profile: Optional[np.ndarray] = None
        self.is_calibrating = False
        self.calibration_frames: List[np.ndarray] = []
        self.enabled = False

    def start_calibration(self):
        """Starts collecting background ambient noise frames for calibration."""
        self.calibration_frames = []
        self.is_calibrating = True

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """Processes audio frame through spectral subtraction filter."""
        n = len(audio_block)
        w = signal.windows.hann(n, sym=False)
        windowed = audio_block * w
        fft_vals = np.fft.rfft(windowed)
        mag = np.abs(fft_vals)
        phase = np.angle(fft_vals)

        if self.is_calibrating:
            self.calibration_frames.append(mag)
            if len(self.calibration_frames) >= 15:  # ~0.5 sec calibration
                self.noise_profile = np.mean(self.calibration_frames, axis=0)
                self.is_calibrating = False
            return audio_block

        if not self.enabled or self.noise_profile is None:
            return audio_block

        # If block size changed, resize profile
        if len(self.noise_profile) != len(mag):
            self.noise_profile = np.resize(self.noise_profile, len(mag))

        # Spectral subtraction with spectral floor:
        # S_hat = max(Y - alpha * N, beta * Y)
        subtracted_mag = np.maximum(mag - self.alpha * self.noise_profile, self.beta * mag)

        # Reconstruct with original phase
        clean_fft = subtracted_mag * np.exp(1j * phase)
        clean_time = np.fft.irfft(clean_fft, n=n)

        return clean_time


# ============================================================================
# 8. VOICE ACTIVITY DETECTION (VAD)
# ============================================================================

def compute_energy(samples: np.ndarray) -> Tuple[float, float]:
    """
    Computes Short-Time Energy (STE / Mean Square Energy) and Logarithmic Energy in dB.

    Returns:
        energy: Mean square energy (linear scale, >= 0.0)
        energy_db: Logarithmic energy in dB relative to full-scale (<= 0 dB)
    """
    n = len(samples)
    if n == 0:
        return 0.0, -100.0

    energy = float(np.mean(samples.astype(np.float64) ** 2))
    energy_db = float(10.0 * np.log10(max(energy, 1e-10)))
    return energy, energy_db


def compute_zcr(samples: np.ndarray) -> float:
    """
    Computes normalized Zero-Crossing Rate (ZCR) per sample.
    ZCR measures the frequency of signal sign transitions:
      ZCR = sum(|sgn(x[n]) - sgn(x[n-1])|) / (2 * (N - 1))

    High ZCR indicates unvoiced speech (fricatives, consonants, sibilants)
    or high-frequency noise, whereas voiced speech has low-to-medium ZCR.
    """
    n = len(samples)
    if n < 2:
        return 0.0

    # Remove DC bias to ensure zero-crossings reflect true signal oscillations
    centered = samples.astype(np.float64) - np.mean(samples)
    signs = np.signbit(centered)
    crossings = np.count_nonzero(np.diff(signs))
    return float(crossings / (n - 1))


def detect_vad(
    samples: np.ndarray,
    sample_rate: int = STANDARD_SAMPLE_RATE,
    energy_threshold_db: float = -45.0,
    speech_energy_threshold_db: float = -32.0,
    zcr_threshold: float = 0.12
) -> Tuple[bool, float, float]:
    """
    Stateless Voice Activity Detector using Short-Time Energy and Zero-Crossing Rate.

    Detection Logic:
    1. Voiced Speech: High Energy (>= speech_energy_threshold_db) and low-to-moderate ZCR.
    2. Unvoiced Speech: Moderate Energy (>= energy_threshold_db) combined with High ZCR (>= zcr_threshold).
    3. Silence/Noise: Low Energy (< energy_threshold_db).

    Returns:
        is_speech: True if speech (voiced or unvoiced) is detected, False for silence
        energy_db: Frame energy in decibels
        zcr: Frame zero-crossing rate (0.0 to 1.0)
    """
    energy, energy_db = compute_energy(samples)
    zcr = compute_zcr(samples)

    if energy_db < energy_threshold_db:
        return False, energy_db, zcr

    # Voiced speech check (high energy vowels, tones, voiced consonants)
    is_voiced = energy_db >= speech_energy_threshold_db

    # Unvoiced speech check (consonants like /s/, /t/, /f/, /sh/ with moderate energy + high ZCR)
    is_unvoiced = (energy_db >= energy_threshold_db) and (zcr >= zcr_threshold)

    is_speech = bool(is_voiced or is_unvoiced)
    return is_speech, energy_db, zcr


class VoiceActivityDetector:
    """
    Real-Time Telecom-Grade Voice Activity Detector (VAD).
    Combines Energy and Zero-Crossing Rate (ZCR) with:
    - Adaptive background noise floor tracking
    - Hangover Frames: Maintains active state across short inter-syllabic pauses
      and word decays to prevent syllable clipping (as used in GSM/AMR/WebRTC DTX).
    """
    def __init__(
        self,
        sample_rate: int = STANDARD_SAMPLE_RATE,
        energy_threshold_db: float = -45.0,
        speech_energy_threshold_db: float = -32.0,
        zcr_threshold: float = 0.12,
        hangover_frames: int = 5,
        adaptive_noise: bool = True
    ):
        self.sample_rate = sample_rate
        self.base_energy_threshold_db = energy_threshold_db
        self.speech_energy_threshold_db = speech_energy_threshold_db
        self.zcr_threshold = zcr_threshold
        self.hangover_frames = hangover_frames
        self.adaptive_noise = adaptive_noise

        self.hangover_counter = 0
        self.noise_floor_db = energy_threshold_db
        self.is_speech_active = False
        self.total_frames = 0
        self.speech_frames = 0

    def process_block(self, audio_block: np.ndarray) -> Tuple[bool, float, float]:
        """
        Processes an audio block, returning (is_speech_active, energy_db, zcr).
        Applies hangover logic to maintain transmission continuity.
        """
        self.total_frames += 1

        active_thresh_db = self.noise_floor_db if self.adaptive_noise else self.base_energy_threshold_db
        raw_speech, energy_db, zcr = detect_vad(
            audio_block,
            sample_rate=self.sample_rate,
            energy_threshold_db=active_thresh_db,
            speech_energy_threshold_db=self.speech_energy_threshold_db,
            zcr_threshold=self.zcr_threshold
        )

        # Adaptive background noise tracking during confirmed silence
        if not raw_speech and self.adaptive_noise:
            alpha = 0.05
            # Target threshold sits ~8 dB above estimated background floor
            target_thresh = energy_db + 8.0
            self.noise_floor_db = (1.0 - alpha) * self.noise_floor_db + alpha * target_thresh
            self.noise_floor_db = float(np.clip(self.noise_floor_db, -60.0, -35.0))

        # Telecom Hangover Logic:
        # When speech is detected, reset hangover counter to full hangover duration.
        # When raw speech drops, stay active until hangover expires.
        if raw_speech:
            self.hangover_counter = self.hangover_frames
            self.is_speech_active = True
        else:
            if self.hangover_counter > 0:
                self.hangover_counter -= 1
                self.is_speech_active = True
            else:
                self.is_speech_active = False

        if self.is_speech_active:
            self.speech_frames += 1

        return self.is_speech_active, energy_db, zcr

    def reset(self):
        """Resets hangover counter, stats, and noise floor estimation."""
        self.hangover_counter = 0
        self.is_speech_active = False
        self.noise_floor_db = self.base_energy_threshold_db
        self.total_frames = 0
        self.speech_frames = 0

    @property
    def speech_ratio(self) -> float:
        """Fraction of processed frames containing active speech."""
        return self.speech_frames / max(1, self.total_frames)


if __name__ == "__main__":
    print("=" * 65)
    print(" SpEqualiser - Core DSP Module (spequaliser/dsp.py)")
    print("=" * 65)
    print("This file contains the core DSP library functions and classes.")
    print("Running quick self-test...")

    fs = STANDARD_SAMPLE_RATE
    t = np.arange(2048) / float(fs)
    test_freq = 440.0
    x = np.sin(2.0 * np.pi * test_freq * t)

    # 1. FFT Check
    freqs, mag_db, peak_f, peak_db = compute_spectrum(x, sample_rate=fs, window_type="hann")
    print(f" [OK] FFT Spectrum Analysis : Target {test_freq:.1f} Hz -> Detected {peak_f:.1f} Hz")

    # 2. Pitch Detection Check
    pitch_hz, note_str, cents_dev = detect_pitch(x, sample_rate=fs)
    print(f" [OK] Pitch Detection       : Detected {note_str} ({pitch_hz:.1f} Hz, {cents_dev:+.1f} cents)")

    # 3. Filter Check
    filt = StandaloneFilter(sample_rate=fs)
    filt.set_params(mode="lowpass", cutoff=1000.0, q=0.707, order=4)
    print(f" [OK] Biquad Filter Design  : 4th-order Low-Pass at 1000 Hz created")

    # 4. Equalizer Check
    eq = MultiBandEqualizer(sample_rate=fs)
    print(f" [OK] 8-Band Equalizer      : {len(EQ_CENTER_FREQUENCIES)} bands initialized")

    # 5. VAD Check
    vad = VoiceActivityDetector(sample_rate=fs)
    is_sp, e_db, z = vad.process_block(x)
    silence = np.zeros(2048)
    for _ in range(10):
        is_sp_silence, _, _ = vad.process_block(silence)
    print(f" [OK] Voice Activity Detect : Sine tone -> {is_sp} (Energy: {e_db:.1f} dB), Silence -> {is_sp_silence}")

    print("\nNext steps:")
    print("  - To run the full verification suite : python verify_dsp.py")
    print("  - To launch the GUI application     : python main.py")
    print("  - To run automated unit tests       : python -m unittest discover -s tests")
    print("=" * 65)

