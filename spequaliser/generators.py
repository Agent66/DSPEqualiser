"""
Signal Generators and Audio File Streamer.

Provides:
- Real-time test signal synthesizers (Sine, Chirp Sweep, White Noise, Pink Noise, Multi-tone Chord)
- Interactive WAV file player with looping, seeking, and sample rate conversion
- Built-in demo music generator for instant audio testing without external files
"""

from typing import Optional, Tuple
import numpy as np
import scipy.signal as signal
try:
    import soundfile as sf
except ImportError:
    sf = None

from .dsp import STANDARD_SAMPLE_RATE


class SignalGenerator:
    """
    Real-time continuous block-based test signal generator.
    Maintains phase continuity across successive block generations.
    """
    def __init__(self, sample_rate: int = STANDARD_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.mode = "sine"  # 'sine', 'sweep', 'white_noise', 'pink_noise', 'chord'
        self.frequency = 440.0
        self.amplitude = 0.5
        self.phase = 0.0

        # Sweep parameters
        self.sweep_f_start = 20.0
        self.sweep_f_end = 20000.0
        self.sweep_duration = 5.0  # seconds
        self.sweep_time = 0.0

        # Pink noise filter states (Paul Kellet's filter)
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.b3 = 0.0
        self.b4 = 0.0
        self.b5 = 0.0
        self.b6 = 0.0

        # Hum & Noise injection controls (for DSP benchmark and auto-detection testing)
        self.inject_hum = False
        self.hum_level = 0.22
        self.hum_phase = 0.0

        self.inject_noise = False
        self.noise_level = 0.12

    def set_mode(self, mode: str):
        self.mode = mode.lower()

    def set_frequency(self, freq: float):
        self.frequency = float(np.clip(freq, 10.0, self.sample_rate / 2.0 - 50.0))

    def set_amplitude(self, amp: float):
        self.amplitude = float(np.clip(amp, 0.0, 1.0))

    def generate_block(self, block_size: int) -> np.ndarray:
        """Generates block_size samples of the selected signal."""
        t = np.arange(block_size) / float(self.sample_rate)

        if self.mode == "sine":
            omega = 2.0 * np.pi * self.frequency
            phase_array = self.phase + omega * t
            samples = self.amplitude * np.sin(phase_array)
            self.phase = (self.phase + omega * (block_size / float(self.sample_rate))) % (2.0 * np.pi)

        elif self.mode == "sweep":
            # Logarithmic frequency chirp
            total_samples = int(self.sweep_duration * self.sample_rate)
            t_sweep = (self.sweep_time + np.arange(block_size)) % total_samples
            normalized_t = t_sweep / total_samples

            # Instantaneous frequency: f(t) = f0 * (f1/f0)^t
            k = (self.sweep_f_end / self.sweep_f_start) ** normalized_t
            inst_freq = self.sweep_f_start * k
            # Phase integral: 2 * pi * f0 * ((f1/f0)^t - 1) / ln(f1/f0) * duration
            r = np.log(self.sweep_f_end / self.sweep_f_start)
            inst_phase = 2.0 * np.pi * self.sweep_f_start * self.sweep_duration * (k - 1.0) / r

            samples = self.amplitude * np.sin(inst_phase)
            self.sweep_time = (self.sweep_time + block_size) % total_samples

        elif self.mode == "white_noise":
            samples = self.amplitude * np.random.uniform(-1.0, 1.0, block_size)

        elif self.mode == "pink_noise":
            # Paul Kellet's refined 1/f pink noise filter
            white = np.random.uniform(-1.0, 1.0, block_size)
            out = np.empty(block_size, dtype=np.float64)
            b0, b1, b2, b3, b4, b5, b6 = (
                self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6
            )
            for i in range(block_size):
                w = white[i]
                b0 = 0.99886 * b0 + w * 0.0555179
                b1 = 0.99332 * b1 + w * 0.0750759
                b2 = 0.96900 * b2 + w * 0.1538520
                b3 = 0.86650 * b3 + w * 0.3104856
                b4 = 0.55000 * b4 + w * 0.5329522
                b5 = -0.7616 * b5 - w * 0.0168980
                out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11
                b6 = w * 0.115926
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6 = (
                b0, b1, b2, b3, b4, b5, b6
            )
            samples = self.amplitude * out

        elif self.mode == "chord":
            # Musical A-major chord (440Hz A4, 554.37Hz C#5, 659.25Hz E5, 880Hz A5) + Sub bass (110Hz A2)
            f1, f2, f3, f4, f5 = 110.0, 440.0, 554.37, 659.25, 880.0
            phase_inc = 2.0 * np.pi * t
            s = (
                0.30 * np.sin(self.phase * (f1 / 440.0) + f1 * phase_inc) +
                0.25 * np.sin(self.phase * (f2 / 440.0) + f2 * phase_inc) +
                0.20 * np.sin(self.phase * (f3 / 440.0) + f3 * phase_inc) +
                0.20 * np.sin(self.phase * (f4 / 440.0) + f4 * phase_inc) +
                0.15 * np.sin(self.phase * (f5 / 440.0) + f5 * phase_inc)
            )
            self.phase = (self.phase + 2.0 * np.pi * 440.0 * (block_size / float(self.sample_rate))) % (2.0 * np.pi)
            samples = self.amplitude * s

        else:
            samples = np.zeros(block_size, dtype=np.float64)

        # Inject 50 Hz Mains Hum & Harmonics (100 Hz, 150 Hz) for Auto Hum Removal testing
        if self.inject_hum:
            w50 = 2.0 * np.pi * 50.0
            w100 = 2.0 * np.pi * 100.0
            w150 = 2.0 * np.pi * 150.0
            hum = (
                self.hum_level * np.sin(self.hum_phase + w50 * t) +
                (self.hum_level * 0.45) * np.sin(2.0 * self.hum_phase + w100 * t) +
                (self.hum_level * 0.25) * np.sin(3.0 * self.hum_phase + w150 * t)
            )
            self.hum_phase = (self.hum_phase + w50 * (block_size / float(self.sample_rate))) % (2.0 * np.pi)
            samples = samples + hum

        # Inject Background Noise (for Spectral Subtraction testing)
        if self.inject_noise:
            noise = self.noise_level * np.random.normal(0.0, 1.0, block_size)
            samples = samples + noise

        return samples


class AudioFileReader:
    """
    Reads an audio file (WAV, FLAC, OGG) and streams contiguous blocks.
    Supports looping, seeking, and mono/stereo channel downmixing.
    """
    def __init__(self, target_sample_rate: int = STANDARD_SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate
        self.audio_data: Optional[np.ndarray] = None
        self.file_sample_rate: int = target_sample_rate
        self.current_pos: int = 0
        self.is_looping: bool = True
        self.file_path: Optional[str] = None
        self.duration_seconds: float = 0.0

    def load_file(self, path: str) -> bool:
        """Loads an audio file and resamples to target sample rate if needed."""
        if sf is None:
            return False

        try:
            data, fs = sf.read(path, dtype='float32')
            # Downmix to mono if multi-channel
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Resample if needed
            if fs != self.target_sample_rate:
                num_samples = int(len(data) * float(self.target_sample_rate) / float(fs))
                data = signal.resample(data, num_samples)

            self.audio_data = data.astype(np.float64)
            self.file_sample_rate = self.target_sample_rate
            self.duration_seconds = len(self.audio_data) / float(self.target_sample_rate)
            self.current_pos = 0
            self.file_path = path
            return True
        except Exception:
            return False

    def seek(self, fraction: float):
        """Seeks to position (0.0 to 1.0)."""
        if self.audio_data is not None and len(self.audio_data) > 0:
            self.current_pos = int(np.clip(fraction, 0.0, 1.0) * len(self.audio_data))

    def get_progress(self) -> float:
        """Returns playback position fraction (0.0 to 1.0)."""
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        return self.current_pos / float(len(self.audio_data))

    def read_block(self, block_size: int) -> np.ndarray:
        """Reads block_size samples from the loaded audio file."""
        if self.audio_data is None or len(self.audio_data) == 0:
            return np.zeros(block_size, dtype=np.float64)

        total_len = len(self.audio_data)
        end_pos = self.current_pos + block_size

        if end_pos <= total_len:
            block = self.audio_data[self.current_pos:end_pos]
            self.current_pos = end_pos
            if self.current_pos >= total_len and self.is_looping:
                self.current_pos = 0
            return block
        else:
            # Reached end of file
            part1 = self.audio_data[self.current_pos:total_len]
            needed = block_size - len(part1)
            if self.is_looping:
                self.current_pos = needed % total_len
                part2 = self.audio_data[:needed]
                return np.concatenate([part1, part2])
            else:
                self.current_pos = total_len
                part2 = np.zeros(needed, dtype=np.float64)
                return np.concatenate([part1, part2])


def generate_synthetic_demo_track(duration_sec: float = 12.0, fs: int = STANDARD_SAMPLE_RATE) -> np.ndarray:
    """
    Generates a rich, warm, multi-instrument demo track (warm Rhodes chords + acoustic bass + soft brush)
    with smooth raised-cosine envelopes, zero phase jumps, and seamless circular looping.
    Eliminates harsh 7 kHz white noise bursts and artificial delayed re-triggers.
    """
    total_samples = int(duration_sec * fs)
    t = np.arange(total_samples) / float(fs)
    audio = np.zeros(total_samples, dtype=np.float64)

    # Chords: Am7 (A2, C3, E3, G3, C4) -> Fmaj7 (F2, C3, E3, A3, C4) -> Cmaj7 (C2, G2, E3, G3, B3) -> Gadd9 (G2, D3, G3, B3, D4)
    chord_data = [
        ([110.0, 130.81, 164.81, 196.00, 261.63], 55.0),   # Am7 + A1 sub
        ([87.31, 130.81, 164.81, 220.00, 261.63], 43.65),  # Fmaj7 + F1 sub
        ([65.41, 98.00, 164.81, 196.00, 246.94], 32.70),   # Cmaj7 + C1 sub
        ([98.00, 146.83, 196.00, 246.94, 293.66], 49.00),  # Gadd9 + G1 sub
    ]

    chord_sec = 3.0

    for i, (notes, sub_bass) in enumerate(chord_data):
        t_start = i * chord_sec
        t_end = (i + 1) * chord_sec
        mask = (t >= t_start) & (t < t_end)
        tl = t[mask] - t_start
        n_seg = len(tl)

        # Smooth ADSR envelope: 25ms raised-cosine attack, gentle decay, 40ms smooth release
        env = np.ones(n_seg, dtype=np.float64)
        att_len = int(0.025 * fs)
        rel_len = int(0.040 * fs)
        if n_seg > (att_len + rel_len):
            env[:att_len] = 0.5 * (1.0 - np.cos(np.pi * np.arange(att_len) / att_len))
            env[-rel_len:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(rel_len) / rel_len))

        # Warm electric piano / Rhodes harmonics with natural decay
        chord_wave = np.zeros(n_seg, dtype=np.float64)
        for freq in notes:
            note_decay = np.exp(-0.35 * tl)
            tone = (
                0.12 * np.sin(2.0 * np.pi * freq * tl) +
                0.045 * np.sin(2.0 * np.pi * 2.0 * freq * tl) * np.exp(-0.75 * tl) +
                0.015 * np.sin(2.0 * np.pi * 3.0 * freq * tl) * np.exp(-1.4 * tl)
            )
            chord_wave += tone * note_decay

        # Smooth warm acoustic bassline that follows the chord progression naturally
        bass_decay = np.exp(-0.45 * tl)
        bass = 0.18 * np.sin(2.0 * np.pi * sub_bass * tl) * bass_decay

        audio[mask] += (chord_wave + bass) * env

    # Soft, organic acoustic brush rhythm (every beat = 0.5s)
    beat_samples = int(0.5 * fs)
    for b in range(0, total_samples, beat_samples):
        b_len = min(int(0.035 * fs), total_samples - b)
        if b_len > 10:
            # Soft windowed noise in warm 2.5 kHz - 4.5 kHz band (no harsh 7 kHz bursts)
            w_brush = np.sin(np.pi * np.arange(b_len) / b_len) ** 2
            rnd = np.random.uniform(-1.0, 1.0, b_len)
            sos_brush = signal.butter(2, [2500.0, 4500.0], btype='bandpass', fs=fs, output='sos')
            filtered_brush = signal.sosfilt(sos_brush, rnd)
            audio[b:b + b_len] += 0.030 * w_brush * filtered_brush

    # Seamless circular loop crossfade at ends (50ms)
    fade_len = int(0.05 * fs)
    fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_in[::-1]

    # Normalize to -3 dB peak
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.70

    return audio
