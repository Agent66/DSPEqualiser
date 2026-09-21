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
            return samples

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
            return samples

        elif self.mode == "white_noise":
            samples = self.amplitude * np.random.uniform(-1.0, 1.0, block_size)
            return samples

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
            return self.amplitude * out

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
            return self.amplitude * s

        else:
            return np.zeros(block_size, dtype=np.float64)


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
    Generates a rich multi-instrument demo track (bassline + chords + drum pulse)
    for instant DSP testing without requiring external media files.
    """
    total_samples = int(duration_sec * fs)
    t = np.arange(total_samples) / float(fs)
    audio = np.zeros(total_samples, dtype=np.float64)

    # Chord progression: Am (440, 523, 659) -> F (349, 440, 523) -> C (261, 329, 392) -> G (392, 493, 587)
    chords = [
        ([220.0, 440.0, 523.25, 659.25], 110.0),  # Am + A2 bass
        ([174.61, 349.23, 440.0, 523.25], 87.31),  # F + F2 bass
        ([130.81, 261.63, 329.63, 392.0], 65.41),  # C + C2 bass
        ([196.0, 392.0, 493.88, 587.33], 98.0),   # G + G2 bass
    ]

    beat_sec = 0.5  # 120 BPM
    bar_sec = beat_sec * 4.0

    for i, (chord_freqs, bass_freq) in enumerate(chords):
        t_start = i * 3.0
        t_end = (i + 1) * 3.0
        idx_mask = (t >= t_start) & (t < t_end)
        t_local = t[idx_mask] - t_start

        # Warm synth pad
        pad = np.zeros_like(t_local)
        for f in chord_freqs:
            pad += 0.08 * np.sin(2.0 * np.pi * f * t_local)
            pad += 0.03 * np.sin(2.0 * np.pi * (2.0 * f) * t_local)

        # Deep Bass with envelope
        bass_env = np.exp(-1.5 * (t_local % 1.5))
        bass = 0.25 * bass_env * np.sin(2.0 * np.pi * bass_freq * t_local)

        audio[idx_mask] += pad + bass

    # Rhythm hi-hat pulse (high frequencies)
    hat_period = int(beat_sec * fs / 2.0)
    for h in range(0, total_samples, hat_period):
        h_len = min(int(0.04 * fs), total_samples - h)
        if h_len > 0:
            env = np.linspace(1.0, 0.0, h_len) ** 3
            noise = np.random.uniform(-1.0, 1.0, h_len)
            # High-pass filter for hi-hat sizzle
            b, a = signal.butter(2, 7000.0, btype='high', fs=fs)
            sizzle = signal.lfilter(b, a, noise)
            audio[h:h + h_len] += 0.12 * env * sizzle

    # Normalize to -3 dB peak
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.70

    return audio
