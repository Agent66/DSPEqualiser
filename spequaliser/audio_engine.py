"""
Real-Time Audio Engine.

Manages:
- sounddevice real-time input/output audio streaming callback
- Input source selection (Microphone, Test Signal Generator, Audio File, Demo Music)
- Real-time DSP pipeline processing (Noise Reduction -> Standalone Filter -> 8-Band Equalizer -> Master Gain)
- Thread-safe data transfer to GUI visualizer
- Resilient fallback to output-only stream if input microphone is unavailable
"""

import threading
import queue
import time
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .dsp import (
    STANDARD_SAMPLE_RATE,
    DEFAULT_BLOCK_SIZE,
    MultiBandEqualizer,
    StandaloneFilter,
    SpectralNoiseReducer,
    compute_spectrum,
    detect_pitch
)
from .generators import SignalGenerator, AudioFileReader, generate_synthetic_demo_track


class AudioEngine:
    """
    Core real-time audio streaming and processing engine.
    """
    def __init__(
        self,
        sample_rate: int = STANDARD_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size

        # DSP Components
        self.equalizer = MultiBandEqualizer(sample_rate=sample_rate)
        self.filter = StandaloneFilter(sample_rate=sample_rate)
        self.noise_reducer = SpectralNoiseReducer(block_size=block_size)

        # Audio Generators & Readers
        self.signal_gen = SignalGenerator(sample_rate=sample_rate)
        self.file_reader = AudioFileReader(target_sample_rate=sample_rate)
        self.demo_reader = AudioFileReader(target_sample_rate=sample_rate)

        # Pre-load demo track
        self._init_demo_track()

        # Audio Routing & Controls
        self.source_mode = "demo"  # 'mic', 'generator', 'file', 'demo'
        self.is_running = False
        self.monitor_output = True  # Output sound to speakers
        self.master_gain = 0.85
        self.mic_input_gain = 1.0

        # Stream & Threading
        self.stream: Optional[Any] = None
        self.stream_has_input = False
        self._lock = threading.Lock()

        # Visualizer data queue (holds latest block for oscilloscope & FFT)
        self.viz_queue: queue.Queue = queue.Queue(maxsize=4)

        # Latency & performance stats
        self.buffer_underruns = 0
        self.last_callback_time = 0.0
        self.peak_in_level = 0.0
        self.peak_out_level = 0.0

    def _init_demo_track(self):
        """Generates synthetic musical track for instant testing."""
        demo_audio = generate_synthetic_demo_track(duration_sec=12.0, fs=self.sample_rate)
        self.demo_reader.audio_data = demo_audio
        self.demo_reader.file_sample_rate = self.sample_rate
        self.demo_reader.duration_seconds = len(demo_audio) / float(self.sample_rate)
        self.demo_reader.is_looping = True

    def get_audio_devices(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Returns list of available input and output devices."""
        if sd is None:
            return [], []
        try:
            devices = sd.query_devices()
            inputs = []
            outputs = []
            for idx, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    inputs.append({'index': idx, 'name': d['name'], 'channels': d['max_input_channels']})
                if d['max_output_channels'] > 0:
                    outputs.append({'index': idx, 'name': d['name'], 'channels': d['max_output_channels']})
            return inputs, outputs
        except Exception:
            return [], []

    def set_source(self, source: str):
        """Switches audio source ('mic', 'generator', 'file', 'demo')."""
        with self._lock:
            self.source_mode = source.lower()

    def set_master_gain(self, gain: float):
        """Sets master output volume (0.0 to 1.5)."""
        self.master_gain = float(np.clip(gain, 0.0, 1.5))

    def set_mic_gain(self, gain: float):
        """Sets microphone input boost/cut (0.0 to 3.0)."""
        self.mic_input_gain = float(np.clip(gain, 0.0, 3.0))

    def load_audio_file(self, file_path: str) -> bool:
        """Loads a user-specified audio file."""
        with self._lock:
            success = self.file_reader.load_file(file_path)
            if success:
                self.source_mode = "file"
            return success

    def start(self, input_device: Optional[int] = None, output_device: Optional[int] = None):
        """Starts real-time audio stream."""
        if self.is_running or sd is None:
            return

        self.is_running = True
        try:
            # First attempt: full duplex input & output
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=(1, 1),
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
            self.stream_has_input = True
        except Exception as e:
            # Fallback: output-only stream (for environments where mic is unavailable/restricted)
            try:
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=output_device,
                    channels=1,
                    dtype='float32',
                    callback=self._audio_callback_output_only
                )
                self.stream.start()
                self.stream_has_input = False
                if self.source_mode == "mic":
                    self.source_mode = "demo"
            except Exception as e_out:
                self.is_running = False
                raise RuntimeError(f"Could not open audio device: {e_out}")

    def stop(self):
        """Stops real-time audio stream."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Full-duplex real-time audio callback."""
        if status:
            self.buffer_underruns += 1

        in_chunk = indata[:, 0].astype(np.float64) * self.mic_input_gain
        self._process_and_output(in_chunk, outdata, frames)

    def _audio_callback_output_only(self, outdata, frames, time_info, status):
        """Output-only real-time audio callback."""
        if status:
            self.buffer_underruns += 1

        in_chunk = np.zeros(frames, dtype=np.float64)
        self._process_and_output(in_chunk, outdata, frames)

    def _process_and_output(self, mic_chunk: np.ndarray, outdata: np.ndarray, frames: int):
        with self._lock:
            # 1. Select Input Signal
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

            self.peak_in_level = float(np.max(np.abs(raw_input))) if len(raw_input) > 0 else 0.0

            # 2. DSP Chain
            # Stage A: Noise Reduction
            x_nr = self.noise_reducer.process_block(raw_input)

            # Stage B: Standalone Filter (LP / HP / BP / Notch)
            x_filt = self.filter.process(x_nr)

            # Stage C: Multi-Band Equalizer (8 bands)
            x_eq = self.equalizer.process(x_filt)

            # Stage D: Master Volume & Soft Limiting
            processed = x_eq * self.master_gain

            # Soft saturation limiter to prevent digital clipping harshness
            processed = np.tanh(processed)

            self.peak_out_level = float(np.max(np.abs(processed))) if len(processed) > 0 else 0.0

            # 3. Audio Output
            if self.monitor_output:
                outdata[:, 0] = processed.astype(np.float32)
            else:
                outdata[:, 0] = 0.0

            # 4. Push to visualizer queue (non-blocking)
            if not self.viz_queue.full():
                try:
                    self.viz_queue.put_nowait({
                        'input': raw_input.copy(),
                        'output': processed.copy(),
                        'timestamp': time.time()
                    })
                except queue.Full:
                    pass

    def get_latest_visualization_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Pulls the freshest audio frame available for the GUI visualizer."""
        latest = None
        while not self.viz_queue.empty():
            try:
                latest = self.viz_queue.get_nowait()
            except queue.Empty:
                break
        return latest
