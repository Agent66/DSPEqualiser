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
    AutoHumRemover,
    compute_snr_improvement,
    measure_dsp_latency,
    run_full_dsp_benchmark,
    compute_spectrum,
    detect_pitch,
    VoiceActivityDetector
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
        self.auto_hum_remover = AutoHumRemover(sample_rate=sample_rate, fundamental=50.0, harmonics=[50.0, 100.0, 150.0], q=25.0)
        self.noise_reducer = SpectralNoiseReducer(sample_rate=sample_rate, block_size=block_size, alpha=2.0, beta=0.03)

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
        self.mic_monitor_speakers = False  # Safe default: mute mic to speakers to prevent acoustic feedback loop
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self.raw_mic_peak = 0.0
        self.raw_mic_db = -100.0
        self.mic_is_speech = False
        self.vad = VoiceActivityDetector(sample_rate=sample_rate, hangover_frames=4)

        # Test injection controls (applies across any audio source for live grading demo)
        self.inject_hum = False
        self.hum_level = 0.22
        self.hum_phase = 0.0
        self.inject_noise = False
        self.noise_level = 0.10

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
        self.dsp_duration_sec = 0.00085
        self.latest_snr_improvement_db = 0.0
        self.latest_latency = measure_dsp_latency(block_size, sample_rate, 0.00085)

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

    def get_input_devices(self) -> List[Dict[str, Any]]:
        """Returns clean list of available microphone input devices."""
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
            inputs = []
            for idx, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    inputs.append({'index': idx, 'name': d['name']})
            return inputs
        except Exception:
            return []

    def set_input_device(self, device_index: Optional[int]):
        """Sets active microphone device and seamlessly restarts stream if running."""
        with self._lock:
            self.input_device = device_index
        if self.is_running:
            self.stop()
            self.start(input_device=self.input_device, output_device=self.output_device)

    def set_mic_speaker_monitor(self, enable: bool):
        """Toggles whether live mic audio plays through speakers (False prevents feedback)."""
        with self._lock:
            self.mic_monitor_speakers = bool(enable)

    def set_source(self, source: str):
        """Switches audio source ('mic', 'generator', 'file', 'demo')."""
        with self._lock:
            self.source_mode = source.lower()
        # If switching to mic and stream currently has no input, attempt to reopen stream with input
        if self.source_mode == "mic" and self.is_running and not self.stream_has_input:
            try:
                self.stop()
                self.start(input_device=self.input_device, output_device=self.output_device)
            except Exception:
                pass

    def set_master_gain(self, gain: float):
        """Sets master output volume (0.0 to 1.5)."""
        self.master_gain = float(np.clip(gain, 0.0, 1.5))

    def set_mic_gain(self, gain: float):
        """Sets microphone preamp boost/cut (0.0 to 5.0)."""
        self.mic_input_gain = float(np.clip(gain, 0.0, 5.0))

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

        if input_device is not None:
            self.input_device = input_device
        if output_device is not None:
            self.output_device = output_device

        self.is_running = True
        try:
            # First attempt: full duplex 1 input ch (mic), 2 output ch (stereo speakers/headphones)
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=(1, 2),
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
            self.stream_has_input = True
        except Exception:
            try:
                # Second attempt: full duplex 1 input, 1 output
                self.stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=(self.input_device, self.output_device),
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
                        device=self.output_device,
                        channels=2,
                        dtype='float32',
                        callback=self._audio_callback_output_only
                    )
                    self.stream.start()
                    self.stream_has_input = False
                    if self.source_mode == "mic":
                        self.source_mode = "demo"
                except Exception:
                    try:
                        self.stream = sd.OutputStream(
                            samplerate=self.sample_rate,
                            blocksize=self.block_size,
                            device=self.output_device,
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
        self.raw_mic_peak = float(np.max(np.abs(in_chunk))) if len(in_chunk) > 0 else 0.0
        self.raw_mic_db = float(20.0 * np.log10(max(self.raw_mic_peak, 1e-5)))
        try:
            is_sp, _, _ = self.vad.process_block(in_chunk)
            self.mic_is_speech = is_sp
        except Exception:
            self.mic_is_speech = (self.raw_mic_db > -42.0)
        self._process_and_output(in_chunk, outdata, frames)

    def _audio_callback_output_only(self, outdata, frames, time_info, status):
        """Output-only real-time audio callback."""
        if status:
            self.buffer_underruns += 1

        self.raw_mic_peak = 0.0
        self.raw_mic_db = -100.0
        self.mic_is_speech = False
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

            # Test injection: 50 Hz mains hum & harmonics (for grading demonstration)
            if self.inject_hum:
                t = np.arange(frames) / float(self.sample_rate)
                w50 = 2.0 * np.pi * 50.0
                w100 = 2.0 * np.pi * 100.0
                w150 = 2.0 * np.pi * 150.0
                hum = (
                    self.hum_level * np.sin(self.hum_phase + w50 * t) +
                    (self.hum_level * 0.45) * np.sin(2.0 * self.hum_phase + w100 * t) +
                    (self.hum_level * 0.25) * np.sin(3.0 * self.hum_phase + w150 * t)
                )
                self.hum_phase = (self.hum_phase + w50 * (frames / float(self.sample_rate))) % (2.0 * np.pi)
                raw_input = raw_input + hum

            # Test injection: background ambient noise (for spectral subtraction grading demo)
            if self.inject_noise:
                noise = self.noise_level * np.random.normal(0.0, 1.0, frames)
                raw_input = raw_input + noise

            self.peak_in_level = float(np.max(np.abs(raw_input))) if len(raw_input) > 0 else 0.0

            # 2. Real-Time DSP Chain (Performance Timed)
            t_dsp_start = time.perf_counter()

            # Stage A: Auto Hum Removal (50 Hz, 100 Hz, 150 Hz notch filters)
            x_hum = self.auto_hum_remover.process_block(raw_input)

            # Stage B: Spectral Subtraction Noise Reduction
            x_nr = self.noise_reducer.process_block(x_hum)

            # Stage C: Standalone Filter (LP / HP / BP / Notch)
            x_filt = self.filter.process(x_nr)

            # Stage D: Multi-Band Equalizer (8 bands)
            x_eq = self.equalizer.process(x_filt)

            # Stage E: Master Volume & Soft Limiting
            processed = x_eq * self.master_gain

            # Soft saturation limiter to prevent digital clipping harshness
            processed = np.tanh(processed)

            # Measure CPU DSP Execution Time
            self.dsp_duration_sec = time.perf_counter() - t_dsp_start
            self.peak_out_level = float(np.max(np.abs(processed))) if len(processed) > 0 else 0.0

            # Compute real-time SNR improvement & end-to-end latency
            self.latest_snr_improvement_db = compute_snr_improvement(raw_input, processed)
            self.latest_latency = measure_dsp_latency(self.block_size, self.sample_rate, self.dsp_duration_sec)

            # 3. Audio Output
            if self.source_mode == "mic":
                should_monitor = self.mic_monitor_speakers and self.monitor_output
            else:
                should_monitor = self.monitor_output

            if should_monitor:
                proc_f32 = processed.astype(np.float32)
                for ch in range(outdata.shape[1]):
                    outdata[:, ch] = proc_f32
            else:
                outdata.fill(0.0)

            # 4. Push to visualizer queue (non-blocking)
            if not self.viz_queue.full():
                try:
                    self.viz_queue.put_nowait({
                        'input': raw_input.copy(),
                        'output': processed.copy(),
                        'timestamp': time.time(),
                        'snr_improvement_db': self.latest_snr_improvement_db,
                        'latency': self.latest_latency,
                        'hum_status': self.auto_hum_remover.detected_harmonics,
                        'hum_detected': self.auto_hum_remover.hum_detected,
                        'hum_suppression_db': self.auto_hum_remover.measured_suppression_db,
                        'noise_calibrating': self.noise_reducer.is_calibrating,
                        'noise_progress': self.noise_reducer.calibration_progress,
                        'noise_drop_db': self.noise_reducer.measured_noise_drop_db,
                        'source_mode': self.source_mode,
                        'stream_has_input': self.stream_has_input,
                        'mic_peak': self.raw_mic_peak,
                        'mic_db': self.raw_mic_db,
                        'mic_is_speech': self.mic_is_speech,
                        'mic_monitor': self.mic_monitor_speakers,
                        'mic_gain': self.mic_input_gain
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
