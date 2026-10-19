"""
Unit tests for spequaliser DSP components.
"""

import unittest
import numpy as np
import scipy.signal as signal

from spequaliser.dsp import (
    STANDARD_SAMPLE_RATE,
    DEFAULT_BLOCK_SIZE,
    compute_spectrum,
    get_window,
    StandaloneFilter,
    MultiBandEqualizer,
    detect_pitch,
    SpectralNoiseReducer,
    AutoHumRemover,
    compute_snr_improvement,
    measure_dsp_latency,
    run_full_dsp_benchmark,
    detect_vad,
    VoiceActivityDetector,
    compute_energy,
    compute_zcr
)
from spequaliser.generators import SignalGenerator, generate_synthetic_demo_track


class TestDSP(unittest.TestCase):

    def test_windowing(self):
        w_hann = get_window("hann", 1024)
        w_rect = get_window("rectangular", 1024)
        self.assertEqual(len(w_hann), 1024)
        self.assertEqual(len(w_rect), 1024)
        self.assertAlmostEqual(w_rect[500], 1.0)
        self.assertTrue(w_hann[0] < 0.05)
        self.assertTrue(w_hann[512] > 0.95)

    def test_fft_peak(self):
        fs = STANDARD_SAMPLE_RATE
        t = np.arange(2048) / float(fs)
        x = np.sin(2.0 * np.pi * 1000.0 * t)
        freqs, mag_db, peak_f, peak_db = compute_spectrum(x, sample_rate=fs, window_type="hann")
        self.assertAlmostEqual(peak_f, 1000.0, delta=3.0)
        self.assertTrue(peak_db > -3.0)

    def test_lowpass_filter(self):
        filt = StandaloneFilter(sample_rate=STANDARD_SAMPLE_RATE)
        filt.set_params(mode="lowpass", cutoff=1000.0, q=1.0, order=4)
        resp = filt.get_frequency_response(np.array([100.0, 1000.0, 5000.0]))
        # Passband should be ~0 dB
        self.assertAlmostEqual(resp[0], 0.0, delta=0.5)
        # Cutoff should be ~ -3 dB
        self.assertAlmostEqual(resp[1], -3.0, delta=1.5)
        # Stopband should be strongly attenuated
        self.assertTrue(resp[2] < -25.0)

    def test_highpass_filter(self):
        filt = StandaloneFilter(sample_rate=STANDARD_SAMPLE_RATE)
        filt.set_params(mode="highpass", cutoff=1000.0, q=1.0, order=4)
        resp = filt.get_frequency_response(np.array([100.0, 1000.0, 5000.0]))
        self.assertTrue(resp[0] < -25.0)
        self.assertAlmostEqual(resp[1], -3.0, delta=1.5)
        self.assertAlmostEqual(resp[2], 0.0, delta=0.5)

    def test_notch_filter(self):
        filt = StandaloneFilter(sample_rate=STANDARD_SAMPLE_RATE)
        filt.set_params(mode="notch", cutoff=60.0, q=2.0)
        resp = filt.get_frequency_response(np.array([60.0, 500.0]))
        self.assertTrue(resp[0] < -20.0)
        self.assertAlmostEqual(resp[1], 0.0, delta=0.5)

    def test_equalizer_boost(self):
        eq = MultiBandEqualizer(sample_rate=STANDARD_SAMPLE_RATE)
        eq.set_band_gain(3, 6.0)  # 1 kHz band
        resp = eq.get_frequency_response(np.array([1000.0]))
        self.assertAlmostEqual(resp[0], 6.0, delta=0.8)

    def test_pitch_detection_a4(self):
        fs = STANDARD_SAMPLE_RATE
        t = np.arange(2048) / float(fs)
        x = np.sin(2.0 * np.pi * 440.0 * t)
        pitch_hz, note_name, cents = detect_pitch(x, sample_rate=fs)
        self.assertEqual(note_name, "A4")
        self.assertAlmostEqual(pitch_hz, 440.0, delta=1.0)
        self.assertAlmostEqual(cents, 0.0, delta=5.0)

    def test_signal_generator_continuity(self):
        gen = SignalGenerator(sample_rate=STANDARD_SAMPLE_RATE)
        gen.set_frequency(440.0)
        block1 = gen.generate_block(1024)
        block2 = gen.generate_block(1024)
        self.assertEqual(len(block1), 1024)
        self.assertEqual(len(block2), 1024)
        # Check boundary continuity between last sample of block1 and first of block2
        combined = np.concatenate([block1, block2])
        diff = np.diff(combined)
        # Gradient should be smooth without sudden step
        self.assertTrue(np.max(np.abs(diff)) < 0.15)

    def test_vad_detection(self):
        fs = STANDARD_SAMPLE_RATE
        # Silence test
        silence = np.zeros(2048)
        is_sp, e_db, zcr = detect_vad(silence, sample_rate=fs)
        self.assertFalse(is_sp)
        self.assertEqual(zcr, 0.0)
        self.assertTrue(e_db <= -99.0)

        # Voiced speech test (220 Hz tone at -9 dB)
        t = np.arange(2048) / float(fs)
        tone = 0.5 * np.sin(2.0 * np.pi * 220.0 * t)
        is_sp_tone, e_db_tone, zcr_tone = detect_vad(tone, sample_rate=fs)
        self.assertTrue(is_sp_tone)
        self.assertTrue(zcr_tone < 0.05)

        # Unvoiced fricative test (high frequency noise)
        np.random.seed(42)
        noise = np.random.normal(0, 1, 2048)
        b, a = signal.butter(2, [3000.0, 7000.0], btype='bandpass', fs=fs)
        fricative = 0.05 * signal.lfilter(b, a, noise)
        is_sp_fric, e_db_fric, zcr_fric = detect_vad(fricative, sample_rate=fs)
        self.assertTrue(is_sp_fric)
        self.assertTrue(zcr_fric >= 0.12)

    def test_vad_telecom_hangover(self):
        fs = STANDARD_SAMPLE_RATE
        vad = VoiceActivityDetector(sample_rate=fs, hangover_frames=3)
        t = np.arange(2048) / float(fs)
        speech_chunk = 0.4 * np.sin(2.0 * np.pi * 300.0 * t)
        silence_chunk = np.zeros(2048)

        # Frame 1: Active speech
        act1, _, _ = vad.process_block(speech_chunk)
        self.assertTrue(act1)

        # Frames 2, 3, 4: Silence, but preserved by 3-frame hangover
        act2, _, _ = vad.process_block(silence_chunk)
        act3, _, _ = vad.process_block(silence_chunk)
        act4, _, _ = vad.process_block(silence_chunk)
        self.assertTrue(act2)
        self.assertTrue(act3)
        self.assertTrue(act4)

        # Frame 5: Hangover expired
        act5, _, _ = vad.process_block(silence_chunk)
        self.assertFalse(act5)

    def test_auto_hum_removal(self):
        fs = STANDARD_SAMPLE_RATE
        remover = AutoHumRemover(sample_rate=fs, q=25.0)
        remover.enabled = True
        n = 2048

        # Stream continuous contiguous frames to confirm detection and steady-state notch attenuation
        for i in range(8):
            t = (i * n + np.arange(n)) / float(fs)
            clean_audio = 0.50 * np.sin(2.0 * np.pi * 1000.0 * t)
            hum_50 = 0.30 * np.sin(2.0 * np.pi * 50.0 * t)
            noisy_audio = clean_audio + hum_50
            cleaned = remover.process_block(noisy_audio)

        self.assertTrue(remover.hum_detected)
        self.assertTrue(remover.detected_harmonics[50.0]["detected"])

        # Check notch suppression on steady-state block
        w = signal.windows.hann(n, sym=False)
        fft_in = np.abs(np.fft.rfft(noisy_audio * w, n=8192))
        fft_out = np.abs(np.fft.rfft(cleaned * w, n=8192))
        freqs = np.fft.rfftfreq(8192, 1.0 / fs)
        bin_50 = np.argmin(np.abs(freqs - 50.0))
        supp_db = 20.0 * np.log10(max(fft_in[bin_50], 1e-9) / max(fft_out[bin_50], 1e-9))
        self.assertTrue(supp_db >= 20.0)

    def test_spectral_subtraction_calibration(self):
        fs = STANDARD_SAMPLE_RATE
        denoiser = SpectralNoiseReducer(sample_rate=fs, block_size=2048, alpha=2.0, beta=0.03)
        np.random.seed(42)
        noise_1s = 0.2 * np.random.normal(0, 1, fs)
        denoiser.record_noise_sample(noise_1s)
        self.assertTrue(denoiser.enabled)
        self.assertIsNotNone(denoiser.noise_profile)

        # Clean a noisy block
        t = np.arange(2048) / float(fs)
        signal_tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        noisy = signal_tone + 0.2 * np.random.normal(0, 1, 2048)
        cleaned = denoiser.process_block(noisy)
        snr_gain = compute_snr_improvement(noisy, cleaned)
        self.assertTrue(snr_gain > 0.0)

    def test_measured_results_and_latency(self):
        lat = measure_dsp_latency(block_size=2048, sample_rate=STANDARD_SAMPLE_RATE)
        self.assertIn("buffer_latency_ms", lat)
        self.assertIn("dsp_compute_ms", lat)
        self.assertIn("total_latency_ms", lat)
        self.assertTrue(lat["total_latency_ms"] < 60.0)

        bench = run_full_dsp_benchmark(sample_rate=STANDARD_SAMPLE_RATE, block_size=2048)
        self.assertTrue(bench["hum_50hz_suppression_db"] > 20.0)
        self.assertTrue(bench["snr_improvement_db"] > 4.0)


if __name__ == "__main__":
    unittest.main()

