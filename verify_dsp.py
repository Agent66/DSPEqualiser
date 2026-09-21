"""
Comprehensive DSP Verification Suite.

Mathematically validates every core concept and objective from the DSP presentation:
1. FFT Frequency Accuracy with sub-bin parabolic interpolation (Slide 9)
2. Windowing & Spectral Leakage Suppression (Hann vs Rectangular) (Slide 6)
3. Digital Filter Performance (Low-Pass, High-Pass, Band-Pass, Notch) (Slides 4, 6, 9)
4. Multi-Band Equalizer Frequency Reshaping & State Continuity (Slides 4, 7)
5. Pitch Detection Accuracy & Musical Note Identification (Slides 4, 10)
6. Spectral Subtraction Noise Reduction SNR Enhancement (Slides 4, 10)
7. Voice Activity Detection (VAD) Speech/Silence Classification & Telecom DTX (Energy + ZCR)
"""

import sys
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
    EQ_CENTER_FREQUENCIES,
    detect_vad,
    VoiceActivityDetector,
    compute_energy,
    compute_zcr
)
from spequaliser.generators import SignalGenerator


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def test_fft_accuracy():
    print_header("Test 1: FFT Frequency Estimation Accuracy (Slide 9)")
    test_freqs = [120.0, 440.0, 1000.0, 3520.0, 7500.0]
    n_samples = DEFAULT_BLOCK_SIZE
    fs = STANDARD_SAMPLE_RATE

    all_passed = True
    for f_target in test_freqs:
        t = np.arange(n_samples) / float(fs)
        # Synthesize pure sine tone
        x = np.sin(2.0 * np.pi * f_target * t)
        freqs, mag_db, peak_f, peak_db = compute_spectrum(x, sample_rate=fs, window_type="hann")

        error_hz = abs(peak_f - f_target)
        error_pct = (error_hz / f_target) * 100.0

        status = "PASS" if error_pct < 0.5 else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  Target: {f_target:7.1f} Hz | Detected: {peak_f:7.1f} Hz | Error: {error_hz:5.2f} Hz ({error_pct:4.2f}%) | Peak dB: {peak_db:5.1f} | [{status}]")

    assert all_passed, "FFT peak accuracy test failed!"
    print(">>> FFT Accuracy Test: PASSED (Sub-bin accuracy < 0.5%)")


def test_windowing_spectral_leakage():
    print_header("Test 2: Windowing & Spectral Leakage Suppression (Slide 6)")
    # Test on an off-grid frequency so spectral leakage is prominent with rectangular window
    fs = STANDARD_SAMPLE_RATE
    n_samples = DEFAULT_BLOCK_SIZE
    t = np.arange(n_samples) / float(fs)
    f_offgrid = 443.7  # Non-integer number of cycles in 2048 samples

    x = np.sin(2.0 * np.pi * f_offgrid * t)

    _, mag_rect, _, _ = compute_spectrum(x, sample_rate=fs, window_type="rectangular")
    _, mag_hann, _, _ = compute_spectrum(x, sample_rate=fs, window_type="hann")
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Measure stopband sidelobe floor between 2 kHz and 10 kHz
    stopband_mask = (freqs >= 2000.0) & (freqs <= 10000.0)
    floor_rect = np.mean(mag_rect[stopband_mask])
    floor_hann = np.mean(mag_hann[stopband_mask])
    suppression_gain = floor_rect - floor_hann

    print(f"  Off-grid frequency: {f_offgrid} Hz")
    print(f"  Rectangular Window Stopband Floor: {floor_rect:5.1f} dB")
    print(f"  Hann Window Stopband Floor:        {floor_hann:5.1f} dB")
    print(f"  Hann Sidelobe Suppression:         {suppression_gain:5.1f} dB improvement")

    assert suppression_gain > 20.0, f"Hann window did not provide expected sidelobe suppression: {suppression_gain} dB"
    print(">>> Windowing Spectral Leakage Test: PASSED (Hann window suppressed leakage by >20 dB)")


def test_digital_filters():
    print_header("Test 3: Digital Filters (Low-Pass, High-Pass, Band-Pass, Notch) (Slides 4 & 6)")
    fs = STANDARD_SAMPLE_RATE
    n_samples = 16384
    t = np.arange(n_samples) / float(fs)

    def measure_gain_db(filt: StandaloneFilter, test_freq: float) -> float:
        x = np.sin(2.0 * np.pi * test_freq * t)
        filt.zi = None  # reset state
        y = filt.process(x)
        # Measure steady state (last 4096 samples)
        rms_in = np.sqrt(np.mean(x[-4096:] ** 2))
        rms_out = np.sqrt(np.mean(y[-4096:] ** 2))
        return 20.0 * np.log10(max(rms_out, 1e-9) / rms_in)

    filt = StandaloneFilter(sample_rate=fs)

    # 1. Low-Pass Filter at 1000 Hz
    filt.set_params(mode="lowpass", cutoff=1000.0, q=1.0, order=4)
    g_pass = measure_gain_db(filt, 200.0)
    g_stop = measure_gain_db(filt, 4000.0)
    print(f"  [Low-Pass 1000 Hz] Passband (200 Hz): {g_pass:5.1f} dB | Stopband (4000 Hz): {g_stop:5.1f} dB")
    assert abs(g_pass) < 1.0, "LP Passband gain deviated from unity"
    assert g_stop < -20.0, "LP Stopband attenuation insufficient"

    # 2. High-Pass Filter at 2000 Hz
    filt.set_params(mode="highpass", cutoff=2000.0, q=1.0, order=4)
    g_stop = measure_gain_db(filt, 300.0)
    g_pass = measure_gain_db(filt, 6000.0)
    print(f"  [High-Pass 2000 Hz] Stopband (300 Hz): {g_stop:5.1f} dB | Passband (6000 Hz): {g_pass:5.1f} dB")
    assert g_stop < -20.0, "HP Stopband attenuation insufficient"
    assert abs(g_pass) < 1.0, "HP Passband gain deviated from unity"

    # 3. Band-Pass Filter centered at 1000 Hz
    filt.set_params(mode="bandpass", cutoff=1000.0, q=2.0)
    g_center = measure_gain_db(filt, 1000.0)
    g_low = measure_gain_db(filt, 250.0)
    g_high = measure_gain_db(filt, 4000.0)
    print(f"  [Band-Pass 1000 Hz] Center (1000 Hz): {g_center:5.1f} dB | Low Rejection (250 Hz): {g_low:5.1f} dB | High Rejection (4000 Hz): {g_high:5.1f} dB")
    assert abs(g_center) < 2.5, "BP Center gain deviated"
    assert g_low < -10.0 and g_high < -10.0, "BP Rejection insufficient"

    # 4. Notch Filter at 60 Hz (Mains Hum)
    filt.set_params(mode="notch", cutoff=60.0, q=2.0)
    g_notch = measure_gain_db(filt, 60.0)
    g_pass = measure_gain_db(filt, 1000.0)
    print(f"  [Notch 60 Hz] Notch Rejection (60 Hz): {g_notch:5.1f} dB | Passband (1000 Hz): {g_pass:5.1f} dB")
    assert g_notch < -20.0, "Notch attenuation insufficient"
    assert abs(g_pass) < 0.5, "Notch passband affected"

    print(">>> Digital Filters Test: PASSED (All 4 filter modes met specifications)")


def test_equalizer_bands_and_continuity():
    print_header("Test 4: Multi-Band Equalizer & State Continuity (Slides 4 & 7)")
    fs = STANDARD_SAMPLE_RATE
    eq = MultiBandEqualizer(sample_rate=fs)

    n_samples = 4096
    t = np.arange(n_samples) / float(fs)

    # Test 1 kHz band boost (+6 dB)
    eq.reset_gains()
    eq.set_band_gain(3, 6.0)  # Band 3 is 1 kHz

    x = np.sin(2.0 * np.pi * 1000.0 * t)
    y = eq.process(x)
    rms_in = np.sqrt(np.mean(x[2048:] ** 2))
    rms_out = np.sqrt(np.mean(y[2048:] ** 2))
    boost_gain = 20.0 * np.log10(rms_out / rms_in)
    print(f"  1 kHz Band (+6.0 dB Target): Measured Boost = {boost_gain:4.2f} dB")
    assert abs(boost_gain - 6.0) < 0.6, "EQ Boost deviated from +6 dB"

    # Test 1 kHz band cut (-6 dB)
    eq.set_band_gain(3, -6.0)
    y_cut = eq.process(x)
    rms_cut = np.sqrt(np.mean(y_cut[2048:] ** 2))
    cut_gain = 20.0 * np.log10(rms_cut / rms_in)
    print(f"  1 kHz Band (-6.0 dB Target): Measured Cut   = {cut_gain:4.2f} dB")
    assert abs(cut_gain - (-6.0)) < 0.6, "EQ Cut deviated from -6 dB"

    # Test Block State Continuity (Zero Boundary Transient across consecutive blocks)
    block1 = x[:1024]
    block2 = x[1024:2048]
    eq.reset_gains()
    eq.set_band_gain(1, 4.0)
    eq.set_band_gain(5, 5.0)

    out1 = eq.process(block1)
    out2 = eq.process(block2)

    # Process all at once
    eq.reset_gains()
    eq.set_band_gain(1, 4.0)
    eq.set_band_gain(5, 5.0)
    out_all = eq.process(x[:2048])

    out_concatenated = np.concatenate([out1, out2])
    max_diff = np.max(np.abs(out_all - out_concatenated))
    print(f"  Block Continuity Max Absolute Error across boundary: {max_diff:.2e}")
    assert max_diff < 1e-12, "Block processing caused boundary discontinuity!"

    print(">>> Multi-Band Equalizer Test: PASSED (Accurate gains & zero boundary transients)")


def test_pitch_detection():
    print_header("Test 5: Pitch Detection Accuracy (Autocorrelation / YIN) (Slides 4 & 10)")
    fs = STANDARD_SAMPLE_RATE
    n_samples = 2048
    t = np.arange(n_samples) / float(fs)

    notes_to_test = [
        ("A2", 110.0),
        ("C3", 130.81),
        ("A3", 220.0),
        ("C4", 261.63),
        ("A4", 440.0),
        ("E4", 329.63),
        ("A5", 880.0)
    ]

    all_passed = True
    for expected_note, freq in notes_to_test:
        x = np.sin(2.0 * np.pi * freq * t)
        detected_hz, detected_note, cents = detect_pitch(x, sample_rate=fs)
        freq_err = abs(detected_hz - freq)

        status = "PASS" if (detected_note == expected_note and freq_err < 1.0) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  Target: {expected_note} ({freq:6.2f} Hz) | Detected: {detected_note} ({detected_hz:6.2f} Hz) | Cents: {cents:4.1f} | [{status}]")

    assert all_passed, "Pitch detection failed on test frequencies"
    print(">>> Pitch Detection Test: PASSED (Sub-Hz accuracy and exact note mapping)")


def test_noise_reduction():
    print_header("Test 6: Spectral Subtraction Noise Reduction (Slides 4 & 10)")
    fs = STANDARD_SAMPLE_RATE
    n_samples = 2048
    t = np.arange(n_samples) / float(fs)

    # Pure speech/signal tone
    clean_signal = 0.5 * np.sin(2.0 * np.pi * 500.0 * t)

    # Stationary white noise
    np.random.seed(42)
    noise = 0.25 * np.random.normal(0, 1, n_samples)
    noisy_signal = clean_signal + noise

    reducer = SpectralNoiseReducer(block_size=n_samples, alpha=2.0, beta=0.02)
    reducer.noise_profile = np.abs(np.fft.rfft(noise * signal.windows.hann(n_samples, sym=False)))
    reducer.enabled = True

    denoised_signal = reducer.process_block(noisy_signal)

    # Measure noise power in high frequencies (> 2000 Hz) where pure signal has 0 energy
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)
    high_mask = freqs > 2000.0

    fft_noisy = np.abs(np.fft.rfft(noisy_signal))
    fft_denoised = np.abs(np.fft.rfft(denoised_signal))

    noise_before = np.mean(fft_noisy[high_mask] ** 2)
    noise_after = np.mean(fft_denoised[high_mask] ** 2)
    noise_suppression_db = 10.0 * np.log10(noise_before / max(noise_after, 1e-9))

    print(f"  High-Frequency Noise Suppression: {noise_suppression_db:4.1f} dB reduction")
    assert noise_suppression_db > 10.0, "Noise suppression did not achieve >10 dB attenuation"
    print(">>> Noise Reduction Test: PASSED (Spectral subtraction achieved >10 dB noise floor drop)")


def test_voice_activity_detection():
    print_header("Test 7: Voice Activity Detection (VAD) & Telecom DTX (Energy + ZCR)")
    fs = STANDARD_SAMPLE_RATE
    n_samples = DEFAULT_BLOCK_SIZE
    t = np.arange(n_samples) / float(fs)

    # 1. Silence & Ambient Noise (< -45 dB)
    silence = np.zeros(n_samples)
    is_sp_silence, e_silence, zcr_silence = detect_vad(silence, sample_rate=fs)
    print(f"  [Silence]        Energy: {e_silence:5.1f} dB | ZCR: {zcr_silence:5.3f} | Speech Detected: {is_sp_silence}")
    assert not is_sp_silence, "Pure silence was incorrectly flagged as speech!"

    np.random.seed(123)
    ambient_noise = 0.001 * np.random.normal(0, 1, n_samples)
    is_sp_amb, e_amb, zcr_amb = detect_vad(ambient_noise, sample_rate=fs)
    print(f"  [Ambient Noise]  Energy: {e_amb:5.1f} dB | ZCR: {zcr_amb:5.3f} | Speech Detected: {is_sp_amb}")
    assert not is_sp_amb, "Ambient noise floor was incorrectly flagged as speech!"

    # 2. Voiced Speech Simulation (Harmonic tone, fundamental 220 Hz + harmonics, high energy, low ZCR)
    voiced = 0.35 * np.sin(2.0 * np.pi * 220.0 * t) + 0.15 * np.sin(2.0 * np.pi * 440.0 * t) + 0.08 * np.sin(2.0 * np.pi * 660.0 * t)
    is_sp_voiced, e_voiced, zcr_voiced = detect_vad(voiced, sample_rate=fs)
    print(f"  [Voiced Speech]  Energy: {e_voiced:5.1f} dB | ZCR: {zcr_voiced:5.3f} | Speech Detected: {is_sp_voiced}")
    assert is_sp_voiced, "Voiced speech was not detected!"
    assert zcr_voiced < 0.10, "Voiced speech ZCR was unexpectedly high"

    # 3. Unvoiced Speech Simulation (Fricative /s/ sound: high-frequency filtered noise, moderate energy, high ZCR)
    white_noise = np.random.normal(0, 1, n_samples)
    # Bandpass filter between 3500 Hz and 8000 Hz
    b_fric, a_fric = signal.butter(2, [3500.0, 8000.0], btype='bandpass', fs=fs)
    fricative = 0.06 * signal.lfilter(b_fric, a_fric, white_noise)
    is_sp_unvoiced, e_unvoiced, zcr_unvoiced = detect_vad(fricative, sample_rate=fs)
    print(f"  [Unvoiced Speech] Energy: {e_unvoiced:5.1f} dB | ZCR: {zcr_unvoiced:5.3f} | Speech Detected: {is_sp_unvoiced}")
    assert is_sp_unvoiced, "Unvoiced speech fricative was not detected!"
    assert zcr_unvoiced >= 0.12, "Unvoiced fricative did not show expected high ZCR"

    # 4. Telecom Hangover Verification (GSM/AMR/WebRTC DTX syllable bridging)
    hangover_frames = 5
    vad = VoiceActivityDetector(sample_rate=fs, hangover_frames=hangover_frames)

    # Frame 1: Active speech
    act_1, _, _ = vad.process_block(voiced)
    assert act_1, "Stateful VAD failed to trigger on speech"

    # Frames 2 to 6: Sudden silence -> MUST stay active for 5 hangover frames
    for f in range(1, hangover_frames + 1):
        act_h, _, _ = vad.process_block(silence)
        print(f"  [Hangover Frame {f}/{hangover_frames}] Speech Active: {act_h} (Preventing word ending clipping)")
        assert act_h, f"Hangover frame {f} prematurely dropped speech!"

    # Frame 7: Hangover expired -> MUST drop to silence
    act_expired, _, _ = vad.process_block(silence)
    print(f"  [Hangover Expired] Speech Active: {act_expired} (Successfully entered bandwidth-saving DTX mode)")
    assert not act_expired, "VAD failed to drop to silence after hangover expiration"

    # 5. Bandwidth / Discontinuous Transmission (DTX) Savings Simulation
    # Alternating speech (8 frames) and pauses (17 frames) = 25 frames total
    vad.reset()
    transmitted_count = 0
    total_frames = 25
    for i in range(total_frames):
        chunk = voiced if (i < 8) else silence
        tx, _, _ = vad.process_block(chunk)
        if tx:
            transmitted_count += 1

    savings_pct = (1.0 - transmitted_count / float(total_frames)) * 100.0
    print(f"  Conversational Telecom DTX: {transmitted_count}/{total_frames} frames transmitted ({savings_pct:.1f}% bandwidth saved)")
    assert savings_pct > 35.0, "DTX bandwidth savings did not meet minimum threshold"

    print(">>> Voice Activity Detection Test: PASSED (Energy + ZCR + Hangover validated)")


def run_all():
    print("\n" + "#" * 70)
    print("  RUNNING REAL-TIME AUDIO EQUALIZER & DSP VERIFICATION SUITE")
    print("#" * 70)

    test_fft_accuracy()
    test_windowing_spectral_leakage()
    test_digital_filters()
    test_equalizer_bands_and_continuity()
    test_pitch_detection()
    test_noise_reduction()
    test_voice_activity_detection()

    print("\n" + "=" * 70)
    print("  ALL DSP TESTS PASSED SUCCESSFULLY! (100% SPEC COMPLIANCE)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all()
