# Real-Time Audio Equalizer & Spectrum Analyzer
### A Digital Signal Processing (DSP) & Secure Audio Streaming Suite in Python

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DSP Suite](https://img.shields.io/badge/DSP-Real--Time-00f2fe.svg)](spequaliser/dsp.py)
[![Security](https://img.shields.io/badge/Security-AES--256--GCM-emerald.svg)](crypto_utils.py)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-24%2F24%20Passing-brightgreen.svg)](tests/)
[![DSP Verification](https://img.shields.io/badge/DSP%20Specs-9%2F9%20Verified-brightgreen.svg)](verify_dsp.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance real-time audio processing system implementing digital signal processing (DSP) fundamentals, advanced audio restoration features, dual-trace visual analytics, and authenticated encrypted network streaming:

- **Core DSP Pipeline**: Continuous-time sampling (44.1 kHz), Hann/Hamming windowing (>57 dB sidelobe suppression), single-sided $N$-point FFT spectrum analysis with sub-bin parabolic peak interpolation (<0.2% error), 8-band graphic equalizer with zero-transient Second-Order Section (SOS) state continuity, and standalone Butterworth digital filters.
- **Advanced Audio Restoration**: Automatic 50 Hz Indian/European mains hum and harmonic notch filtering (>24 dB suppression with 0.00 dB passband loss) and 1-second calibrated Spectral Subtraction noise reduction (>29 dB SNR gain) with time-smoothed masking to prevent musical noise.
- **Visual Analytics**: Simultaneous 30–40 FPS oscilloscope waveform, dual-trace before-and-after frequency spectrum overlay, live scrolling waterfall spectrogram, and autocorrelation-based pitch tuner ($F_0$ extraction, musical note mapping, and needle gauge).
- **Voice Activity Detection & Telecom DTX**: Real-time energy and zero-crossing rate (ZCR) speech classifier with hangover buffering, demonstrating Discontinuous Transmission (DTX) with ~48% bandwidth savings.
- **Secure Encrypted UDP Streaming**: An authenticated **AES-256-GCM** network sender and receiver with sequence-number AAD binding, elastic jitter buffer, live packet telemetry, and interactive cryptographic tamper attack rejection.

---

## Architecture & Processing Pipelines

### 1. Equalizer Studio Real-Time DSP Pipeline

```
[Audio Input] (Microphone / Demo Music / Signal Generator / WAV File)
      ↓
[Test Injections] (Optional: 50 Hz Hum & Harmonics / Gaussian White Noise)
      ↓
[Auto Hum Removal] (Automatic 50 Hz, 100 Hz, 150 Hz Cascaded IIR Notch Filters)
      ↓
[Spectral Subtraction] (Calibrated Background Noise Profile Subtraction & Gain Mask)
      ↓
[Standalone Filter] (Selectable 2nd/4th Order Butterworth: LP, HP, BP, or Notch)
      ↓
[Multi-Band Equalizer] (8-Band ISO Biquad Filter Cascade with Persistent SOS State)
      ↓
[Master Volume & Soft Limiter] (tanh Non-linear Saturation Limiter)
      ↓
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                                                                             │
▼                                                                             ▼
[Real-Time Speaker Playback]                                  [Dual-Trace Visualizer Queue]
(sounddevice Stream Callback)                                 ├── Oscilloscope (Time Domain)
                                                              ├── Dual Spectrum (Pre vs Post FFT)
                                                              ├── Scrolling Spectrogram (Waterfall)
                                                              ├── Pitch Tuner Gauge (Autocorrelation)
                                                              └── Measured Results (SNR & Latency)
```

### 2. Secure Encrypted Audio Streaming Pipeline (AES-256-GCM over UDP)

```
[SENDER NODE]
Audio Source (Mic / Demo / Gen / WAV)
      ↓
Real-Time DSP Chain (Windowing, 8-Band Equalizer, Digital Filtering, Soft Limiter)
      ↓
16-Bit PCM Quantization (2048 Samples / Block)
      ↓
AES-256-GCM Encryption (256-bit Key, 96-bit Unique Nonce, 128-bit Tag, Seq No. AAD Binding)
      ↓
UDP Network Packet Transmission (Dedicated Worker Thread)
      ↓
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ [Network / LAN / Internet] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ↓
[RECEIVER NODE]
UDP Network Receiver (Dedicated Worker Thread)
      ↓
AES-256-GCM Authenticated Decryption & Tamper Verification
  ├── [If Forged / Bit-Flipped] ──> DROP PACKET (Increment Tampered Counter)
  └── [If Valid & Verified]     ──> Extract Sequence & Forward PCM to Buffer
      ↓
Elastic Jitter Buffer (Pre-buffering, Delay Smoothing & Starvation Concealment)
      ↓
16-Bit PCM to Float32 De-quantization
      ↓
Real-Time Speaker Playback (`sounddevice` output callback)
      ↓
Live Oscilloscope Waveform, FFT Spectrum Display & Telemetry Dashboard
```

---

## Key Features & DSP Concepts

| Feature / Concept | Mathematical & Algorithmic Formulation | Implementation Details |
|---|---|---|
| **Sampling & Real-Time I/O** | $f_s = 44.1\text{ kHz}$, block size $N = 2048$ samples ($\tau \approx 46.4\text{ ms}$) | Low-latency duplex or output-only stream via `sounddevice` with ring-buffered queues. |
| **Windowing & Sidelobe Leakage** | Hann: $w[n] = 0.5 - 0.5\cos\left(\frac{2\pi n}{N-1}\right)$, Hamming, Rectangular | Suppresses spectral sidelobe leakage by **>57 dB** compared to rectangular windowing. |
| **FFT Spectrum Analysis** | Single-sided $N$-point Discrete Fourier Transform: $X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j2\pi kn/N}$ | Calibrated dB magnitude scale, 30–40 FPS UI refresh, parabolic sub-bin peak interpolation (<0.2% error). |
| **8-Band Graphic Equalizer** | Cascaded Biquad Second-Order Sections (SOS): $H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}$ | Standard ISO center frequencies: **63, 160, 400, 1k, 2.5k, 6k, 12k, 16k Hz** ($\pm 12\text{ dB}$). Zero-transient boundary state preservation (`zi`). |
| **Digital Filters** | 2nd & 4th-order Butterworth Low-Pass, High-Pass, Band-Pass, and IIR Notch | Max flat passband response, >40 dB stopband attenuation, interactive cutoff frequency and resonance ($Q$). |
| **Auto 50 Hz Hum Removal** | Multi-harmonic notch filters centered at $50\text{ Hz}$, $100\text{ Hz}$, and $150\text{ Hz}$ | Dynamic spectral peak detection for 50 Hz Indian/European mains hum. Achieves **>24 dB** hum suppression with **0.00 dB** passband loss. |
| **Spectral Subtraction Denoising** | $G(f) = \max\left(\frac{Y(f) - \alpha N(f)}{Y(f)}, \beta\right)$ | 1-second noise calibration profiling, over-subtraction factor $\alpha$, spectral floor $\beta$, time-smoothed gain masking to eliminate musical noise, and boundary crossfades. |
| **Pitch Detection & Tuner** | Normalized Autocorrelation Function (NACF): $R_{xx}(\tau) = \sum x[n] x[n+\tau]$ | Robust $F_0$ fundamental frequency extraction (50–2000 Hz), musical note mapping (A0 to C8), cents deviation, and live needle gauge. |
| **Voice Activity Detection (VAD)** | Short-time Energy $E = 10\log_{10}\left(\frac{1}{N}\sum x[n]^2\right)$ and Zero-Crossing Rate (ZCR) | Multi-feature speech classification with a 5-frame hangover buffer to prevent syllable clipping. Telecom DTX saves **~48% bandwidth**. |
| **Measured Results & Latency** | Algorithmic buffer delay + CPU DSP compute time + Audio hardware I/O driver latency | Real-time calculation of measured SNR gain (dB) and millisecond latency breakdown ($53.29\text{ ms}$ total round-trip). |
| **Secure AEAD Encryption** | **AES-256-GCM** (Galois/Counter Mode): 256-bit symmetric key, 96-bit unique nonces, 128-bit authentication tag | Cryptographic integrity: Sequence numbers authenticated as Additional Authenticated Data (AAD) to prevent replay/reordering attacks. |
| **Elastic Jitter Buffer** | Thread-safe FIFO queue with adaptive pre-buffering and starvation concealment | Absorbs network packet jitter, smooths audio playback, and tracks lost/tampered packets in real time. |
| **Tamper Attack Rejection** | Real-time cryptographic MAC verification of encrypted UDP audio datagrams | Drops tampered packets instantly without audio glitching or system crashes, updating live security telemetry counters. |

---

## Project Structure

```
DSPEqualiser/
├── requirements.txt            # Package dependencies (numpy, scipy, sounddevice, cryptography, etc.)
├── crypto_utils.py             # Shared AES-256-GCM engine & key management
├── generate_key.py             # Generates 256-bit shared_secret.key
├── main.py                     # Unified launcher (Equalizer Studio, Sender, Receiver)
├── sender.py                   # Encrypted Audio Sender (Live DSP + AES-GCM + UDP + GUI)
├── receiver.py                 # Encrypted Audio Receiver (UDP + Decrypt + Jitter Buffer + Playback + GUI)
├── verify_dsp.py               # Complete mathematical validation suite for all 9 DSP objectives
├── shared_secret.key           # Generated 256-bit AES symmetric key
├── spequaliser/
│   ├── __init__.py             # Package exports and version metadata (v1.1.0)
│   ├── dsp.py                  # Core DSP: Filters, SOS continuity, Auto Hum, Spectral Subtraction, VAD, Pitch
│   ├── audio_engine.py         # Real-time audio stream worker (sounddevice, routing, test injections)
│   ├── crypto.py               # Cryptographic package interface
│   ├── network.py              # UDPAudioSender, UDPAudioReceiver, JitterBuffer
│   ├── generators.py           # Signal synthesizers (Sine, Chirp sweep, Pink/White noise, Chord) & WAV player
│   └── gui.py                  # Dark-themed GUI with dual live visualization & interactive studio cards
└── tests/
    ├── test_dsp.py             # DSP unit tests (Filters, VAD, Hum Removal, Denoising, Pitch, Latency)
    ├── test_crypto.py          # AES-256-GCM and tamper detection unit tests
    └── test_network.py         # UDP network transmission and jitter buffer tests
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.10** or higher
- Working audio input/output device (or built-in audio fallback)

### 2. Clone Repository & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Agent66/DSPEqualiser.git
cd DSPEqualiser

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## Operating Guide

### Mode 1: Real-Time Equalizer & DSP Studio GUI

Launch the primary Equalizer Application:
```bash
python main.py
```
*(or explicitly: `python main.py --mode gui`)*

#### Studio Controls & Demonstration Guide:
1. **Audio Sources**:
   - **Microphone**: Live microphone capture with input gain slider. *(Note: Speaker monitoring for microphone is disabled by default to eliminate acoustic feedback howl; toggle `Monitor Mic to Speakers` if using headphones).*
   - **Demo Music**: Built-in 4-voice polyphonic synthetic demo track for immediate testing without external files.
   - **Signal Generator**: Synthesize clean Sine tones (10–20,000 Hz), continuous Chirp frequency sweeps (20 Hz–20 kHz), White noise, Pink noise (Paul Kellet 3 dB/oct filter), and multi-tone Chords.
   - **WAV Player**: Load any standard WAV file with pause, seek, and loop functionality.
2. **Auto Hum Removal Studio (Card 1)**:
   - Toggle **Auto Hum Removal** to activate real-time 50 Hz, 100 Hz, and 150 Hz notch filters.
   - Click **⚡ Inject 50Hz Hum** to inject real-world mains hum into the audio stream. Observe the red trace in the spectrum analyzer spike at 50 Hz, and the cyan trace notch down by **>24 dB** with the green "50Hz Hum Detected & Suppressed" badge active!
3. **Spectral Subtraction Denoiser Studio (Card 2)**:
   - Click **⚡ Inject Noise** to add background Gaussian noise.
   - Click **🎙️ Record Noise (1s)** to calibrate the ambient noise profile. The calibration progress bar will fill, after which background noise is actively subtracted, displaying the real-time measured noise floor reduction in dB.
4. **8-Band Graphic Equalizer**:
   - Adjust individual band sliders (63 Hz to 16 kHz) with $\pm 12\text{ dB}$ range.
   - Select from 10+ professional presets: *Flat, Bass Boost, Treble Boost, Vocal Clarify, Acoustic, Rock, Electronic, Speech Enhance, Hum Reject, Noise Suppress*.
5. **Pitch Tuner & Live Spectrogram**:
   - Speak or sing into the microphone or play a sine wave to observe the real-time Pitch Tuner needle gauge display the fundamental frequency $F_0$, musical note, and cents deviation.
   - Watch the scrolling waterfall Spectrogram display continuous time-frequency evolution.
6. **Measured Results & Academic Benchmark (Card 4)**:
   - Live telemetry monitors **SNR Improvement (dB)** and **End-to-End Latency Breakdown (ms)**.
   - Click **📊 Run Academic Benchmark** to generate a full formal DSP evaluation report directly in a popup dialog.

---

### Mode 2: Secure Encrypted UDP Sender & Receiver

The suite includes an encrypted streaming pair that preserves the complete DSP pipeline while securing transmission over UDP using **AES-256-GCM**.

#### Step 1: Generate the Shared Secret Key
```bash
python generate_key.py
```
This generates `shared_secret.key` (256-bit cryptographically secure key).
- *Single computer testing*: The file is read locally by both sender and receiver.
- *Two computers on a LAN*: Copy `shared_secret.key` to the receiver machine.

#### Step 2: Start the Receiver (Start this first)
```bash
python receiver.py --port 5005
```
*Key Options:*
- `--port 5005`: UDP listening port.
- `--bind 0.0.0.0`: Network interface bind address.
- `--key shared_secret.key`: Path to symmetric key file.
- `--no-gui`: Headless terminal mode.

#### Step 3: Start the Sender
```bash
# Testing on the same computer:
python sender.py --host 127.0.0.1 --port 5005

# Testing across separate machines on a network:
python sender.py --host <RECEIVER_IP_ADDRESS> --port 5005
```

#### Step 4: Demonstrate Cryptographic Tamper Rejection
1. Audio captured on the sender is filtered through the 8-band equalizer and digital filters, encrypted with AES-256-GCM, and transmitted over UDP.
2. The receiver decrypts packets, verifies the 128-bit authentication tag and sequence number AAD, buffers samples in an elastic jitter buffer, and plays clean audio through the speakers.
3. On the receiver GUI, click **⚠️ Simulate Tamper Attack**:
   - The receiver deliberately alters bits in incoming packets.
   - AES-GCM authentication immediately fails.
   - Corrupted packets are dropped instantly; the **Tampered** packet counter increments; audio playback remains completely stable with zero glitching or crashes!

---

## Verified Academic Benchmarks & Evaluation Results

All algorithms have been mathematically validated using the automated verification suite (`verify_dsp.py`) and unit test suite (`tests/`).

### DSP Verification Scorecard (`verify_dsp.py`)

| Test Objective | DSP Requirement | Measured Result | Verdict |
|---|---|---|---|
| **Test 1: FFT Accuracy** | Sub-bin frequency error < 0.5% | Max error: **0.20 Hz (0.17%)** across 120–7500 Hz | **PASS** |
| **Test 2: Windowing Leakage** | Hann sidelobe suppression > 20 dB vs rectangular | Stopband floor: **-120.0 dB** (**57.6 dB improvement**) | **PASS** |
| **Test 3: Low-Pass Filter** | 1000 Hz cutoff: Flat passband, >40 dB stopband | Passband: **-0.0 dB** \| Stopband (4 kHz): **-49.1 dB** | **PASS** |
| **Test 3: High-Pass Filter** | 2000 Hz cutoff: Flat passband, >40 dB stopband | Stopband (300 Hz): **-66.1 dB** \| Passband: **-0.0 dB** | **PASS** |
| **Test 3: Band-Pass Filter** | 1000 Hz center: Flat peak, symmetric rolloff | Center: **0.0 dB** \| 250 Hz: **-33.8 dB** \| 4 kHz: **-35.5 dB** | **PASS** |
| **Test 3: Notch Filter** | 60 Hz mains notch: >40 dB notch depth | Notch rejection: **-82.1 dB** \| Passband: **-0.0 dB** | **PASS** |
| **Test 4: Equalizer Gains** | $\pm 6\text{ dB}$ gain accuracy with zero boundary clicks | Measured: **+6.00 dB / -6.00 dB** \| Boundary error: **0.00e+00** | **PASS** |
| **Test 5: Pitch Detection** | Sub-Hz fundamental frequency accuracy ($F_0$) | 7 musical notes (A2–A5): **< 4.4 cents deviation** | **PASS** |
| **Test 6: Spectral Subtraction** | High-frequency noise suppression > 10 dB | Measured noise floor drop: **34.0 dB reduction** | **PASS** |
| **Test 7: Voice Activity (VAD)** | Energy + ZCR classification + 5-frame hangover | Speech correctly identified; DTX saves **48.0% bandwidth** | **PASS** |
| **Test 8: Auto Hum Removal** | 50 Hz & harmonics notch depth > 20 dB | 50 Hz: **24.3 dB**, 100 Hz: **24.4 dB**, 150 Hz: **32.7 dB** (Passband loss: **0.00 dB**) | **PASS** |
| **Test 9: System Latency** | Real-time processing budget < 60 ms | Buffer: **46.44 ms**, DSP compute: **0.85 ms**, Total: **53.29 ms** | **PASS** |

### Execution Latency Breakdown
$$\text{Total System Latency} = \tau_{\text{buffer}} + \tau_{\text{DSP}} + \tau_{\text{driver}} = 46.44\text{ ms} + 0.85\text{ ms} + 6.00\text{ ms} = 53.29\text{ ms}$$
- **Algorithmic Buffer Latency**: $\frac{N}{f_s} = \frac{2048}{44100} \approx 46.44\text{ ms}$
- **CPU DSP Compute Execution Time**: $\approx 0.85\text{ ms}$ (<2% of audio frame budget)
- **Audio Hardware I/O Latency**: $\approx 6.00\text{ ms}$

---

## Verification & Testing Commands

### 1. Run Complete DSP Mathematical Verification Suite
```bash
python verify_dsp.py
```
*Executes all 9 mathematical DSP tests, verifying FFT interpolation, filter frequency responses, state continuity, pitch tracking, spectral subtraction, VAD, and auto hum notch depths.*

### 2. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*Executes 24 automated unit tests covering:*
- `test_dsp.py`: Windowing, FFT accuracy, biquad filters, state continuity, pitch tracking, VAD, telecom hangover, auto hum removal, spectral subtraction, and latency benchmarks.
- `test_crypto.py`: AES-256-GCM encryption/decryption, sequence number AAD validation, nonce tampering, ciphertext tampering, invalid keys, and short packets.
- `test_network.py`: Jitter buffer pre-buffering, overflow management, starvation concealment, and UDP sender/receiver loopback transmission.

---

## Project Report Guide

When presenting or writing your report on this implementation, emphasize the following points:

### 1. Preservation of Continuous DSP Mathematical State
- Standard digital filtering often fails when processing audio in discrete chunks due to transient boundary clicks caused by filter state resets.
- This implementation stores persistent filter state vectors (`zi` second-order section delays) across successive block calls in both `MultiBandEqualizer` and `StandaloneFilter`, achieving mathematical continuity with **$0.00\times 10^0$ boundary error**.

### 2. Industry-Standard AEAD Security: AES-256-GCM
- **Galois/Counter Mode (GCM)** provides Authenticated Encryption with Associated Data (AEAD), the exact cipher suite mandated by modern secure communications protocols including **SRTP** (WebRTC/VoIP) and **TLS 1.3**.
- **Tamper Resistance**: If any bit of the ciphertext or header is modified in transit, tag validation fails and the packet is immediately dropped before touching the audio pipeline.
- **Replay Protection**: The packet sequence number is bound as Authenticated Additional Data (AAD), preventing adversaries from altering transmission order or re-injecting stale audio packets.

### 3. Voice Activity Detection & Telecom Bandwidth Optimization
- Combines short-time frame energy with Zero-Crossing Rate (ZCR) to differentiate voiced vowels (high energy, low ZCR) from unvoiced fricatives (moderate energy, high ZCR) and ambient background noise.
- Features a 5-frame hangover window that prevents clipping the natural decays of words, saving **~48% of transmission bandwidth** during conversational pauses (Discontinuous Transmission / DTX).

### 4. Engineering Trade-offs & Future Scope
- **Dynamic Key Negotiation**: In this educational implementation, keys are pre-shared via `shared_secret.key`. A commercial VoIP implementation would negotiate session keys dynamically using Ephemeral Elliptic Curve Diffie-Hellman (**ECDHE**) via DTLS.
- **Audio Codec Compression**: Audio is streamed as uncompressed 16-bit linear PCM at 44.1 kHz (~705 kbps). Integrating an adaptive psychoacoustic speech codec such as **Opus** prior to encryption would reduce network bandwidth to 24–64 kbps while preserving voice fidelity.
- **Acoustic Echo Cancellation (AEC)**: For full-duplex speakerphone operation, integrating adaptive normalized least mean squares (NLMS) AEC would eliminate acoustic echo during speaker playback.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
