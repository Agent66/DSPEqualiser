"""
Real-Time Audio Equalizer & Spectrum Analyzer
A Digital Signal Processing & Secure Audio Streaming Package
"""

__version__ = "1.1.0"

from .crypto import AudioCrypto, generate_key, load_key, save_key
from .network import UDPAudioSender, UDPAudioReceiver, JitterBuffer, UDPDuplexLink
