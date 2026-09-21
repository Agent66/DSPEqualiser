"""
Package-level cryptographic utilities for spequaliser.
Re-exports AudioCrypto, generate_key, load_key, save_key.
"""

from crypto_utils import (
    KEY_SIZE,
    NONCE_SIZE,
    TAG_SIZE,
    HEADER_MAGIC,
    PROTOCOL_VERSION,
    HEADER_FORMAT,
    HEADER_SIZE,
    generate_key,
    save_key,
    load_key,
    generate_dh_keypair,
    derive_session_key,
    AudioCrypto
)

__all__ = [
    "KEY_SIZE",
    "NONCE_SIZE",
    "TAG_SIZE",
    "HEADER_MAGIC",
    "PROTOCOL_VERSION",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "generate_key",
    "save_key",
    "load_key",
    "generate_dh_keypair",
    "derive_session_key",
    "AudioCrypto"
]

__all__ = [
    "KEY_SIZE",
    "NONCE_SIZE",
    "TAG_SIZE",
    "HEADER_MAGIC",
    "PROTOCOL_VERSION",
    "HEADER_FORMAT",
    "HEADER_SIZE",
    "generate_key",
    "save_key",
    "load_key",
    "AudioCrypto"
]
