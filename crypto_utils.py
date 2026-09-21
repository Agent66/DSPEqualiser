"""
Shared Cryptographic Utilities for Real-Time Encrypted Audio Streaming.

Uses AES-256-GCM (Galois/Counter Mode) authenticated encryption:
- 256-bit symmetric encryption key
- 96-bit (12-byte) cryptographically random nonce per packet
- 128-bit (16-byte) authentication tag for cryptographic integrity and tamper detection
- Authenticated Additional Data (AAD) binding packet sequence number and protocol header
"""

import os
import struct
import secrets
import time
from typing import Tuple, Optional, Union
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    AESGCM = None
    InvalidTag = Exception
    X25519PrivateKey = None
    X25519PublicKey = None
    HKDF = None
    hashes = None


KEY_SIZE = 32          # 256 bits
NONCE_SIZE = 12        # 96 bits (standard for AES-GCM)
TAG_SIZE = 16          # 128 bits authentication tag
HEADER_MAGIC = b"SPEQ" # 4-byte protocol identifier
PROTOCOL_VERSION = 1   # 1-byte protocol version

# Packet header layout:
# [Magic: 4 bytes] [Version: 1 byte] [Sequence: 4 bytes uint32] [Nonce: 12 bytes]
# Total Header = 21 bytes
HEADER_FORMAT = ">4sBI12s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 21 bytes
AAD_SIZE = 4 + 1 + 4                          # 9 bytes (Magic + Version + Seq)
TIMESTAMP_SIZE = 8                            # uint64 nanoseconds prepended inside ciphertext
TIMESTAMP_SKEW_NS = 10 * 60 * 10**9           # 10 minutes — used to detect legacy packets
DH_PUBLIC_KEY_SIZE = 32                       # X25519 raw public key
HKDF_SALT = b"SpEqualiser-X25519-v1"
HKDF_INFO = b"aes-256-gcm-session"


def generate_key() -> bytes:
    """
    Generates a cryptographically secure 256-bit (32-byte) random key.
    """
    return secrets.token_bytes(KEY_SIZE)


def save_key(key: bytes, file_path: Union[str, Path] = "shared_secret.key") -> str:
    """
    Saves a 32-byte key to a binary file.
    """
    if len(key) != KEY_SIZE:
        raise ValueError(f"Key must be exactly {KEY_SIZE} bytes (got {len(key)})")
    
    path = Path(file_path)
    path.write_bytes(key)
    return str(path.resolve())


def load_key(file_path: Union[str, Path] = "shared_secret.key") -> bytes:
    """
    Loads and validates a 32-byte secret key from a file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Key file not found: '{file_path}'. "
            f"Run 'python generate_key.py' first to generate a shared secret key."
        )
    
    key_bytes = path.read_bytes()
    
    # Support hex-encoded or base64 strings if saved as text
    if len(key_bytes) == KEY_SIZE * 2:
        try:
            key_bytes = bytes.fromhex(key_bytes.decode("ascii").strip())
        except Exception:
            pass
    elif len(key_bytes) == 44 and (key_bytes.endswith(b"=\n") or key_bytes.endswith(b"=")):
        import base64
        try:
            key_bytes = base64.b64decode(key_bytes)
        except Exception:
            pass

    if len(key_bytes) != KEY_SIZE:
        raise ValueError(
            f"Invalid key length in '{file_path}': expected {KEY_SIZE} bytes, got {len(key_bytes)} bytes."
        )
    
    return key_bytes


def generate_dh_keypair() -> Tuple[object, bytes]:
    """
    Generates an X25519 (Elliptic-Curve Diffie-Hellman) key pair.

    Returns:
        (private_key, public_bytes) where public_bytes is 32 raw bytes.
    """
    if X25519PrivateKey is None:
        raise RuntimeError("The 'cryptography' package is required for Diffie-Hellman.")
    private_key = X25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private_key, public_bytes


def derive_session_key(private_key, peer_public_bytes: bytes) -> bytes:
    """
    Completes X25519 ECDH and derives a 256-bit AES-GCM session key via HKDF-SHA256.
    Both peers obtain the same key from the same two public keys.
    """
    if X25519PublicKey is None or HKDF is None:
        raise RuntimeError("The 'cryptography' package is required for Diffie-Hellman.")
    if len(peer_public_bytes) != DH_PUBLIC_KEY_SIZE:
        raise ValueError(f"Peer public key must be {DH_PUBLIC_KEY_SIZE} bytes.")
    peer_public = X25519PublicKey.from_public_bytes(peer_public_bytes)
    shared_secret = private_key.exchange(peer_public)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=HKDF_SALT,
        info=HKDF_INFO,
    ).derive(shared_secret)


class AudioCrypto:
    """
    High-performance AES-256-GCM encryption and tamper-detection engine.
    """
    def __init__(self, key: Union[bytes, str, Path]):
        if AESGCM is None:
            raise RuntimeError(
                "The 'cryptography' package is required. Install with: pip install cryptography>=42.0.0"
            )

        if isinstance(key, (str, Path)):
            if os.path.isfile(key):
                self.key = load_key(key)
            elif isinstance(key, str) and len(key) == KEY_SIZE * 2:
                self.key = bytes.fromhex(key)
            else:
                raise ValueError("Key parameter must be 32 bytes or a valid path to key file.")
        elif isinstance(key, bytes):
            if len(key) != KEY_SIZE:
                raise ValueError(f"Key must be exactly {KEY_SIZE} bytes (got {len(key)}).")
            self.key = key
        else:
            raise TypeError("Key must be bytes or a file path string.")

        self.cipher = AESGCM(self.key)
        self.last_latency_ms: Optional[float] = None

    def set_key(self, key: bytes) -> None:
        """Hot-swap the AES-256 session key after Diffie-Hellman completes."""
        if len(key) != KEY_SIZE:
            raise ValueError(f"Key must be exactly {KEY_SIZE} bytes (got {len(key)}).")
        self.key = key
        self.cipher = AESGCM(self.key)

    def encrypt_packet(self, pcm_bytes: bytes, seq_num: int, timestamp_ns: Optional[int] = None) -> bytes:
        """
        Encrypts raw audio PCM bytes using AES-256-GCM.

        Structure of generated datagram:
        [0..3]   : Magic bytes b"SPEQ"
        [4]      : Protocol version (1)
        [5..8]   : Sequence number (uint32 big-endian)
        [9..20]  : Nonce (12 bytes unique random)
        [21..end]: Ciphertext + 16-byte GCM authentication tag

        The header prefix [0..8] is passed as Authenticated Additional Data (AAD)
        to cryptographically bind sequence number to the payload.
        """
        seq_uint32 = seq_num & 0xFFFFFFFF
        nonce = os.urandom(NONCE_SIZE)
        aad = struct.pack(">4sBI", HEADER_MAGIC, PROTOCOL_VERSION, seq_uint32)
        ts = int(time.time_ns() if timestamp_ns is None else timestamp_ns)
        payload = struct.pack(">Q", ts & 0xFFFFFFFFFFFFFFFF) + pcm_bytes

        # AES-GCM encrypt produces ciphertext + 16-byte auth tag appended
        ciphertext = self.cipher.encrypt(nonce, payload, aad)
        
        # Assemble final packet
        return aad + nonce + ciphertext

    def decrypt_packet(self, packet_bytes: bytes) -> Tuple[Optional[bytes], Optional[int], Optional[str]]:
        """
        Authenticates and decrypts an incoming UDP datagram.

        Returns:
            (pcm_data, seq_num, error_message)
            - If authentication & decryption succeeds: (bytes, seq_num, None)
            - If corrupted or tampered: (None, seq_num or None, error_string)

        A 64-bit send timestamp is stripped from the plaintext when present.
        The last observed one-way latency (ms) is stored on ``last_latency_ms``.
        """
        min_size = HEADER_SIZE + TAG_SIZE
        if len(packet_bytes) < min_size:
            return None, None, f"PACKET_TOO_SHORT ({len(packet_bytes)} < {min_size} bytes)"

        # Unpack header
        magic, version, seq_num, nonce = struct.unpack(HEADER_FORMAT, packet_bytes[:HEADER_SIZE])
        
        if magic != HEADER_MAGIC:
            return None, None, f"INVALID_MAGIC ({magic!r})"
            
        if version != PROTOCOL_VERSION:
            return None, None, f"UNSUPPORTED_VERSION ({version})"

        aad = packet_bytes[:AAD_SIZE]
        ciphertext = packet_bytes[HEADER_SIZE:]

        try:
            plain = self.cipher.decrypt(nonce, ciphertext, aad)
            pcm_bytes, _ts_ns, latency_ms = _split_timestamped_payload(plain)
            self.last_latency_ms = latency_ms
            return pcm_bytes, seq_num, None
        except InvalidTag:
            # Cryptographic authentication failed: packet was altered or forged!
            return None, seq_num, "AUTHENTICATION_FAILED_TAMPERED"
        except Exception as e:
            return None, seq_num, f"DECRYPTION_ERROR: {e}"


def _split_timestamped_payload(plain: bytes) -> Tuple[bytes, Optional[int], Optional[float]]:
    """
    Splits an optional 8-byte nanosecond timestamp from the front of plaintext.
    Legacy packets (no timestamp) are returned unchanged.
    """
    if len(plain) < TIMESTAMP_SIZE:
        return plain, None, None
    ts_ns = struct.unpack(">Q", plain[:TIMESTAMP_SIZE])[0]
    now_ns = time.time_ns()
    if abs(now_ns - ts_ns) > TIMESTAMP_SKEW_NS:
        return plain, None, None
    latency_ms = max(0.0, (now_ns - ts_ns) / 1_000_000.0)
    return plain[TIMESTAMP_SIZE:], ts_ns, latency_ms
