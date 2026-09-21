"""
Unit tests for SpEqualiser Cryptographic Subsystem (AES-256-GCM).
"""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from crypto_utils import (
    generate_key,
    save_key,
    load_key,
    AudioCrypto,
    KEY_SIZE,
    HEADER_SIZE,
    TAG_SIZE
)


class TestCrypto(unittest.TestCase):

    def setUp(self):
        self.test_key = generate_key()
        self.crypto = AudioCrypto(self.test_key)

    def test_key_generation_and_io(self):
        # 1. Key must be exactly 32 bytes (256 bits)
        self.assertEqual(len(self.test_key), KEY_SIZE)

        # 2. Key file saving and loading
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "test_shared.key"
            save_key(self.test_key, key_path)
            self.assertTrue(key_path.is_file())

            loaded = load_key(key_path)
            self.assertEqual(loaded, self.test_key)

    def test_encrypt_decrypt_roundtrip(self):
        # Generate 2048 16-bit PCM audio samples
        samples = (np.sin(np.linspace(0, 2 * np.pi * 10, 2048)) * 32000).astype(np.int16)
        pcm_bytes = samples.tobytes()
        seq_num = 42

        # Encrypt
        packet = self.crypto.encrypt_packet(pcm_bytes, seq_num)
        self.assertTrue(len(packet) > len(pcm_bytes))

        # Decrypt
        decrypted_pcm, decrypted_seq, err = self.crypto.decrypt_packet(packet)
        self.assertIsNone(err)
        self.assertEqual(decrypted_seq, seq_num)
        self.assertEqual(decrypted_pcm, pcm_bytes)

    def test_tamper_detection_ciphertext(self):
        """Modifying 1 bit in ciphertext MUST fail AES-GCM tag verification."""
        pcm_bytes = b"Sample 16-bit PCM Audio Payload " * 16
        packet = bytearray(self.crypto.encrypt_packet(pcm_bytes, seq_num=100))

        # Tamper with the last byte
        packet[-1] ^= 0x01

        decrypted_pcm, decrypted_seq, err = self.crypto.decrypt_packet(bytes(packet))
        self.assertIsNone(decrypted_pcm)
        self.assertEqual(err, "AUTHENTICATION_FAILED_TAMPERED")

    def test_tamper_detection_sequence_number_aad(self):
        """Tampering with sequence number in the header MUST fail AAD authentication."""
        pcm_bytes = b"Secure audio test chunk" * 8
        packet = bytearray(self.crypto.encrypt_packet(pcm_bytes, seq_num=10))

        # Sequence number is located at bytes [5..8]
        packet[8] ^= 0xFF

        decrypted_pcm, decrypted_seq, err = self.crypto.decrypt_packet(bytes(packet))
        self.assertIsNone(decrypted_pcm)
        self.assertEqual(err, "AUTHENTICATION_FAILED_TAMPERED")

    def test_tamper_detection_nonce(self):
        """Tampering with nonce in header MUST fail authentication."""
        pcm_bytes = b"Secure audio test chunk" * 8
        packet = bytearray(self.crypto.encrypt_packet(pcm_bytes, seq_num=5))

        # Nonce is at bytes [9..20]
        packet[10] ^= 0x55

        decrypted_pcm, decrypted_seq, err = self.crypto.decrypt_packet(bytes(packet))
        self.assertIsNone(decrypted_pcm)
        self.assertEqual(err, "AUTHENTICATION_FAILED_TAMPERED")

    def test_wrong_key_decryption(self):
        """Decrypting with a different key must fail."""
        other_crypto = AudioCrypto(generate_key())
        pcm_bytes = b"Private confidential speech" * 10
        packet = self.crypto.encrypt_packet(pcm_bytes, seq_num=1)

        decrypted_pcm, decrypted_seq, err = other_crypto.decrypt_packet(packet)
        self.assertIsNone(decrypted_pcm)
        self.assertEqual(err, "AUTHENTICATION_FAILED_TAMPERED")

    def test_short_packet_handling(self):
        """Packets smaller than header + tag must be rejected gracefully."""
        decrypted_pcm, seq, err = self.crypto.decrypt_packet(b"TooShort")
        self.assertIsNone(decrypted_pcm)
        self.assertTrue("PACKET_TOO_SHORT" in err)

    def test_invalid_magic_handling(self):
        """Packets with wrong magic header must be rejected."""
        pcm_bytes = b"Audio data block" * 4
        packet = bytearray(self.crypto.encrypt_packet(pcm_bytes, seq_num=1))
        # Overwrite magic 'SPEQ' with 'XXXX'
        packet[:4] = b"XXXX"

        decrypted_pcm, seq, err = self.crypto.decrypt_packet(bytes(packet))
        self.assertIsNone(decrypted_pcm)
        self.assertTrue("INVALID_MAGIC" in err)


if __name__ == "__main__":
    unittest.main()
