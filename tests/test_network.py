"""
Unit tests for SpEqualiser Network Transport & Jitter Buffer Subsystem.
"""

import time
import unittest
import numpy as np

from crypto_utils import generate_key, AudioCrypto
from spequaliser.network import JitterBuffer, UDPAudioSender, UDPAudioReceiver


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.key = generate_key()
        self.crypto = AudioCrypto(self.key)

    def test_jitter_buffer_prebuffering_and_pop(self):
        block_size = 512
        jb = JitterBuffer(block_size=block_size, target_prebuffer_blocks=2, max_blocks=5)

        # Initially empty: should return zeros
        silence = jb.pop(block_size)
        self.assertEqual(len(silence), block_size)
        self.assertTrue(np.all(silence == 0.0))

        # Push block 1 (below prebuffer target of 2 blocks)
        data1 = (np.ones(block_size, dtype=np.int16) * 16384).tobytes()
        jb.push(1, data1)
        self.assertTrue(jb.is_prebuffering)

        # Still prebuffering: should return silence
        silence2 = jb.pop(block_size)
        self.assertTrue(np.all(silence2 == 0.0))

        # Push block 2 (reaches prebuffer target)
        data2 = (np.ones(block_size, dtype=np.int16) * 32767).tobytes()
        jb.push(2, data2)
        self.assertFalse(jb.is_prebuffering)

        # Now popping should yield block 1 (value approx 0.5)
        out1 = jb.pop(block_size)
        self.assertAlmostEqual(out1[0], 0.5, delta=0.01)

        # Next pop should yield block 2 (value approx 1.0)
        out2 = jb.pop(block_size)
        self.assertAlmostEqual(out2[0], 1.0, delta=0.01)

        # Now buffer is starved: should return silence and re-enter prebuffering
        starved = jb.pop(block_size)
        self.assertTrue(np.all(starved == 0.0))
        self.assertTrue(jb.is_prebuffering)
        self.assertTrue(jb.underruns > 0)

    def test_jitter_buffer_overflow(self):
        block_size = 256
        max_blocks = 3
        jb = JitterBuffer(block_size=block_size, target_prebuffer_blocks=1, max_blocks=max_blocks)

        # Push 5 blocks into max 3 capacity
        for i in range(5):
            d = (np.ones(block_size, dtype=np.int16) * i).tobytes()
            jb.push(i, d)

        # Depth should never exceed max_blocks
        self.assertLessEqual(jb.get_depth_blocks(), max_blocks)

    def test_udp_sender_receiver_loopback(self):
        """Tests live UDP loopback transmission between sender and receiver."""
        port = 15005
        block_size = 1024
        jb = JitterBuffer(block_size=block_size, target_prebuffer_blocks=2)

        receiver = UDPAudioReceiver(port=port, crypto=self.crypto, jitter_buffer=jb, bind_ip="127.0.0.1")
        sender = UDPAudioSender(host="127.0.0.1", port=port, crypto=self.crypto)

        receiver.start()
        sender.start()

        try:
            # Send 5 test audio blocks
            for seq in range(5):
                sample_block = (np.sin(np.linspace(0, np.pi, block_size)) * 10000).astype(np.int16)
                sender.send_pcm_chunk(sample_block.tobytes())
                time.sleep(0.02)

            # Give network worker a brief moment to process
            time.sleep(0.2)

            stats = receiver.get_stats()
            self.assertGreaterEqual(stats["packets_received"], 3)
            self.assertGreaterEqual(stats["packets_ok"], 3)
            self.assertEqual(stats["packets_tampered"], 0)
            self.assertEqual(stats["packets_lost"], 0)
        finally:
            sender.stop()
            receiver.stop()


if __name__ == "__main__":
    unittest.main()
