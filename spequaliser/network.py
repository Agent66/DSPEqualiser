"""
Real-Time Network Transport & Jitter Buffer Subsystem for Encrypted Audio.

Implements:
- UDPAudioSender: Non-blocking UDP transport running on a dedicated worker thread
- JitterBuffer: Elastic, thread-safe real-time audio FIFO with pre-buffering & underrun protection
- UDPAudioReceiver: UDP listener with sequence tracking, tamper rejection, and telemetry
"""

import socket
import threading
import queue
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .crypto import AudioCrypto
from .handshake import (
    DiffieHellmanSession,
    STATE_ESTABLISHED,
    STATE_PSK,
    STATE_WAITING,
    is_handshake_packet,
)


class UDPAudioSender:
    """
    Dedicated UDP audio sender.
    Decouples network socket operations from real-time audio callback threads
    using a non-blocking queue.
    """
    def __init__(
        self,
        host: str,
        port: int,
        crypto: AudioCrypto,
        max_queue_size: int = 32
    ):
        self.host = host
        self.port = int(port)
        self.crypto = crypto
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        self.socket: Optional[socket.socket] = None
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Sequence & Telemetry
        self.seq_num = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.bitrate_kbps = 0.0
        self._last_time = time.time()
        self._bytes_window = 0
        self._lock = threading.Lock()

    def start(self):
        """Starts UDP socket and worker thread."""
        if self.is_running:
            return
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow fast rebinding
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass

        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._sender_loop,
            name="UDPAudioSenderThread",
            daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        """Stops worker thread and closes socket."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.worker_thread = None

        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def send_pcm_chunk(self, pcm_bytes: bytes):
        """
        Called by audio callback. Non-blocking queue insertion.
        If queue is full, drops the oldest frame to preserve live timing.
        """
        if not self.is_running:
            return
        
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass

    def _sender_loop(self):
        """Dedicated network I/O loop: encrypts and sends UDP packets."""
        while self.is_running:
            try:
                pcm_bytes = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with self._lock:
                    current_seq = self.seq_num
                    self.seq_num = (self.seq_num + 1) & 0xFFFFFFFF

                packet = self.crypto.encrypt_packet(pcm_bytes, current_seq)
                self.socket.sendto(packet, (self.host, self.port))

                with self._lock:
                    self.packets_sent += 1
                    self.bytes_sent += len(packet)
                    self._bytes_window += len(packet)
                    now = time.time()
                    elapsed = now - self._last_time
                    if elapsed >= 0.5:
                        self.bitrate_kbps = (self._bytes_window * 8.0) / (elapsed * 1000.0)
                        self._bytes_window = 0
                        self._last_time = now
            except Exception:
                # Silently catch network errors to keep worker loop resilient
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Returns snapshot of sender transmission telemetry."""
        with self._lock:
            return {
                "host": self.host,
                "port": self.port,
                "packets_sent": self.packets_sent,
                "bytes_sent": self.bytes_sent,
                "bitrate_kbps": round(self.bitrate_kbps, 1),
                "queue_size": self.queue.qsize()
            }


class JitterBuffer:
    """
    Elastic, thread-safe Jitter Buffer for real-time audio playback.
    Pre-buffers packets to absorb network delay variance (jitter)
    and conceals packet starvation/underruns.
    """
    def __init__(
        self,
        block_size: int = 2048,
        target_prebuffer_blocks: int = 3,
        max_blocks: int = 12
    ):
        self.block_size = block_size
        self.target_prebuffer_blocks = max(1, target_prebuffer_blocks)
        self.max_blocks = max_blocks
        
        self.buffer: queue.Queue = queue.Queue(maxsize=self.max_blocks)
        self._lock = threading.Lock()
        self.is_prebuffering = True
        self.underruns = 0
        self.total_frames_popped = 0

    def push(self, seq_num: int, pcm_bytes: bytes):
        """Pushes authenticated audio block into the jitter buffer."""
        with self._lock:
            # If buffer exceeds max capacity, drop oldest to avoid high latency drift
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.buffer.put_nowait((seq_num, pcm_bytes))
            except queue.Full:
                pass

            # Check if prebuffering threshold is satisfied
            if self.is_prebuffering and self.buffer.qsize() >= self.target_prebuffer_blocks:
                self.is_prebuffering = False

    def pop(self, num_samples: int) -> np.ndarray:
        """
        Called by the audio playback callback thread.
        Returns a float32 numpy array in range [-1.0, 1.0].
        If starved or prebuffering, returns zeros (silence) to prevent popping.
        """
        with self._lock:
            if self.is_prebuffering:
                return np.zeros(num_samples, dtype=np.float32)

            if self.buffer.empty():
                # Underrun occurred! Re-enter pre-buffering to allow buffer refill
                self.underruns += 1
                self.is_prebuffering = True
                return np.zeros(num_samples, dtype=np.float32)

            try:
                _, pcm_bytes = self.buffer.get_nowait()
                self.total_frames_popped += 1
            except queue.Empty:
                self.underruns += 1
                self.is_prebuffering = True
                return np.zeros(num_samples, dtype=np.float32)

        # Convert 16-bit PCM bytes to float32
        samples_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0

        # Adjust length if it deviates from requested block size
        if len(samples_float32) < num_samples:
            padded = np.zeros(num_samples, dtype=np.float32)
            padded[:len(samples_float32)] = samples_float32
            return padded
        elif len(samples_float32) > num_samples:
            return samples_float32[:num_samples]

        return samples_float32

    def reset(self):
        """Clears all buffered audio."""
        with self._lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break
            self.is_prebuffering = True

    def get_depth_blocks(self) -> int:
        """Returns current number of audio blocks queued in buffer."""
        return self.buffer.qsize()


class UDPAudioReceiver:
    """
    UDP Audio Receiver running on a dedicated network thread.
    Authenticates and decrypts incoming packets via AES-256-GCM,
    detects corrupted/tampered packets, tracks packet sequence/loss,
    and feeds valid audio blocks into the Jitter Buffer.
    """
    def __init__(
        self,
        port: int,
        crypto: AudioCrypto,
        jitter_buffer: JitterBuffer,
        bind_ip: str = "0.0.0.0"
    ):
        self.port = int(port)
        self.crypto = crypto
        self.jitter_buffer = jitter_buffer
        self.bind_ip = bind_ip

        self.socket: Optional[socket.socket] = None
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Live Telemetry Stats
        self.packets_received = 0
        self.packets_ok = 0
        self.packets_tampered = 0
        self.packets_lost = 0
        self.last_seq_num: Optional[int] = None
        self.bytes_received = 0
        self.bitrate_kbps = 0.0
        self.latency_ms = 0.0
        self._latencies = []
        self._bytes_window = 0
        self._last_time = time.time()
        self._lock = threading.Lock()

        # Demonstration feature: simulate intentional packet tampering
        self.simulate_tampering = False
        self.on_handshake = None  # optional callable(data, addr)

    def start(self):
        """Binds UDP socket and launches receiver thread."""
        if self.is_running:
            return

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass

        self.socket.bind((self.bind_ip, self.port))
        self.socket.settimeout(0.4)  # Allows clean shutdown checks

        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._receiver_loop,
            name="UDPAudioReceiverThread",
            daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        """Stops worker thread and releases socket."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.worker_thread = None

        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def _receiver_loop(self):
        """Dedicated UDP packet receiving and decryption loop."""
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception:
                if not self.is_running:
                    break
                continue

            if is_handshake_packet(data):
                if self.on_handshake is not None:
                    try:
                        self.on_handshake(data, addr)
                    except Exception:
                        pass
                continue

            with self._lock:
                self.packets_received += 1
                self.bytes_received += len(data)
                self._bytes_window += len(data)

                now = time.time()
                elapsed = now - self._last_time
                if elapsed >= 0.5:
                    self.bitrate_kbps = (self._bytes_window * 8.0) / (elapsed * 1000.0)
                    self._bytes_window = 0
                    self._last_time = now

                # Optional live demo of cryptographic tamper detection
                if self.simulate_tampering:
                    # Invert a byte in payload to forge/alter packet
                    tampered = bytearray(data)
                    tampered[-1] ^= 0xFF
                    data = bytes(tampered)

            # Authenticate and decrypt with AES-256-GCM
            pcm_bytes, seq_num, err = self.crypto.decrypt_packet(data)

            with self._lock:
                if err is not None:
                    # Packet was tampered with, corrupted, or forged!
                    self.packets_tampered += 1
                    continue

                # Decryption successful and cryptographically verified!
                self.packets_ok += 1

                # Track sequence numbers and packet loss
                if seq_num is not None:
                    if self.last_seq_num is not None:
                        expected = (self.last_seq_num + 1) & 0xFFFFFFFF
                        # Handle standard sequence increment and wrap-around
                        if seq_num > expected and (seq_num - expected) < 10000:
                            lost_count = seq_num - expected
                            self.packets_lost += lost_count
                    self.last_seq_num = seq_num

            # Feed authenticated PCM data into the Jitter Buffer
            if pcm_bytes is not None and seq_num is not None:
                lat = getattr(self.crypto, "last_latency_ms", None)
                if lat is not None:
                    with self._lock:
                        self._latencies.append(lat)
                        if len(self._latencies) > 64:
                            self._latencies.pop(0)
                        self.latency_ms = sum(self._latencies) / len(self._latencies)
                self.jitter_buffer.push(seq_num, pcm_bytes)

    def get_stats(self) -> Dict[str, Any]:
        """Returns comprehensive real-time receiver stats."""
        with self._lock:
            depth_blocks = self.jitter_buffer.get_depth_blocks()
            # Approx buffer depth in ms (at 44.1 kHz, 2048 samples = 46.44 ms per block)
            depth_ms = (depth_blocks * self.jitter_buffer.block_size / 44.100)
            
            return {
                "port": self.port,
                "packets_received": self.packets_received,
                "packets_ok": self.packets_ok,
                "packets_tampered": self.packets_tampered,
                "packets_lost": self.packets_lost,
                "bitrate_kbps": round(self.bitrate_kbps, 1),
                "jitter_blocks": depth_blocks,
                "jitter_ms": round(depth_ms, 1),
                "underruns": self.jitter_buffer.underruns,
                "simulate_tampering": self.simulate_tampering,
                "latency_ms": round(self.latency_ms, 2),
            }


class UDPDuplexLink:
    """
    Full-duplex encrypted UDP endpoint: one bound socket for TX + RX,
    optional X25519 handshake, then AES-256-GCM audio.
    """

    def __init__(
        self,
        local_port: int,
        peer_host: str,
        peer_port: int,
        crypto: AudioCrypto,
        jitter_buffer: JitterBuffer,
        bind_ip: str = "0.0.0.0",
        enable_dh: bool = True,
        max_queue_size: int = 32,
    ):
        self.local_port = int(local_port)
        self.peer_host = peer_host
        self.peer_port = int(peer_port)
        self.crypto = crypto
        self.jitter_buffer = jitter_buffer
        self.bind_ip = bind_ip
        self.enable_dh = enable_dh

        self.tx_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.socket: Optional[socket.socket] = None
        self.is_running = False
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._hello_thread: Optional[threading.Thread] = None

        self.dh = DiffieHellmanSession() if enable_dh else None
        self.dh_state = STATE_WAITING if enable_dh else STATE_PSK
        self.audio_ready = not enable_dh

        self.seq_num = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.tx_bitrate_kbps = 0.0
        self._tx_bytes_window = 0
        self._tx_last_time = time.time()

        self.packets_received = 0
        self.packets_ok = 0
        self.packets_tampered = 0
        self.packets_lost = 0
        self.last_seq_num: Optional[int] = None
        self.bytes_received = 0
        self.rx_bitrate_kbps = 0.0
        self.latency_ms = 0.0
        self._latencies = []
        self._rx_bytes_window = 0
        self._rx_last_time = time.time()
        self.simulate_tampering = False
        self._lock = threading.Lock()

    def set_peer(self, host: str, port: int):
        self.peer_host = host
        self.peer_port = int(port)

    def start(self):
        if self.is_running:
            return
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        self.socket.bind((self.bind_ip, self.local_port))
        self.socket.settimeout(0.2)

        self.is_running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name="DuplexRX", daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, name="DuplexTX", daemon=True)
        self._rx_thread.start()
        self._tx_thread.start()
        if self.enable_dh:
            self._hello_thread = threading.Thread(target=self._hello_loop, name="DuplexDH", daemon=True)
            self._hello_thread.start()

    def stop(self):
        self.is_running = False
        for t in (self._rx_thread, self._tx_thread, self._hello_thread):
            if t and t.is_alive():
                t.join(timeout=0.8)
        self._rx_thread = self._tx_thread = self._hello_thread = None
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def send_pcm_chunk(self, pcm_bytes: bytes):
        if not self.is_running or not self.audio_ready:
            return
        if self.tx_queue.full():
            try:
                self.tx_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.tx_queue.put_nowait(pcm_bytes)
        except queue.Full:
            pass

    def _hello_loop(self):
        while self.is_running and self.dh_state != STATE_ESTABLISHED:
            try:
                pkt = self.dh.local_hello()
                self.socket.sendto(pkt, (self.peer_host, self.peer_port))
            except Exception:
                pass
            time.sleep(0.25)

    def _apply_session_key(self):
        if self.dh and self.dh.session_key:
            self.crypto.set_key(self.dh.session_key)
            self.audio_ready = True
            self.dh_state = STATE_ESTABLISHED

    def _rx_loop(self):
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception:
                if not self.is_running:
                    break
                continue

            if is_handshake_packet(data):
                if self.dh is not None:
                    just_set, reply = self.dh.handle_packet(data)
                    if reply is not None:
                        try:
                            self.socket.sendto(reply, (self.peer_host, self.peer_port))
                        except Exception:
                            pass
                    if just_set:
                        self._apply_session_key()
                continue

            with self._lock:
                self.packets_received += 1
                self.bytes_received += len(data)
                self._rx_bytes_window += len(data)
                now = time.time()
                elapsed = now - self._rx_last_time
                if elapsed >= 0.5:
                    self.rx_bitrate_kbps = (self._rx_bytes_window * 8.0) / (elapsed * 1000.0)
                    self._rx_bytes_window = 0
                    self._rx_last_time = now
                if self.simulate_tampering:
                    tampered = bytearray(data)
                    tampered[-1] ^= 0xFF
                    data = bytes(tampered)

            if not self.audio_ready:
                continue

            pcm_bytes, seq_num, err = self.crypto.decrypt_packet(data)
            with self._lock:
                if err is not None:
                    self.packets_tampered += 1
                    continue
                self.packets_ok += 1
                if seq_num is not None:
                    if self.last_seq_num is not None:
                        expected = (self.last_seq_num + 1) & 0xFFFFFFFF
                        if seq_num > expected and (seq_num - expected) < 10000:
                            self.packets_lost += seq_num - expected
                    self.last_seq_num = seq_num

            if pcm_bytes is not None and seq_num is not None:
                lat = getattr(self.crypto, "last_latency_ms", None)
                if lat is not None:
                    with self._lock:
                        self._latencies.append(lat)
                        if len(self._latencies) > 64:
                            self._latencies.pop(0)
                        self.latency_ms = sum(self._latencies) / len(self._latencies)
                self.jitter_buffer.push(seq_num, pcm_bytes)

    def _tx_loop(self):
        while self.is_running:
            try:
                pcm_bytes = self.tx_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self.audio_ready:
                continue
            try:
                with self._lock:
                    current_seq = self.seq_num
                    self.seq_num = (self.seq_num + 1) & 0xFFFFFFFF
                packet = self.crypto.encrypt_packet(pcm_bytes, current_seq)
                self.socket.sendto(packet, (self.peer_host, self.peer_port))
                with self._lock:
                    self.packets_sent += 1
                    self.bytes_sent += len(packet)
                    self._tx_bytes_window += len(packet)
                    now = time.time()
                    elapsed = now - self._tx_last_time
                    if elapsed >= 0.5:
                        self.tx_bitrate_kbps = (self._tx_bytes_window * 8.0) / (elapsed * 1000.0)
                        self._tx_bytes_window = 0
                        self._tx_last_time = now
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            depth_blocks = self.jitter_buffer.get_depth_blocks()
            depth_ms = depth_blocks * self.jitter_buffer.block_size / 44.100
            return {
                "local_port": self.local_port,
                "peer": f"{self.peer_host}:{self.peer_port}",
                "dh_state": self.dh_state,
                "audio_ready": self.audio_ready,
                "packets_sent": self.packets_sent,
                "bytes_sent": self.bytes_sent,
                "tx_kbps": round(self.tx_bitrate_kbps, 1),
                "packets_received": self.packets_received,
                "packets_ok": self.packets_ok,
                "packets_tampered": self.packets_tampered,
                "packets_lost": self.packets_lost,
                "rx_kbps": round(self.rx_bitrate_kbps, 1),
                "latency_ms": round(self.latency_ms, 2),
                "jitter_blocks": depth_blocks,
                "jitter_ms": round(depth_ms, 1),
                "underruns": self.jitter_buffer.underruns,
                "simulate_tampering": self.simulate_tampering,
                "queue_size": self.tx_queue.qsize(),
            }
