"""
X25519 Elliptic-Curve Diffie-Hellman session establishment over UDP.

Handshake datagram (38 bytes, unencrypted):
  [Magic: 4 = b"SPDH"] [Version: 1] [Type: 1] [Public key: 32]

Types:
  0x01 HELLO — advertise local public key
  0x02 ACK   — confirm receipt of peer public key
"""

from typing import Optional, Tuple

from crypto_utils import (
    generate_dh_keypair,
    derive_session_key,
    DH_PUBLIC_KEY_SIZE,
)

DH_MAGIC = b"SPDH"
DH_VERSION = 1
DH_HELLO = 0x01
DH_ACK = 0x02
DH_PACKET_SIZE = 4 + 1 + 1 + DH_PUBLIC_KEY_SIZE  # 38

STATE_WAITING = "waiting"
STATE_ESTABLISHED = "established"
STATE_PSK = "psk"


def is_handshake_packet(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == DH_MAGIC


def build_handshake_packet(msg_type: int, public_bytes: bytes) -> bytes:
    if len(public_bytes) != DH_PUBLIC_KEY_SIZE:
        raise ValueError("Invalid X25519 public key length")
    return DH_MAGIC + bytes([DH_VERSION, msg_type & 0xFF]) + public_bytes


def parse_handshake_packet(data: bytes) -> Tuple[Optional[int], Optional[bytes], Optional[str]]:
    if len(data) < DH_PACKET_SIZE:
        return None, None, "DH_PACKET_TOO_SHORT"
    if data[:4] != DH_MAGIC:
        return None, None, "DH_INVALID_MAGIC"
    version = data[4]
    if version != DH_VERSION:
        return None, None, f"DH_UNSUPPORTED_VERSION ({version})"
    msg_type = data[5]
    if msg_type not in (DH_HELLO, DH_ACK):
        return None, None, f"DH_UNKNOWN_TYPE ({msg_type})"
    return msg_type, data[6:6 + DH_PUBLIC_KEY_SIZE], None


class DiffieHellmanSession:
    """
    Completes ECDH with a peer and yields a 256-bit AES-GCM session key.
    Retransmits HELLO until the peer public key arrives.
    """

    def __init__(self):
        self.private_key, self.public_bytes = generate_dh_keypair()
        self.peer_public: Optional[bytes] = None
        self.session_key: Optional[bytes] = None
        self.state = STATE_WAITING
        self.hellos_sent = 0
        self.acks_sent = 0

    def local_hello(self) -> bytes:
        self.hellos_sent += 1
        return build_handshake_packet(DH_HELLO, self.public_bytes)

    def local_ack(self) -> bytes:
        self.acks_sent += 1
        return build_handshake_packet(DH_ACK, self.public_bytes)

    def handle_packet(self, data: bytes) -> Tuple[bool, Optional[bytes]]:
        """
        Process an incoming handshake datagram.

        Returns:
            (just_established, reply_packet_or_none)
        """
        msg_type, peer_pub, err = parse_handshake_packet(data)
        if err or peer_pub is None:
            return False, None
        if peer_pub == self.public_bytes:
            return False, None

        already = self.state == STATE_ESTABLISHED
        self.peer_public = peer_pub
        self.session_key = derive_session_key(self.private_key, peer_pub)
        self.state = STATE_ESTABLISHED
        reply = self.local_ack() if msg_type == DH_HELLO else None
        return (not already), reply
