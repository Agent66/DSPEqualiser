"""
Key Generation Script for AES-256-GCM Secure Audio Streaming.

Generates a cryptographically secure 256-bit (32-byte) random secret key
and writes it to 'shared_secret.key'.
"""

import argparse
import sys
from pathlib import Path
from crypto_utils import generate_key, save_key, KEY_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 256-bit AES-GCM shared key for SpEqualiser secure audio streaming."
    )
    parser.add_argument(
        "-o", "--out",
        default="shared_secret.key",
        help="Destination path for shared secret key (default: shared_secret.key)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing key file without asking"
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Warning: Key file '{out_path}' already exists.")
        try:
            choice = input("Overwrite existing key? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if choice != 'y':
            print("Aborted without modifying key file.")
            sys.exit(0)

    key = generate_key()
    saved_path = save_key(key, out_path)

    print("=" * 65)
    print("  AES-256-GCM Shared Key Generated Successfully")
    print("=" * 65)
    print(f"  Key Size    : {KEY_SIZE * 8} bits ({KEY_SIZE} bytes)")
    print(f"  Saved To    : {saved_path}")
    print("  Key Fingerprint (SHA-256 / 8 bytes):", key[:8].hex())
    print("-" * 65)
    print("Next steps:")
    print("  1. If testing on one machine: keep 'shared_secret.key' in the project folder.")
    print("  2. If testing on two machines: copy 'shared_secret.key' to the receiver machine.")
    print("  3. Start receiver first     : python receiver.py --port 5005")
    print("  4. Start sender             : python sender.py --host 127.0.0.1 --port 5005")
    print("=" * 65)


if __name__ == "__main__":
    main()
