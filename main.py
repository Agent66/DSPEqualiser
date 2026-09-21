import sys
import argparse
import tkinter as tk
from spequaliser.gui import EqualizerApp


def main():
    parser = argparse.ArgumentParser(description="SpEqualiser - Real-Time Audio Equalizer & Secure Streaming Suite")
    parser.add_argument(
        "--mode",
        choices=["gui", "sender", "receiver"],
        default="gui",
        help="Launch mode: 'gui' (Standard Equalizer Suite), 'sender' (Encrypted UDP Sender), 'receiver' (Encrypted UDP Receiver)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Sender destination host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default: 5005)")
    parser.add_argument("--key", default="shared_secret.key", help="Shared secret key path")
    parser.add_argument("--no-gui", action="store_true", help="Run sender or receiver in headless console mode")
    args, unknown = parser.parse_known_args()

    if args.mode == "sender":
        from sender import run_sender
        run_sender(host=args.host, port=args.port, key_path=args.key, no_gui=args.no_gui)
    elif args.mode == "receiver":
        from receiver import run_receiver
        run_receiver(port=args.port, key_path=args.key, no_gui=args.no_gui)
    else:
        root = tk.Tk()
        app = EqualizerApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()


if __name__ == "__main__":
    main()
