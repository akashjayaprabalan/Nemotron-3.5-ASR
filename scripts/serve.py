"""Detached Gradio server runner for local verification."""

from pathlib import Path
import os
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_live_translate.ui import build_demo


def main() -> None:
    demo = build_demo()
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        prevent_thread_lock=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        demo.close()


if __name__ == "__main__":
    main()

