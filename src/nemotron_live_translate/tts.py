"""macOS text-to-speech wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import uuid


@dataclass(frozen=True, slots=True)
class TTSResult:
    audio_path: str | None
    status: str


class MacSayTTS:
    def __init__(self, voice: str = "Samantha", output_dir: str | Path | None = None, timeout_seconds: int = 60) -> None:
        self.voice = voice
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "nemotron-live-translate"
        self.timeout_seconds = timeout_seconds

    def speak(self, text: str) -> TTSResult:
        clean_text = (text or "").strip()
        if clean_text == "":
            return TTSResult(audio_path=None, status="TTS skipped for empty text")
        if platform.system() != "Darwin":
            raise RuntimeError("macOS 'say' TTS is only available on Darwin/macOS.")

        say = shutil.which("say")
        afconvert = shutil.which("afconvert")
        if say is None:
            raise RuntimeError("macOS 'say' command was not found.")
        if afconvert is None:
            raise RuntimeError("macOS 'afconvert' command was not found.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"spoken-{uuid.uuid4().hex}"
        aiff_path = self.output_dir / f"{stem}.aiff"
        wav_path = self.output_dir / f"{stem}.wav"

        subprocess.run(
            [say, "-v", self.voice, "-o", str(aiff_path), clean_text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout_seconds,
        )
        subprocess.run(
            [afconvert, str(aiff_path), str(wav_path), "-f", "WAVE", "-d", "LEI16"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout_seconds,
        )
        try:
            aiff_path.unlink()
        except FileNotFoundError:
            pass
        return TTSResult(audio_path=str(wav_path), status=f"English speech generated with macOS say ({self.voice})")

