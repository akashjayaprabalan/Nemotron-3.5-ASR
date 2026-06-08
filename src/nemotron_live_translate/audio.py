"""Audio normalization helpers for Nemotron ASR."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence
import uuid
import wave


TARGET_SAMPLE_RATE = 16_000
ATT_CONTEXT_BY_CHUNK_MS = {
    80: [56, 0],
    160: [56, 1],
    320: [56, 3],
    560: [56, 6],
    1120: [56, 13],
}


def att_context_for_chunk_ms(chunk_ms: int | str) -> list[int]:
    value = int(chunk_ms)
    try:
        return list(ATT_CONTEXT_BY_CHUNK_MS[value])
    except KeyError as exc:
        allowed = ", ".join(str(item) for item in ATT_CONTEXT_BY_CHUNK_MS)
        raise ValueError(f"Unsupported chunk size {value} ms. Choose one of: {allowed}.") from exc


def chunk_frames_for_chunk_ms(chunk_ms: int | str) -> int:
    context = att_context_for_chunk_ms(chunk_ms)
    return context[1] + 1


def coerce_audio_to_wav(audio: object, work_dir: str | Path | None = None) -> Path:
    """Convert Gradio audio input or a file path to mono 16 kHz WAV."""

    if audio is None:
        raise ValueError("No audio was provided.")

    output_dir = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="nemotron-audio-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        if _is_compatible_wav(path):
            return path
        return _convert_file_with_ffmpeg(path, output_dir)

    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, samples = audio
        sample_rate = int(sample_rate)
        mono = normalize_samples(samples)
        resampled = resample_linear(mono, sample_rate, TARGET_SAMPLE_RATE)
        output_path = output_dir / f"input-{uuid.uuid4().hex}.wav"
        write_wav(output_path, resampled, TARGET_SAMPLE_RATE)
        return output_path

    raise TypeError("Audio must be a file path or a Gradio-style (sample_rate, samples) tuple.")


def normalize_samples(samples: object) -> list[float]:
    """Return mono float samples in [-1.0, 1.0]."""

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - depends on optional runtime dependency
        np = None

    if np is not None and isinstance(samples, np.ndarray):
        array = samples
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim == 2:
            array = array.mean(axis=1)
        elif array.ndim > 2:
            array = array.reshape(-1, array.shape[-1]).mean(axis=1)

        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            scale = float(max(abs(info.min), info.max))
            array = array.astype(np.float32) / scale
        else:
            array = array.astype(np.float32)
        return np.clip(array, -1.0, 1.0).tolist()

    values: list[float] = []
    for frame in _as_iterable(samples):
        if isinstance(frame, (list, tuple)):
            if len(frame) == 0:
                continue
            values.append(sum(float(item) for item in frame) / len(frame))
        else:
            values.append(float(frame))

    if not values:
        raise ValueError("Audio sample buffer is empty.")

    max_abs = max(abs(item) for item in values)
    if max_abs > 1.0:
        scale = 32768.0 if max_abs <= 32768.0 else 2147483648.0
        values = [item / scale for item in values]

    return [max(-1.0, min(1.0, item)) for item in values]


def resample_linear(samples: Sequence[float], source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> list[float]:
    if source_rate <= 0:
        raise ValueError("Source sample rate must be positive.")
    if not samples:
        raise ValueError("Audio sample buffer is empty.")
    if source_rate == target_rate:
        return list(samples)

    target_len = max(1, int(round(len(samples) * target_rate / source_rate)))
    if target_len == 1:
        return [float(samples[0])]

    step = (len(samples) - 1) / (target_len - 1)
    output: list[float] = []
    for index in range(target_len):
        pos = index * step
        left = int(math.floor(pos))
        right = min(left + 1, len(samples) - 1)
        ratio = pos - left
        output.append(float(samples[left]) * (1.0 - ratio) + float(samples[right]) * ratio)
    return output


def write_wav(path: str | Path, samples: Sequence[float], sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            clipped = max(-1.0, min(1.0, float(sample)))
            value = int(round(clipped * 32767.0))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(frames))
    return output


def _as_iterable(value: object) -> Iterable[object]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError("Audio samples must be iterable.")


def _is_compatible_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getnchannels() == 1
                and wav.getframerate() == TARGET_SAMPLE_RATE
                and wav.getsampwidth() == 2
            )
    except wave.Error:
        return False


def _convert_file_with_ffmpeg(path: Path, output_dir: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to convert uploaded audio. Install it with: brew install ffmpeg")
    output_path = output_dir / f"input-{uuid.uuid4().hex}.wav"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-f", "wav", str(output_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_path

