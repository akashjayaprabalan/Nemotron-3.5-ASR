"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranslationResult:
    source_text: str
    detected_locale: str
    english_text: str
    audio_path: str | None
    latency_ms: int
    status: str

