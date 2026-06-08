"""End-to-end speech-to-English translation pipeline."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time

from .asr import NemotronASR
from .audio import coerce_audio_to_wav
from .locales import ADAPTATION_READY, locale_info_for, normalize_source_locale
from .schemas import TranslationResult
from .text import split_language_tagged_text
from .translation import NLLBTranslator
from .tts import MacSayTTS


class TranslationPipeline:
    def __init__(
        self,
        asr: object | None = None,
        translator: object | None = None,
        tts: object | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.asr = asr if asr is not None else NemotronASR()
        self.translator = translator if translator is not None else NLLBTranslator()
        self.tts = tts if tts is not None else MacSayTTS()
        self.work_dir = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir()) / "nemotron-live-translate"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def translate_audio(self, audio: object, source_locale: str = "auto", chunk_ms: int | str = 320) -> TranslationResult:
        started = time.perf_counter()
        selected_locale = normalize_source_locale(source_locale)
        try:
            if audio is None:
                return self._result(started, status="No audio received.")

            if selected_locale != "auto":
                locale_info_for(selected_locale)

            wav_path = coerce_audio_to_wav(audio, self.work_dir)
            asr_result = self.asr.transcribe_file(wav_path, selected_locale, int(chunk_ms))
            source_text, tagged_locale = split_language_tagged_text(asr_result.text)
            detected_locale = tagged_locale or (selected_locale if selected_locale != "auto" else "")
            if not detected_locale:
                raise RuntimeError("Nemotron returned no language tag. Select the source language and retry.")

            locale = locale_info_for(detected_locale)
            translation = self.translator.translate(source_text, detected_locale)
            spoken = self.tts.speak(translation.text)

            status_parts = [asr_result.status, translation.status, spoken.status]
            if locale.tier == ADAPTATION_READY:
                status_parts.append(f"{locale.nemotron_locale} is adaptation-ready; transcription quality may be limited.")

            return self._result(
                started,
                source_text=source_text,
                detected_locale=detected_locale,
                english_text=translation.text,
                audio_path=spoken.audio_path,
                status="; ".join(part for part in status_parts if part),
            )
        except Exception as exc:
            return self._result(started, status=f"Error: {exc}")

    def _result(
        self,
        started: float,
        source_text: str = "",
        detected_locale: str = "",
        english_text: str = "",
        audio_path: str | None = None,
        status: str = "",
    ) -> TranslationResult:
        return TranslationResult(
            source_text=source_text,
            detected_locale=detected_locale,
            english_text=english_text,
            audio_path=audio_path,
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
            status=status,
        )


_DEFAULT_PIPELINE: TranslationPipeline | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_pipeline() -> TranslationPipeline:
    global _DEFAULT_PIPELINE
    with _DEFAULT_LOCK:
        if _DEFAULT_PIPELINE is None:
            _DEFAULT_PIPELINE = TranslationPipeline()
        return _DEFAULT_PIPELINE


def translate_audio(audio: object, source_locale: str = "auto", chunk_ms: int | str = 320) -> TranslationResult:
    return get_default_pipeline().translate_audio(audio, source_locale=source_locale, chunk_ms=chunk_ms)

