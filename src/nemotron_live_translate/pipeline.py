"""End-to-end speech-to-English translation pipeline."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import wave

from .asr import NemotronASR
from .audio import coerce_audio_to_wav
from .locales import ADAPTATION_READY, PROMPT_ONLY, locale_info_for, normalize_source_locale
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
        result = None
        for result in self.iter_translate_audio(audio, source_locale=source_locale, chunk_ms=chunk_ms):
            pass
        assert result is not None
        return result

    def iter_translate_audio(self, audio: object, source_locale: str = "auto", chunk_ms: int | str = 320):
        started = time.perf_counter()
        selected_locale = normalize_source_locale(source_locale)
        try:
            if audio is None:
                yield self._result(started, status="No audio received.")
                return

            if selected_locale != "auto":
                locale_info_for(selected_locale)

            yield self._result(started, status="Audio received from browser. Converting to mono 16 kHz WAV...")
            wav_path = coerce_audio_to_wav(audio, self.work_dir)
            yield self._result(
                started,
                status=f"Audio ready for backend: {_audio_summary(wav_path)}. Loading/running Nemotron ASR...",
            )

            asr_result = self.asr.transcribe_file(wav_path, selected_locale, int(chunk_ms))
            source_text, tagged_locale = split_language_tagged_text(asr_result.text)
            detected_locale = tagged_locale or (selected_locale if selected_locale != "auto" else "")
            if not detected_locale:
                raise RuntimeError("Nemotron returned no language tag. Select the source language and retry.")

            locale = locale_info_for(detected_locale)
            if not source_text.strip():
                yield self._result(
                    started,
                    source_text=source_text,
                    detected_locale=detected_locale,
                    status=f"Error: {_empty_transcript_message(locale, selected_locale)}",
                )
                return

            if locale.tier == PROMPT_ONLY:
                yield self._result(
                    started,
                    source_text=source_text,
                    detected_locale=detected_locale,
                    status=f"Error: {_prompt_only_transcript_message(locale, source_text)}",
                )
                return

            if _looks_unusable_transcript(source_text):
                yield self._result(
                    started,
                    source_text=source_text,
                    detected_locale=detected_locale,
                    status=(
                        f"Error: Nemotron returned an unusable transcript for {detected_locale}. "
                        "The output contains too many unknown-token placeholders, so translation and TTS were skipped."
                    ),
                )
                return

            yield self._result(
                started,
                source_text=source_text,
                detected_locale=detected_locale,
                status=f"{asr_result.status}. Translating {detected_locale} to English...",
            )

            translation = self.translator.translate(source_text, detected_locale)
            yield self._result(
                started,
                source_text=source_text,
                detected_locale=detected_locale,
                english_text=translation.text,
                status=f"{asr_result.status}; {translation.status}. Generating spoken English...",
            )

            spoken = self.tts.speak(translation.text)

            status_parts = [asr_result.status, translation.status, spoken.status]
            if locale.tier == ADAPTATION_READY:
                status_parts.append(f"{locale.nemotron_locale} is adaptation-ready; transcription quality may be limited.")
            if locale.tier == PROMPT_ONLY:
                status_parts.append(
                    f"{locale.nemotron_locale} is prompt-only in this checkpoint; use a fine-tuned Nemotron model for reliable transcription."
                )

            yield self._result(
                started,
                source_text=source_text,
                detected_locale=detected_locale,
                english_text=translation.text,
                audio_path=spoken.audio_path,
                status="; ".join(part for part in status_parts if part),
            )
        except Exception as exc:
            yield self._result(started, status=f"Error: {exc}")

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


def _audio_summary(path: Path) -> str:
    try:
        with wave.open(str(path), "rb") as wav:
            duration = wav.getnframes() / float(wav.getframerate())
            return (
                f"{path.name}, {duration:.2f}s, {wav.getframerate()} Hz, "
                f"{wav.getnchannels()} channel, {path.stat().st_size} bytes"
            )
    except Exception:
        return f"{path.name}, {path.stat().st_size} bytes"


def _empty_transcript_message(locale, selected_locale: str) -> str:
    if getattr(locale, "tier", "") == PROMPT_ONLY:
        return (
            f"Nemotron accepted {locale.nemotron_locale}, but the released checkpoint produced an empty transcript. "
            "Tamil is present in the model prompt dictionary but is not listed in NVIDIA's supported 40 language-locales; "
            "use a fine-tuned Nemotron checkpoint for Tamil transcription."
        )
    if selected_locale == "auto":
        return "Nemotron returned an empty transcript. Try a longer recording or select the source language explicitly."
    return f"Nemotron returned an empty transcript for {locale.nemotron_locale}. Try a longer, clearer recording."


def _prompt_only_transcript_message(locale, source_text: str) -> str:
    message = (
        f"{locale.nemotron_locale} is prompt-only in this Nemotron checkpoint. "
        "The model can accept the language prompt, but NVIDIA does not list Tamil in the supported 40 language-locales, "
        "so the ASR transcript is not reliable enough to translate. Use a fine-tuned Nemotron checkpoint for Tamil."
    )
    if _looks_unusable_transcript(source_text):
        message += " The transcript also contains unknown-token placeholder symbols."
    return message


def _looks_unusable_transcript(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return True

    placeholder_count = sum(1 for char in clean if char in {"?", "\u2047", "\ufffd"})
    visible_count = sum(1 for char in clean if not char.isspace())
    if placeholder_count >= 3 and visible_count > 0 and placeholder_count / visible_count >= 0.2:
        return True

    letters = sum(1 for char in clean if char.isalpha())
    if visible_count >= 12 and letters == 0:
        return True
    return False
