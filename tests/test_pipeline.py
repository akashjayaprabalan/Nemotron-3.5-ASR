import tempfile
import unittest
from pathlib import Path

from nemotron_live_translate.asr import ASRResult
from nemotron_live_translate.pipeline import TranslationPipeline
from nemotron_live_translate.translation import TranslationText
from nemotron_live_translate.tts import TTSResult


class FakeASR:
    def __init__(self, text):
        self.text = text

    def transcribe_file(self, audio_path, source_locale, chunk_ms):
        return ASRResult(text=self.text, device="fake", status="fake asr")


class FakeTranslator:
    def __init__(self, text):
        self.text = text

    def translate(self, text, source_locale):
        return TranslationText(text=self.text, device="fake", status="fake translation")


class FakeTTS:
    def __init__(self, path):
        self.path = path

    def speak(self, text):
        return TTSResult(audio_path=str(self.path), status="fake tts")


class FailIfCalled:
    def translate(self, text, source_locale):
        raise AssertionError("translator should not be called")

    def speak(self, text):
        raise AssertionError("tts should not be called")


class PipelineTests(unittest.TestCase):
    def test_pipeline_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_out = Path(tmp) / "spoken.wav"
            pipeline = TranslationPipeline(
                asr=FakeASR("hola. <es-ES>"),
                translator=FakeTranslator("hello."),
                tts=FakeTTS(audio_out),
                work_dir=tmp,
            )
            result = pipeline.translate_audio((8000, [0.0, 0.1, 0.0]), source_locale="auto", chunk_ms=320)

            self.assertEqual(result.source_text, "hola.")
            self.assertEqual(result.detected_locale, "es-ES")
            self.assertEqual(result.english_text, "hello.")
            self.assertEqual(result.audio_path, str(audio_out))
            self.assertIn("fake asr", result.status)

    def test_pipeline_reports_backend_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_out = Path(tmp) / "spoken.wav"
            pipeline = TranslationPipeline(
                asr=FakeASR("hola. <es-ES>"),
                translator=FakeTranslator("hello."),
                tts=FakeTTS(audio_out),
                work_dir=tmp,
            )
            results = list(pipeline.iter_translate_audio((8000, [0.0, 0.1, 0.0]), source_locale="auto", chunk_ms=320))

            self.assertGreaterEqual(len(results), 5)
            self.assertIn("Audio received from browser", results[0].status)
            self.assertIn("Audio ready for backend", results[1].status)
            self.assertEqual(results[-1].source_text, "hola.")
            self.assertEqual(results[-1].english_text, "hello.")

    def test_unknown_detected_locale_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = TranslationPipeline(
                asr=FakeASR("hello. <xx-XX>"),
                translator=FakeTranslator("hello."),
                tts=FakeTTS(Path(tmp) / "spoken.wav"),
                work_dir=tmp,
            )
            result = pipeline.translate_audio((8000, [0.0, 0.1, 0.0]))
            self.assertIn("Unsupported Nemotron locale", result.status)
            self.assertEqual(result.english_text, "")

    def test_empty_tamil_transcript_reports_prompt_only_model_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = TranslationPipeline(
                asr=FakeASR(""),
                translator=FailIfCalled(),
                tts=FailIfCalled(),
                work_dir=tmp,
            )
            result = pipeline.translate_audio((8000, [0.0, 0.1, 0.0]), source_locale="ta-IN")
            self.assertIn("produced an empty transcript", result.status)
            self.assertIn("not listed in NVIDIA's supported 40 language-locales", result.status)
            self.assertEqual(result.detected_locale, "")
            self.assertEqual(result.english_text, "")


if __name__ == "__main__":
    unittest.main()
