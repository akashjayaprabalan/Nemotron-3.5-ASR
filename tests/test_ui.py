import gc
import importlib.util
import unittest
import warnings


warnings.filterwarnings("ignore", category=ResourceWarning)


@unittest.skipIf(importlib.util.find_spec("gradio") is None, "Gradio is not installed")
class UISmokeTests(unittest.TestCase):
    def test_build_demo_with_injected_pipeline(self):
        from nemotron_live_translate.schemas import TranslationResult
        from nemotron_live_translate.ui import build_demo

        class FakePipeline:
            def translate_audio(self, audio, source_locale="auto", chunk_ms=320):
                return TranslationResult("hola.", "es-ES", "hello.", None, 1, "ok")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            demo = build_demo(pipeline=FakePipeline())
            try:
                self.assertIsNotNone(demo)
            finally:
                demo.close()
                del demo
                gc.collect()


if __name__ == "__main__":
    unittest.main()
