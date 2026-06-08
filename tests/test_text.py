import unittest

from nemotron_live_translate.locales import locale_info_for
from nemotron_live_translate.text import parse_language_tag, split_language_tagged_text, strip_language_tag
from nemotron_live_translate.translation import NLLBTranslator


class TextTests(unittest.TestCase):
    def test_parse_and_strip_language_tag(self):
        text = "Bonjour. <fr-FR>"
        self.assertEqual(parse_language_tag(text), "fr-FR")
        self.assertEqual(strip_language_tag(text), "Bonjour.")
        self.assertEqual(split_language_tagged_text(text), ("Bonjour.", "fr-FR"))

    def test_no_language_tag(self):
        self.assertIsNone(parse_language_tag("Hello."))
        self.assertEqual(strip_language_tag("Hello."), "Hello.")

    def test_unknown_locale_error(self):
        with self.assertRaises(ValueError):
            locale_info_for("xx-XX")

    def test_english_identity_translation_does_not_load_model(self):
        translator = NLLBTranslator()
        result = translator.translate("Hello.", "en-US")
        self.assertEqual(result.text, "Hello.")
        self.assertIn("skipped", result.status)
        self.assertEqual(translator.device_name, "not loaded")


if __name__ == "__main__":
    unittest.main()

