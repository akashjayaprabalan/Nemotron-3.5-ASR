import unittest

from nemotron_live_translate.locales import (
    ADAPTATION_READY_LOCALES,
    ENGLISH_LOCALES,
    SUPPORTED_LOCALES,
    nllb_code_for_locale,
)


class LocaleTests(unittest.TestCase):
    def test_all_40_locales_have_nllb_codes(self):
        self.assertEqual(len(SUPPORTED_LOCALES), 40)
        self.assertEqual(len({item.nemotron_locale for item in SUPPORTED_LOCALES}), 40)
        for item in SUPPORTED_LOCALES:
            self.assertTrue(item.nllb_code)
            self.assertEqual(nllb_code_for_locale(item.nemotron_locale), item.nllb_code)

    def test_english_and_adaptation_sets(self):
        self.assertEqual(ENGLISH_LOCALES, {"en-US", "en-GB"})
        self.assertEqual(
            ADAPTATION_READY_LOCALES,
            {"el-GR", "lt-LT", "lv-LV", "mt-MT", "sl-SI", "he-IL", "th-TH", "nn-NO"},
        )


if __name__ == "__main__":
    unittest.main()

