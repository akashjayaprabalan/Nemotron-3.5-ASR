"""Supported Nemotron locales and NLLB translation mapping."""

from __future__ import annotations

from dataclasses import dataclass


TRANSCRIPTION_READY = "transcription-ready"
BROAD_COVERAGE = "broad-coverage"
ADAPTATION_READY = "adaptation-ready"
PROMPT_ONLY = "prompt-only"


@dataclass(frozen=True, slots=True)
class LocaleInfo:
    nemotron_locale: str
    name: str
    nllb_code: str
    tier: str


SUPPORTED_LOCALES: tuple[LocaleInfo, ...] = (
    LocaleInfo("en-US", "English (US)", "eng_Latn", TRANSCRIPTION_READY),
    LocaleInfo("en-GB", "English (UK)", "eng_Latn", TRANSCRIPTION_READY),
    LocaleInfo("es-US", "Spanish (US)", "spa_Latn", TRANSCRIPTION_READY),
    LocaleInfo("es-ES", "Spanish (Spain)", "spa_Latn", TRANSCRIPTION_READY),
    LocaleInfo("fr-FR", "French (France)", "fra_Latn", TRANSCRIPTION_READY),
    LocaleInfo("fr-CA", "French (Canada)", "fra_Latn", TRANSCRIPTION_READY),
    LocaleInfo("it-IT", "Italian", "ita_Latn", TRANSCRIPTION_READY),
    LocaleInfo("pt-BR", "Portuguese (Brazil)", "por_Latn", TRANSCRIPTION_READY),
    LocaleInfo("pt-PT", "Portuguese (Portugal)", "por_Latn", TRANSCRIPTION_READY),
    LocaleInfo("nl-NL", "Dutch", "nld_Latn", TRANSCRIPTION_READY),
    LocaleInfo("de-DE", "German", "deu_Latn", TRANSCRIPTION_READY),
    LocaleInfo("tr-TR", "Turkish", "tur_Latn", TRANSCRIPTION_READY),
    LocaleInfo("ru-RU", "Russian", "rus_Cyrl", TRANSCRIPTION_READY),
    LocaleInfo("ar-AR", "Arabic", "arb_Arab", TRANSCRIPTION_READY),
    LocaleInfo("hi-IN", "Hindi", "hin_Deva", TRANSCRIPTION_READY),
    LocaleInfo("ja-JP", "Japanese", "jpn_Jpan", TRANSCRIPTION_READY),
    LocaleInfo("ko-KR", "Korean", "kor_Hang", TRANSCRIPTION_READY),
    LocaleInfo("vi-VN", "Vietnamese", "vie_Latn", TRANSCRIPTION_READY),
    LocaleInfo("uk-UA", "Ukrainian", "ukr_Cyrl", TRANSCRIPTION_READY),
    LocaleInfo("pl-PL", "Polish", "pol_Latn", BROAD_COVERAGE),
    LocaleInfo("sv-SE", "Swedish", "swe_Latn", BROAD_COVERAGE),
    LocaleInfo("cs-CZ", "Czech", "ces_Latn", BROAD_COVERAGE),
    LocaleInfo("nb-NO", "Norwegian Bokmal", "nob_Latn", BROAD_COVERAGE),
    LocaleInfo("da-DK", "Danish", "dan_Latn", BROAD_COVERAGE),
    LocaleInfo("bg-BG", "Bulgarian", "bul_Cyrl", BROAD_COVERAGE),
    LocaleInfo("fi-FI", "Finnish", "fin_Latn", BROAD_COVERAGE),
    LocaleInfo("hr-HR", "Croatian", "hrv_Latn", BROAD_COVERAGE),
    LocaleInfo("sk-SK", "Slovak", "slk_Latn", BROAD_COVERAGE),
    LocaleInfo("zh-CN", "Mandarin", "zho_Hans", BROAD_COVERAGE),
    LocaleInfo("hu-HU", "Hungarian", "hun_Latn", BROAD_COVERAGE),
    LocaleInfo("ro-RO", "Romanian", "ron_Latn", BROAD_COVERAGE),
    LocaleInfo("et-EE", "Estonian", "est_Latn", BROAD_COVERAGE),
    LocaleInfo("el-GR", "Greek", "ell_Grek", ADAPTATION_READY),
    LocaleInfo("lt-LT", "Lithuanian", "lit_Latn", ADAPTATION_READY),
    LocaleInfo("lv-LV", "Latvian", "lvs_Latn", ADAPTATION_READY),
    LocaleInfo("mt-MT", "Maltese", "mlt_Latn", ADAPTATION_READY),
    LocaleInfo("sl-SI", "Slovenian", "slv_Latn", ADAPTATION_READY),
    LocaleInfo("he-IL", "Hebrew", "heb_Hebr", ADAPTATION_READY),
    LocaleInfo("th-TH", "Thai", "tha_Thai", ADAPTATION_READY),
    LocaleInfo("nn-NO", "Norwegian Nynorsk", "nno_Latn", ADAPTATION_READY),
    LocaleInfo("ta-IN", "Tamil", "tam_Taml", PROMPT_ONLY),
)

LOCALE_BY_CODE = {item.nemotron_locale: item for item in SUPPORTED_LOCALES}
ENGLISH_LOCALES = frozenset({"en-US", "en-GB"})
ADAPTATION_READY_LOCALES = frozenset(item.nemotron_locale for item in SUPPORTED_LOCALES if item.tier == ADAPTATION_READY)
PROMPT_ONLY_LOCALES = frozenset(item.nemotron_locale for item in SUPPORTED_LOCALES if item.tier == PROMPT_ONLY)


def locale_info_for(locale: str) -> LocaleInfo:
    normalized = normalize_source_locale(locale)
    try:
        return LOCALE_BY_CODE[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Nemotron locale: {locale}") from exc


def nllb_code_for_locale(locale: str) -> str:
    return locale_info_for(locale).nllb_code


def is_english_locale(locale: str) -> bool:
    return normalize_source_locale(locale) in ENGLISH_LOCALES


def normalize_source_locale(locale: str | None) -> str:
    if locale is None:
        return "auto"
    value = str(locale).strip()
    if value == "" or value.lower() == "auto":
        return "auto"
    return value


def gradio_locale_choices() -> list[tuple[str, str]]:
    return [("Auto", "auto")] + [
        (_choice_label(item), item.nemotron_locale) for item in SUPPORTED_LOCALES
    ]


def _choice_label(item: LocaleInfo) -> str:
    if item.tier == PROMPT_ONLY:
        return f"{item.name} ({item.nemotron_locale}, prompt-only)"
    return f"{item.name} ({item.nemotron_locale})"
