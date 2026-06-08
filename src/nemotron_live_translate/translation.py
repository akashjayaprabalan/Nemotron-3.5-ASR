"""NLLB text translation wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import threading

from .locales import is_english_locale, nllb_code_for_locale


NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
ENGLISH_NLLB_CODE = "eng_Latn"


@dataclass(frozen=True, slots=True)
class TranslationText:
    text: str
    device: str
    status: str


class NLLBTranslator:
    def __init__(self, model_name: str = NLLB_MODEL_ID, prefer_mps: bool = True) -> None:
        self.model_name = model_name
        self.prefer_mps = prefer_mps
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._forced_device: str | None = None
        self._lock = threading.Lock()

    @property
    def device_name(self) -> str:
        if self._device is None:
            return "not loaded"
        return str(self._device)

    def translate(self, text: str, source_locale: str) -> TranslationText:
        clean_text = (text or "").strip()
        if clean_text == "":
            return TranslationText(text="", device=self.device_name, status="Translation skipped for empty transcript")
        if is_english_locale(source_locale):
            return TranslationText(text=clean_text, device=self.device_name, status="English source; translation skipped")

        try:
            return self._translate_once(clean_text, source_locale)
        except Exception as exc:
            if self.device_name == "mps":
                mps_error = _brief_error(exc)
                self._reset(force_device="cpu")
                result = self._translate_once(clean_text, source_locale)
                return TranslationText(
                    text=result.text,
                    device=result.device,
                    status=f"{result.status}; retried on CPU after MPS error: {mps_error}",
                )
            raise

    def _translate_once(self, text: str, source_locale: str) -> TranslationText:
        source_code = nllb_code_for_locale(source_locale)
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        self._tokenizer.src_lang = source_code
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if hasattr(encoded, "to"):
            encoded = encoded.to(self._device)

        forced_bos_token_id = _forced_bos_token_id(self._tokenizer, ENGLISH_NLLB_CODE)
        with self._torch.inference_mode():
            generated = self._model.generate(**encoded, forced_bos_token_id=forced_bos_token_id)
        translated = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return TranslationText(text=translated, device=self.device_name, status=f"NLLB translation completed on {self.device_name}")

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except Exception as exc:
                raise RuntimeError("NLLB translation requires PyTorch and Transformers. Install the app dependencies.") from exc

            device = self._select_device(torch)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            model = model.to(device)
            model.eval()

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device

    def _select_device(self, torch_module):
        if self._forced_device is not None:
            return torch_module.device(self._forced_device)
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        mps = getattr(torch_module.backends, "mps", None)
        if self.prefer_mps and mps is not None and mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def _reset(self, force_device: str | None = None) -> None:
        with self._lock:
            self._tokenizer = None
            self._model = None
            self._torch = None
            self._device = None
            self._forced_device = force_device


def _forced_bos_token_id(tokenizer, lang_code: str) -> int:
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if mapping and lang_code in mapping:
        return int(mapping[lang_code])
    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"Tokenizer does not know NLLB language code: {lang_code}")
    return int(token_id)


def _brief_error(exc: BaseException, limit: int = 160) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")

