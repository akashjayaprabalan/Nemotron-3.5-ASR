"""Lazy Nemotron 3.5 ASR wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from .audio import att_context_for_chunk_ms
from .locales import locale_info_for, normalize_source_locale


NEMOTRON_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    device: str
    status: str


class NemotronASR:
    """Cache-aware streaming inference for Nemotron 3.5 ASR.

    The implementation mirrors NVIDIA NeMo's cache-aware streaming example but
    keeps model loading lazy so the Gradio app can start quickly.
    """

    def __init__(self, model_name: str = NEMOTRON_MODEL_ID, prefer_mps: bool = True) -> None:
        self.model_name = model_name
        self.prefer_mps = prefer_mps
        self._model = None
        self._torch = None
        self._streaming_buffer_cls = None
        self._device = None
        self._lock = threading.Lock()
        self._forced_device: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str:
        if self._device is None:
            return "not loaded"
        return str(self._device)

    def transcribe_file(self, audio_path: str | Path, source_locale: str = "auto", chunk_ms: int | str = 320) -> ASRResult:
        source = normalize_source_locale(source_locale)
        if source != "auto":
            locale_info_for(source)

        try:
            return self._transcribe_file_once(Path(audio_path), source, int(chunk_ms))
        except Exception as exc:
            if self.device_name == "mps":
                mps_error = _brief_error(exc)
                self._reset(force_device="cpu")
                result = self._transcribe_file_once(Path(audio_path), source, int(chunk_ms))
                return ASRResult(
                    text=result.text,
                    device=result.device,
                    status=f"{result.status}; retried on CPU after MPS error: {mps_error}",
                )
            raise

    def _transcribe_file_once(self, audio_path: Path, source_locale: str, chunk_ms: int) -> ASRResult:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        self._load()
        assert self._model is not None
        assert self._torch is not None
        assert self._streaming_buffer_cls is not None

        self._configure_model(source_locale, chunk_ms)
        streaming_buffer = self._streaming_buffer_cls(
            model=self._model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        streaming_buffer.append_audio_file(str(audio_path), stream_id=-1)
        text = self._perform_streaming(streaming_buffer)
        return ASRResult(text=text, device=self.device_name, status=f"Nemotron ASR completed on {self.device_name}")

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                import nemo.collections.asr as nemo_asr
                from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            except Exception as exc:
                raise RuntimeError(
                    "Nemotron ASR requires NVIDIA NeMo and PyTorch. Follow README setup, including the NeMo "
                    "install from GitHub."
                ) from exc

            device = self._select_device(torch)
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            model = model.to(device=device, dtype=torch.float32)
            model.eval()

            self._torch = torch
            self._streaming_buffer_cls = CacheAwareStreamingAudioBuffer
            self._device = device
            self._model = model

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
            self._model = None
            self._torch = None
            self._streaming_buffer_cls = None
            self._device = None
            self._forced_device = force_device

    def _configure_model(self, source_locale: str, chunk_ms: int) -> None:
        assert self._model is not None
        context = att_context_for_chunk_ms(chunk_ms)
        if hasattr(self._model.encoder, "set_default_att_context_size"):
            self._model.encoder.set_default_att_context_size(att_context_size=context)
        else:
            raise RuntimeError("The loaded ASR model does not support configurable cache-aware lookahead.")

        if not hasattr(self._model, "set_inference_prompt"):
            raise RuntimeError("The loaded ASR model does not expose Nemotron language-ID prompt conditioning.")

        prompt = source_locale if source_locale != "auto" else "auto"
        self._model.set_inference_prompt(prompt)
        decoding = getattr(self._model, "decoding", None)
        if decoding is not None and hasattr(decoding, "set_strip_lang_tags"):
            decoding.set_strip_lang_tags(False)

    def _perform_streaming(self, streaming_buffer) -> str:
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        torch = self._torch
        batch_size = len(streaming_buffer.streams_length)
        cache_last_channel, cache_last_time, cache_last_channel_len = self._model.encoder.get_initial_cache_state(
            batch_size=batch_size
        )
        previous_hypotheses = None
        pred_out_stream = None
        transcribed_texts = None

        for step_num, (chunk_audio, chunk_lengths) in enumerate(iter(streaming_buffer)):
            with torch.inference_mode():
                chunk_audio = chunk_audio.to(self._device).to(torch.float32)
                if hasattr(chunk_lengths, "to"):
                    chunk_lengths = chunk_lengths.to(self._device)
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=self._drop_extra_pre_encoded(step_num),
                    return_transcription=True,
                )

        if transcribed_texts is None:
            raise RuntimeError("Nemotron ASR returned no transcription.")
        return _extract_transcriptions(transcribed_texts)[0]

    def _drop_extra_pre_encoded(self, step_num: int) -> int:
        assert self._model is not None
        if step_num == 0:
            return 0
        return int(getattr(self._model.encoder.streaming_cfg, "drop_extra_pre_encoded", 0))


def _extract_transcriptions(hypotheses) -> list[str]:
    output: list[str] = []
    for item in hypotheses:
        output.append(getattr(item, "text", item))
    return [str(item) for item in output]


def _brief_error(exc: BaseException, limit: int = 160) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")

