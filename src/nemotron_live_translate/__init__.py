"""Local Nemotron 3.5 speech-to-English translator."""

from .pipeline import TranslationPipeline, get_default_pipeline, translate_audio
from .schemas import TranslationResult

__all__ = [
    "TranslationPipeline",
    "TranslationResult",
    "get_default_pipeline",
    "translate_audio",
]

