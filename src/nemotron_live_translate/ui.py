"""Gradio user interface."""

from __future__ import annotations

import os

from .audio import ATT_CONTEXT_BY_CHUNK_MS
from .locales import gradio_locale_choices
from .pipeline import TranslationPipeline, get_default_pipeline


def build_demo(pipeline: TranslationPipeline | None = None):
    import gradio as gr

    active_pipeline = pipeline

    def run_translate(audio, source_locale, chunk_ms):
        nonlocal active_pipeline
        if active_pipeline is None:
            active_pipeline = get_default_pipeline()
        result = active_pipeline.translate_audio(audio, source_locale=source_locale, chunk_ms=int(chunk_ms))
        return (
            result.status,
            result.detected_locale,
            result.source_text,
            result.english_text,
            result.audio_path,
        )

    chunk_choices = [(f"{chunk_ms} ms", str(chunk_ms)) for chunk_ms in ATT_CONTEXT_BY_CHUNK_MS]

    with gr.Blocks(title="Nemotron 3.5 Speech-to-English") as demo:
        gr.Markdown("# Nemotron 3.5 Speech-to-English")
        with gr.Row():
            source_locale = gr.Dropdown(
                choices=gradio_locale_choices(),
                value="auto",
                label="Source",
                filterable=True,
            )
            chunk_ms = gr.Radio(
                choices=chunk_choices,
                value="320",
                label="Latency",
            )

        speech = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Speech",
            format="wav",
        )
        with gr.Row():
            translate = gr.Button("Translate", variant="primary")
            clear = gr.ClearButton()

        status = gr.Textbox(label="Status", lines=3)
        detected = gr.Textbox(label="Detected Locale", lines=1)
        with gr.Row():
            source_text = gr.Textbox(label="Transcript", lines=5)
            english_text = gr.Textbox(label="English", lines=5)
        spoken = gr.Audio(label="Spoken English", type="filepath", autoplay=True)

        translate.click(
            fn=run_translate,
            inputs=[speech, source_locale, chunk_ms],
            outputs=[status, detected, source_text, english_text, spoken],
            api_name="translate",
            concurrency_limit=1,
        )
        clear.add([speech, status, detected, source_text, english_text, spoken])

    return demo.queue(default_concurrency_limit=1)


def main() -> None:
    demo = build_demo()
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.launch(server_name=server_name, server_port=server_port, share=share, show_error=True)
