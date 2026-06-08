# Local Nemotron 3.5 Speech-to-English Translator

A local Gradio app that records speech, transcribes it with NVIDIA Nemotron 3.5 ASR, translates the recognized text into English with NLLB-200, and speaks the English result back through macOS `say`.

This project intentionally uses **Nemotron ASR only**. There is no Whisper or lightweight ASR fallback. On this Apple Silicon Mac, ASR defaults to CPU for demo reliability because NeMo cache-aware streaming is designed around NVIDIA/CUDA. Translation can still use the local PyTorch runtime.

## Models

- ASR: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- Translation: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
- TTS: local macOS `say` plus `afconvert`

## Setup

Use Python 3.11. The system Python 3.14 on this machine is too new for a comfortable NeMo/PyTorch environment.

```bash
cd /Users/akash/Desktop/Nemotron_3.5_ASR
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
brew install ffmpeg libsndfile
pip install -e ".[app]"
pip install Cython packaging
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
```

The first translation run downloads the Nemotron and NLLB model weights into the Hugging Face cache. That can take a while and uses several GB of disk.

Before a live demo, warm up the model cache so the first recorded clip does not sit in a long download step:

```bash
source .venv/bin/activate
python scripts/warmup.py --download-only
```

For a stricter runtime check, load both models once before launching the UI:

```bash
source .venv/bin/activate
python scripts/warmup.py
```

The UI also streams backend status updates during Translate: browser audio receipt, WAV conversion, Nemotron ASR, NLLB translation, and macOS speech generation.

## Run

```bash
source .venv/bin/activate
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

For a detached local server, use:

```bash
screen -dmS nemotron_translate zsh -lc 'cd /Users/akash/Desktop/Nemotron_3.5_ASR && GRADIO_SERVER_PORT=7860 .venv/bin/python scripts/serve.py'
```

Stop the detached server with:

```bash
screen -S nemotron_translate -X quit
```

## Tests

The unit tests avoid loading the heavyweight model stack.

```bash
PYTHONPATH=src python3.11 -m unittest discover -s tests
```

If Gradio is not installed, the UI smoke test is skipped.

## Supported Locales

The app exposes NVIDIA's 40 documented Nemotron source language-locales, plus Tamil (`ta-IN`) as a prompt-only experimental option because the released `.nemo` checkpoint includes that prompt key. NVIDIA's model card lists 32 transcription-ready or broad-coverage locales plus 8 adaptation-ready locales; Tamil is not part of those documented 40 language-locales. If forced `ta-IN` returns an empty transcript, use a fine-tuned Nemotron checkpoint for Tamil transcription.

All supported locales translate to English (`eng_Latn`). English source locales skip translation and are spoken back directly.

## License Notes

- Nemotron 3.5 ASR is governed by OpenMDW-1.1.
- NLLB-200 distilled 600M is released under CC-BY-NC-4.0.
- This repo does not commit or redistribute model weights.
