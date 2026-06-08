"""Pre-download and optionally load the demo models before presenting the app."""

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_live_translate.asr import NEMOTRON_MODEL_ID, NemotronASR
from nemotron_live_translate.translation import NLLB_MODEL_ID, NLLBTranslator


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up Nemotron ASR and NLLB translation assets.")
    parser.add_argument("--download-only", action="store_true", help="Download model files without instantiating models.")
    parser.add_argument("--skip-nllb", action="store_true", help="Skip NLLB warmup.")
    parser.add_argument("--skip-nemotron", action="store_true", help="Skip Nemotron warmup.")
    args = parser.parse_args()

    if args.download_only:
        from huggingface_hub import snapshot_download

        if not args.skip_nemotron:
            print(f"Downloading {NEMOTRON_MODEL_ID}...")
            print(snapshot_download(repo_id=NEMOTRON_MODEL_ID))
        if not args.skip_nllb:
            print(f"Downloading {NLLB_MODEL_ID}...")
            print(snapshot_download(repo_id=NLLB_MODEL_ID))
        print("Download warmup complete.")
        return

    if not args.skip_nemotron:
        print(f"Loading {NEMOTRON_MODEL_ID} on CPU to verify ASR runtime...")
        asr = NemotronASR(prefer_mps=False)
        asr._load()
        print(f"Nemotron loaded on {asr.device_name}.")

    if not args.skip_nllb:
        print(f"Loading {NLLB_MODEL_ID} to verify translation runtime...")
        translator = NLLBTranslator(prefer_mps=False)
        translator._load()
        print(f"NLLB loaded on {translator.device_name}.")

    print("Model warmup complete.")


if __name__ == "__main__":
    main()

