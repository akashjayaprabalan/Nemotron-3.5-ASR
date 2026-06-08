import tempfile
import unittest
import wave

from nemotron_live_translate.audio import (
    TARGET_SAMPLE_RATE,
    att_context_for_chunk_ms,
    chunk_frames_for_chunk_ms,
    coerce_audio_to_wav,
    normalize_samples,
    resample_linear,
)


class AudioTests(unittest.TestCase):
    def test_chunk_context_mapping(self):
        self.assertEqual(att_context_for_chunk_ms(80), [56, 0])
        self.assertEqual(att_context_for_chunk_ms("320"), [56, 3])
        self.assertEqual(chunk_frames_for_chunk_ms(1120), 14)
        with self.assertRaises(ValueError):
            att_context_for_chunk_ms(240)

    def test_mono_normalization_from_stereo(self):
        samples = [(0.5, -0.5), (1.0, 0.0), (-1.0, -1.0)]
        self.assertEqual(normalize_samples(samples), [0.0, 0.5, -1.0])

    def test_resample_linear_length(self):
        samples = [0.0, 0.5, 1.0, 0.5]
        out = resample_linear(samples, source_rate=4, target_rate=8)
        self.assertEqual(len(out), 8)
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[-1], 0.5)

    def test_coerce_tuple_to_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = coerce_audio_to_wav((8000, [0.0, 0.25, -0.25, 0.0]), tmp)
            with wave.open(str(wav_path), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getframerate(), TARGET_SAMPLE_RATE)
                self.assertEqual(wav.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()

