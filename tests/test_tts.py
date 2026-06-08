import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nemotron_live_translate.tts import MacSayTTS


class TTSTests(unittest.TestCase):
    def test_say_and_afconvert_are_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if "afconvert" in cmd[0]:
                    Path(cmd[2]).write_bytes(b"RIFF")
                return mock.Mock(returncode=0)

            with mock.patch("platform.system", return_value="Darwin"), mock.patch(
                "shutil.which", side_effect=lambda name: f"/usr/bin/{name}"
            ), mock.patch("subprocess.run", side_effect=fake_run):
                result = MacSayTTS(output_dir=tmp).speak("Hello.")

            self.assertTrue(result.audio_path.endswith(".wav"))
            self.assertEqual(Path(result.audio_path).read_bytes(), b"RIFF")
            self.assertEqual(calls[0][0], "/usr/bin/say")
            self.assertEqual(calls[1][0], "/usr/bin/afconvert")


if __name__ == "__main__":
    unittest.main()

