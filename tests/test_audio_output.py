def test_speak_text_returns_bytes(monkeypatch):
    from audio import audio_output as audio_output_module

    class FakeTTS:
        def __init__(self, text, lang):
            self.text = text

        def write_to_fp(self, fp):
            fp.write(b"fake-mp3-bytes")

    monkeypatch.setattr(audio_output_module, "gTTS", FakeTTS)

    result = audio_output_module.speak_text("hello world")

    assert result == b"fake-mp3-bytes"


def test_speak_text_returns_empty_bytes_for_empty_input():
    from audio.audio_output import speak_text

    assert speak_text("") == b""


def test_speak_text_returns_empty_bytes_on_tts_failure(monkeypatch):
    from audio import audio_output as audio_output_module

    class FailingTTS:
        def __init__(self, text, lang):
            raise ValueError("tts service unavailable")

    monkeypatch.setattr(audio_output_module, "gTTS", FailingTTS)

    result = audio_output_module.speak_text("hello world")

    assert result == b""
