import io


def test_has_processed_false_before_marking():
    from audio.audio_input import has_processed

    session_state = {}
    assert has_processed(session_state, b"some-wav-bytes") is False


def test_mark_processed_then_has_processed_true():
    from audio.audio_input import has_processed, mark_processed

    session_state = {}
    mark_processed(session_state, b"some-wav-bytes")
    assert has_processed(session_state, b"some-wav-bytes") is True


def test_has_processed_false_for_different_bytes():
    from audio.audio_input import has_processed, mark_processed

    session_state = {}
    mark_processed(session_state, b"first-recording")
    assert has_processed(session_state, b"second-recording") is False


def test_get_audio_input_returns_none_on_malformed_bytes():
    from audio.audio_input import get_audio_input

    assert get_audio_input(b"not a real wav file") is None


def test_get_audio_input_returns_none_on_empty_bytes():
    from audio.audio_input import get_audio_input

    assert get_audio_input(b"") is None


def test_get_audio_input_returns_recognized_text(monkeypatch):
    from audio import audio_input as audio_input_module

    class FakeRecognizer:
        def record(self, source):
            return "fake-audio-data"

        def recognize_google(self, audio):
            return "hello world"

    class FakeAudioFile:
        def __init__(self, source):
            pass

        def __enter__(self):
            return "fake-source"

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(audio_input_module.sr, "Recognizer", FakeRecognizer)
    monkeypatch.setattr(audio_input_module.sr, "AudioFile", FakeAudioFile)

    result = audio_input_module.get_audio_input(b"RIFF....WAVEfmt ")

    assert result == "hello world"
