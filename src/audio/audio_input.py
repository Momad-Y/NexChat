import hashlib
import io

import speech_recognition as sr

PROCESSED_HASH_KEY = "audio_input_last_processed_hash"


def get_audio_input(audio_bytes: bytes) -> str | None:
    """Recognizes speech from an in-memory WAV blob (browser-captured, no PyAudio)."""
    if not audio_bytes:
        return None

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
    except Exception:
        return None

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


def has_processed(session_state: dict, content: bytes, key: str = PROCESSED_HASH_KEY) -> bool:
    """Guards against re-processing the same sticky value on an unrelated rerun.

    `key` lets independent call sites (different widgets, or non-audio content
    like summarization text/files) use distinct session_state slots instead of
    sharing one global "last processed" marker.
    """
    current_hash = hashlib.sha256(content).hexdigest()
    return session_state.get(key) == current_hash


def mark_processed(session_state: dict, content: bytes, key: str = PROCESSED_HASH_KEY) -> None:
    session_state[key] = hashlib.sha256(content).hexdigest()
