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


def has_processed(session_state: dict, audio_bytes: bytes) -> bool:
    """Guards against re-processing the same sticky st.audio_input value on an unrelated rerun."""
    current_hash = hashlib.sha256(audio_bytes).hexdigest()
    return session_state.get(PROCESSED_HASH_KEY) == current_hash


def mark_processed(session_state: dict, audio_bytes: bytes) -> None:
    session_state[PROCESSED_HASH_KEY] = hashlib.sha256(audio_bytes).hexdigest()
