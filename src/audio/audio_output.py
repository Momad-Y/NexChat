from gtts import gTTS
from io import BytesIO


def speak_text(text: str) -> bytes:
    """Synthesizes speech and returns MP3 bytes for browser playback via st.audio."""
    if not text:
        return b""

    try:
        tts = gTTS(text=text, lang="en")
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        return buffer.getvalue()
    except Exception:
        return b""
