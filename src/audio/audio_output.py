import pygame
from gtts import gTTS
from io import BytesIO


def speak_text(text: str) -> None:
    """
    Speaks the given text using the default audio output device.

    Args:
        text (str): The text to be spoken.

    Returns:
        None
    """
    pygame.mixer.init()

    try:
        tts = gTTS(text=text, lang="en")
        bytes = BytesIO()
        tts.write_to_fp(bytes)
        bytes.seek(0)
        pygame.mixer.music.load(bytes)
        pygame.mixer.music.play()

        # Wait for the speech to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except:
        pass

    return None
