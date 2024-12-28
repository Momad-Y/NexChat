import speech_recognition as sr
import chime

DURATION = 5

def get_audio_input(duration: int = DURATION) -> str:
    """
    Function to get audio input from the user using the microphone.

    Args:
        duration (int): The duration in seconds for which the audio input should be recorded. Default is 5 seconds.

    Returns:
        str: The text recognized from the audio input, or None if the audio input could not be recognized.
    """
    chime.theme("chime")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        chime.info()
        audio = recognizer.record(source, duration=DURATION)

        try:
            text = recognizer.recognize_google(audio)
            chime.success()
            return text
        except sr.UnknownValueError:
            chime.warning()
            return None
        except sr.RequestError:
            chime.warning()
            return None
