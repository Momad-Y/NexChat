import speech_recognition as sr


def get_audio_input() -> str:
    """
    Function to get audio input from the user using the microphone.

    Args:
        None

    Returns:
        str: The text recognized from the audio input, or None if the audio input could not be recognized.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
