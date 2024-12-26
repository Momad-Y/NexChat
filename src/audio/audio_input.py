import speech_recognition as sr

def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            return "Sorry, my speech service is down."