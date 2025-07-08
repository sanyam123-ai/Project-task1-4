import speech_recognition as sr

r = sr.Recognizer()

while True:
    try:
        with sr.Microphone() as source:
            print("say something good")
            audio = r.listen(source)
            text = r.recognize_google(audio)
            text = text.lower()

            print(f"recognized text : {text}")

    except:
        print("you were trying to be funny")
        r = sr.recognizer()
        continue        