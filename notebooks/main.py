import os
import json
import sounddevice as sd
import wave
from groq import Groq
import dotenv
import numpy as np

dotenv.load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ----------- RECORD SPEECH FROM MICROPHONE -----------
def record_audio(filename="speech.wav", duration=5, samplerate=16000):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete, saved to", filename)

    # Save as WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return filename


# ----------- TRANSCRIBE SPEECH TO TEXT -----------
def transcribe_audio(filename="speech.wav"):
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            language="en"
        )
    return transcription.text


# ----------- ASK QUESTION TO LLM -----------
def ask_question(question):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant, answer briefly."},
            {"role": "user", "content": question}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content


# ----------- TEXT TO SPEECH -----------
def speak_answer(text, filename="output.wav"):
    response = client.audio.speech.create(
        model="playai-tts",
        voice="Fritz-PlayAI",
        input=text,
        response_format="wav"
    )
    response.write_to_file(filename)
    print("🔊 Answer saved to", filename)


# ----------- MAIN FLOW -----------
if __name__ == "__main__":
    # Step 1: Record from mic
    audio_file = record_audio(duration=5)

    # Step 2: Transcribe speech
    question = transcribe_audio(audio_file)
    print("📝 You asked:", question)

    # Step 3: Get LLM answer
    answer = ask_question(question)
    print("🤖 Answer:", answer)

    # Step 4: Speak answer
    speak_answer(answer)
