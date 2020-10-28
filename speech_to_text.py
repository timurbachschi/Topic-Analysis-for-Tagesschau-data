"""Module for the Speech2Text part of the project"""
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr


def speech_to_text(path):
    """
    Convert a video to text via Speech2Text software

            Parameters:
                    path (str): Path of the episode mp3 that is to be transcripted.

            Returns:
                    None
    """
    recognizer = sr.Recognizer()
    sound_file = AudioSegment.from_mp3(path)
    audio_chunks = split_on_silence(sound_file, min_silence_len=300, silence_thresh=-44)
    Path("episodes").mkdir(parents=True, exist_ok=True)
    with open("test.txt", "w") as f:
        for i, chunk in enumerate(audio_chunks):
            try:
                chunk.export("chunks/chunk_{}.wav".format(i), format="wav")
                with sr.AudioFile("chunks/chunk_{}.wav".format(i)) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio, language="de-DE") + ". "
                    f.write(text)
            except sr.UnknownValueError:
                continue

#speech_to_text("episodes/tagesschau_14102020_2000_.mp3")
