"""Module for the Speech2Text part of the project"""
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from timeit import default_timer as timer


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
    # Divide mp3 into chunks (speech2text only works for <1min input) based on pauses.
    # This might not always work out as intended.
    audio_chunks = split_on_silence(sound_file, min_silence_len=300, silence_thresh=-44)
    Path("chunks").mkdir(parents=True, exist_ok=True)
    Path("transcripts").mkdir(parents=True, exist_ok=True)
    with open("transcripts/test.txt", "w") as f:
        for i, chunk in enumerate(audio_chunks):
            try:
                # TODO: Is there a way to do this without saving the chunks?
                chunk.export("chunks/chunk_{}.wav".format(i), format="wav")
                with sr.AudioFile("chunks/chunk_{}.wav".format(i)) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio, language="de-DE") + ". "
                    f.write(text)
            except sr.UnknownValueError:
                continue
            