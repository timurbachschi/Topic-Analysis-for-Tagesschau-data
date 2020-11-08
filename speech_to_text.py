"""Module for the Speech2Text part of the project"""
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from timeit import default_timer as timer
import scipy.io.wavfile as wav


def speech_to_text(path):
    """
    Convert a video to text via Speech2Text software

            Parameters:
                    path (str): Path of the episode mp3 that is to be transcripted.

            Returns:
                    None
    """
    recognizer = sr.Recognizer()
    sound_file = AudioSegment.from_file(path)
    # Divide mp3 into chunks (speech2text only works for <1min input) based on pauses.
    # This might not always work out as intended.
    audio_chunks = split_on_silence(sound_file, min_silence_len=390, silence_thresh=-39) #390,-39
    Path("chunks").mkdir(parents=True, exist_ok=True)
    Path("transcripts").mkdir(parents=True, exist_ok=True)
    with open("transcripts/{}.txt".format(Path(path).name.replace(".mp3", "")).replace(".flac", ""), "w") as f:
        for i, chunk in enumerate(audio_chunks):
            try:
                # TODO: Is there a way to do this without saving the chunks?
                chunk_path = "chunks/chunk_{}.wav".format(i)
                chunk.export(chunk_path, format="wav")
                with sr.AudioFile(chunk_path) as source:
                    (source_rate, source_sig) = wav.read(chunk_path)
                    duration_seconds = len(source_sig) / float(source_rate)
                    # Split the chunk into smaller chunks if it is too long
                    if duration_seconds > 60:
                        print("Chunk too long!")
                        seg = AudioSegment.from_wav(chunk_path)
                        split_chunks = split_chunk(seg, duration_seconds)
                        for i, new_chunk in enumerate(split_chunks):
                            new_chunk.export('split_chunk{}.wav'.format(i), format="wav")
                            audio_file = sr.AudioFile('split_chunk{}.wav'.format(i))
                            audio = recognizer.record(audio_file)
                            text = recognizer.recognize_google(audio_file, language="de-DE") + ". "
                            f.write(text)
                    else:
                        audio = recognizer.record(source)
                        text = recognizer.recognize_google(audio, language="de-DE") + ". "
                        f.write(text)
            except sr.UnknownValueError:
                print("Chunk not readable: {}".format(i))


def split_chunk(chunk_audioseg, duration_sec):
    # Divide chunk until it is small enough
    if duration_sec > 60:
        new_chunks = [chunk_audioseg[0:duration_sec*1000/2],
                      chunk_audioseg[duration_sec*1000/2:duration_sec*1000]]
        res = []
        for chunk in new_chunks:
            res.extend(split_chunk(chunk, duration_sec/2))
        return res
    else:
        return [chunk_audioseg]
