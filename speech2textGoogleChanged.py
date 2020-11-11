"""Module for the Speech2Text part of the project"""
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from timeit import default_timer as timer
import scipy.io.wavfile as wav
import soundfile as sf
import os
import shlex
import re
import datetime
import subprocess
                            
def speech_to_text(path):
    """
    Convert a video to text via Speech2Text software

            Parameters:
                    path (str): Path of the episode mp3 that is to be transcripted.

            Returns:
                    None
    """
    data_format = (path.split('.')[1])
    print("data format is:",data_format)
    print("path is:",path.split("/")[1].split(".")[0]+".txt")
    if str(path.split("/")[1].split(".")[0]+".txt") not in os.listdir("transcripts/"):

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
                    chunk_path = "chunks/chunk_{}.{}".format(i,data_format)
                    chunk.export(chunk_path, format=data_format)
                    with sr.AudioFile(chunk_path) as source:
                        if data_format == 'wav':
                            (source_rate, source_sig) = wav.read(chunk_path)
                        elif data_format == 'flac':
                            (source_sig, source_rate) = sf.read(chunk_path)
                        duration_seconds = len(source_sig) / float(source_rate)
                        # Split the chunk into smaller chunks if it is too long
                        if duration_seconds > 60:
                            print("Chunk too long!")
                            seg = AudioSegment.from_file(chunk_path)
                            split_chunks = split_chunk(seg, duration_seconds)
                            for i, new_chunk in enumerate(split_chunks):
                                new_chunk_path = "split_chunk{}.{}".format(i,data_format)
                                new_chunk.export(new_chunk_path, format=data_format)
                                with sr.AudioFile(new_chunk_path) as split_source:
                                    audio = recognizer.record(split_source)
                                    text = recognizer.recognize_google(audio, language="de-DE") + ". "
                                    print(text)
                                    f.write(text)
                        else:
                            audio = recognizer.record(source)
                            text = recognizer.recognize_google(audio, language="de-DE") + ". "
                            print(text)
                            f.write(str(text)+"\n")
                except sr.UnknownValueError:
                    print("Chunk not readable: {}".format(i))
        
    else:
        print("transcription already exists in folder:",path)


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

