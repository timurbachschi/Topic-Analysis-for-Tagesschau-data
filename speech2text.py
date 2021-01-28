"""Module for the Speech2Text part of the project"""
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from timeit import default_timer as timer
import scipy.io.wavfile as wav
import soundfile as sf
import shlex
import re
import datetime
import subprocess
from get_video import transform_episode, download_videos_by_date
import sys
import wave
import ast
import pandas as pd
import os, shutil


def speech_to_text(path):
    """
    Convert a video to text via Google Speech Recognition

            Parameters:
                    path (str): Path of the episode mp3/flac that is to be transcripted.

            Returns:
                    None
    """
    # Make sure all directories for data storage are created
    Path("chunks").mkdir(parents=True, exist_ok=True)
    Path("transcripts").mkdir(parents=True, exist_ok=True)
    # Get transcript file name
    data_format = (path.split('.')[1])
    print("data format is:", data_format)
    transcript_filename = os.path.basename(path).split(".")[0]+".txt"
    print("path is:", "transcripts/{}".format(transcript_filename))
    # Check whether transcript of that episode was already made (based on name only)
    if transcript_filename not in os.listdir("transcripts/"):
        recognizer = sr.Recognizer()
        sound_file = AudioSegment.from_file(path)
        # Divide file into chunks (Google SR only works for <1min input) based on pauses.
        # This might not always work out as intended.
        audio_chunks = split_on_silence(
            sound_file, min_silence_len=390, silence_thresh=-39)
        with open("transcripts/{}.txt".format(transcript_filename), "w") as f:
            for i, chunk in enumerate(audio_chunks):
                # Try to transcribe readable chunks
                try:
                    chunk_path = "chunks/chunk_{}.{}".format(i, data_format)
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
                            text = transcribe_long_chunk(
                                chunk_path, duration_seconds, data_format, recognizer)
                            f.write(text)
                        else:
                            audio = recognizer.record(source)
                            text = recognizer.recognize_google(
                                audio, language="de-DE") + ". "
                            print(text)
                            f.write(str(text)+"\n")
                except sr.UnknownValueError:
                    print("Chunk not readable: {}".format(i))

    else:
        print("transcription already exists in folder:", path)


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


def transcribe_long_chunk(chunk_path, chunk_duration, data_format, recognizer):
    # Gets called if a too long chunk needs to be transcribed
    seg = AudioSegment.from_file(chunk_path)
    split_chunks = split_chunk(seg, chunk_duration)
    for i, new_chunk in enumerate(split_chunks):
        new_chunk_path = "split_chunk{}.{}".format(i, data_format)
        new_chunk.export(new_chunk_path, format=data_format)
        with sr.AudioFile(new_chunk_path) as split_source:
            audio = recognizer.record(split_source)
            text = recognizer.recognize_google(audio, language="de-DE") + ". "
            return text


def transcribe_from_daterange(start_date, end_date, vosk=False, download=False):
    """
    Transcribe all episodes found in episodes_mp4 dir for a range of dates.

            Parameters:
                    start_date (str): Start date (format "YYYYMMDD")
                    end_date (str): End date (format "YYYYMMDD")
                    vosk (bool): If True use VOSK instead of Google SR
                    download (bool): If True, download each individual episode.
                                    If False it is assumed the episodes are already downloaded

            Returns:
                    None
    """
    # Transcribe episodes from range of dates (format "YYYYMMDD"). Use vosk if vosk=True.
    daterange = pd.date_range(start_date, end_date)
    filenames = []
    for date in daterange:
        if download:
            # Download episodes as mp4
            folder = 'episodes_mp4'
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            filenames = download_videos_by_date(date.strftime("%Y%m%d"))
        # Transcribe (downloaded) episodes
        for filename in os.listdir("episodes_mp4"):
            # Ignore "Tagesschau vor 20 Jahren"
            if date.strftime("%d%m%Y") in filename and "tagesschau" in filename and "Jahren" not in filename and "mit" not in filename:
                print("Transcribe: {}".format(filename))
                path_to_sound = "episodes_mp4/{}".format(
                    filename.replace("mp4", "wav"))
                transform_episode(
                    "episodes_mp4/{}".format(filename), path_to_sound)
                if vosk == True:
                    speech_to_text_vosk(path_to_sound)
                else:
                    speech_to_text(path_to_sound)


def speech_to_text_vosk(mp4_path):
    # Transcription using VOSK 
    # https://github.com/alphacep/vosk
    SetLogLevel(0)
    if not os.path.exists("model"):
        print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
        exit(1)
    sample_rate = 16000
    model = Model("model")
    rec = KaldiRecognizer(model, sample_rate)
    # Transform episode from mp4 to WAV
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                mp4_path,
                                '-ar', str(sample_rate), '-ac', '1', '-f', 's16le', '-'],
                               stdout=subprocess.PIPE)
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
    # Save transcription
    path_to_txt = "transcripts_vosk/{}".format(
        mp4_path.replace("episodes_mp4/", "").replace("wav", "txt"))
    with open(path_to_txt, "w") as f:
        f.write(ast.literal_eval(rec.FinalResult())["text"])
