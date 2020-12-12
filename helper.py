import os
from pathlib import Path
import pandas as pd
import spacy
from nltk.corpus import stopwords
from asrtoolkit import wer

def cleanup_textfile(path_to_file):
    txt_list = []
    with open(path_to_file, 'r') as in_file:
        for line in in_file:
            line = line.lower()
            line = line.replace(".", "")
            line = line.replace("  ", " ")
            line = line.replace("\n", "")
            txt_list.append(line)
    return txt_list

def calculate_wer(file1, file2):
    # 1. Remove dots and spaces
    file1_str = " ".join(cleanup_textfile(file1)).replace("  ", " ")
    file2_str = " ".join(cleanup_textfile(file2)).replace("  ", " ")
    return wer(file1_str, file2_str)

def calculate_wer_for_testset(path_to_transcripts, path_to_testset):
    print("Calculate WER for all files in testset...")
    wer_list = []
    for filename in os.listdir(path_to_testset):
        # Find corresponding transcript
        # TODO: This will not work for non-tagesschau and/or if filename format is changed!
        transcript_name = "tagesschau_{}_.txt.txt".format(filename.replace(".txt", ""))
        path1 = "{}/{}".format(path_to_testset, filename)
        path2 = "{}/{}".format(path_to_transcripts, transcript_name)
        # TODO: Very hacky, change that by changing file names!
        try:
            wer = calculate_wer(path1, path2)
        except FileNotFoundError:
            path2 = path2.replace(".txt.txt", ".txt")
            wer = calculate_wer(path1, path2)
        wer_list.append(wer)
        print("File name: {} | WER: {}".format(filename, wer))
    print("Average WER: {}".format(sum(wer_list)/len(wer_list)))


def move_episodes_to_dir_by_year(dir_eps, year):
    # Moves episodes from a given year in a given directory to a seperate dir
    Path("{}/{}".format(dir_eps,year)).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(dir_eps):
        if "{}_".format(year) in filename:
            os.rename("{}/{}".format(dir_eps, filename), "{}/{}/{}".format(dir_eps, year, filename))

def transcripts_to_df(path_to_transcripts, save_df=False):
    # Create dataframe out of transcripts
    df = pd.DataFrame(columns=['transcriptionName','content','year','month','day'])
    df = df.fillna(0)
    for transcription in os.listdir(path_to_transcripts):
        with open("{}/{}".format(path_to_transcripts, transcription), "r") as f:
            df.loc[transcription,'transcriptionName'] = str(transcription)
            df.loc[transcription,'content'] = f.read()
            date_string = str(transcription).split("_")[1]
            df.loc[transcription, 'year'] = date_string[4:]
            df.loc[transcription, 'month'] = date_string[2:4]
            df.loc[transcription, 'day'] = date_string[0:2]
    df = df.reset_index(drop=True)
    if save_df:
        df.to_csv("episodes.csv", index=False)
    return df

def preprocess_df(df):
    nlp = spacy.load('de_core_news_sm')
    non_nouns=[] # init array for all non-nouns found by spacy

    # preprocessing for topic modelling, in this case LDA (latent dirichlet analysis)
    for transcript in range(len(df)):
        # remove "newline" and punctuation
        df.loc[transcript,'content'] = df.loc[transcript,'content'].replace("\n","").replace(".","")
        # create token for spacy
        doc = nlp(df.loc[transcript,'content'])
        # lemmatize with help of spacy (maybe we can try another library for this)
        df.loc[transcript,'content'] = ((",".join([x.lemma_ for x in doc])).replace(","," "))
        doc = nlp(df.loc[transcript,'content']) 
        # write everything lowercase
        df.loc[transcript,'content'] = df.loc[transcript,'content'].lower()
        df.at[transcript,'content'] = df.loc[transcript,'content'].split()
        # stopword removal
        stop_words=set(stopwords.words('german'))
        df.at[transcript,'content'] = [word for word in df.at[transcript,'content'] if word not in stop_words]
        # compensate structure for splitting
        df.at[transcript,'content'] = ','.join(df.loc[transcript,'content'])
        df.at[transcript,'content'] = (df.loc[transcript,'content']).replace(','," ")
    # convert to matching format
    for transcript in range(len(df)):
        df.at[transcript,'content'] = df.loc[transcript,'content'].split()