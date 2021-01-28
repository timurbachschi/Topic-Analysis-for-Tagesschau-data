import os
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA
import warnings
import time
import datetime
warnings.simplefilter("ignore", DeprecationWarning)
import matplotlib.pyplot as plt
import spacy
import de_core_news_lg
from gensim import corpora,models
import gensim


nlp = de_core_news_lg.load()


# load the s2t transcriptions
def get_transcription_paths(path):
    transcriptions = os.listdir(path)
    return ["{}/{}".format(path, t) for t in transcriptions]


# Preprocessing for topic modelling, in this case LDA
# use SpaCy Pipe for faster lemmatizing
def preprocess_pipe(texts):
    n_cpu_cores = int(os.cpu_count()/2-1)
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=int(len(texts)/n_cpu_cores),n_process=int(n_cpu_cores)):
        preproc_pipe.append([str(word.lemma_).lower() for word in doc if not word.is_stop and word.is_alpha])
    return preproc_pipe


# preprocess transcriptions: remove: "newline", punctuations, stopwords (automatically and manually chosen ones
# create bi-/tri-grams and add meta-data date.
def preprocess_transcriptions(transcriptions, number_cpu_cores=os.cpu_count()-1):
    # save transcriptions of tagesschau in DataFrame
    df = pd.DataFrame(index=[i for i in range(len(transcriptions)) if "tagesschau" in transcriptions[i]], columns=['transcriptionName','content','preprocessed','year','month','day'])
    for transcription in range(len(transcriptions)):
        if "tagesschau" in str(transcriptions[transcription]) and 'vor' not in str(transcriptions[transcription]):
            with open(transcriptions[transcription], "r") as f:
                df.at[transcription,'transcriptionName'] = str(transcriptions[transcription])
                df.at[transcription,'content'] = f.read()
    df = df.reset_index()
    
    # remove "newline" and punctuation
    for transcript in range(len(df)):
        df.at[transcript,'preprocessed'] = df.loc[transcript,'content'].replace("\n","").replace(".","")
    start_timer_preprocess = datetime.datetime.now()
    df['preprocessed'] = preprocess_pipe(df['preprocessed'])
    
    print("lemmatizing needs:",round((datetime.datetime.now()-start_timer_preprocess).total_seconds(),2),"seconds")

    # Create substitutions for a better overview and easier usage/changing of manual stopwords.
    # Furthermore, useable for correcting of text2speech (korona->corona)
    manual = [[['wetter'],['sonne','regen','wind','schnee','schauer','luft','wolken','gewitter','gewittern']],
              [['himmelsrichtung'],['norden','sueden','osten','westen']],
              [['wochentag'],['montag','dienstag','mittwoch','donnerstag','freitag','samstag','sonntag']],
              [['monat'],['januar','februar','maerz','april','mai','juni','juli','august','september','oktober','november','dezember']],
              [['corona'],['korona']]]
    
    start_manual = datetime.datetime.now()
    
    # Apply above manual substitutions
    for i in range(len(df)):
        for k in manual:
            for l in range(len(k[1])):
                df.at[i,'preprocessed'] = [k[0][0] if k[1][l]==x else x for x in df.loc[i,'preprocessed']]
    
    print("replacing e.g. 'sonne','regen'... by 'wetter' needs:",
          round((datetime.datetime.now()-start_manual).total_seconds(),2),"seconds")    
    
    # As we use automatic stopword removal (+ filter_extreme method later), we only remove weather, cardinal direction, 
    # weekday and month
    manual_stopwords = ['wetter','wochentag','monat','himmelsrichtung']
    
    # Remove manual chosen stopwords.
    for i in range(len(df)):
        df.at[i,'preprocessed'] = [word for word in df.loc[i,'preprocessed'] if word not in manual_stopwords]
        
    # Create bi-/tri-grams for words which occur together multiple times.
    start_bi_trigram = datetime.datetime.now()
    list_of_all_for_bigram = []
    for i in range(len(df)):
        list_of_all_for_bigram.append(df.loc[i,'preprocessed'])
    bigram = models.Phrases(list_of_all_for_bigram, min_count=20, threshold=50)
    trigram = models.Phrases(bigram[list_of_all_for_bigram], threshold=50)
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    for i in range(len(df)):
        df.at[i,'preprocessed'] = bigram_mod[df.loc[i,'preprocessed']]
        df.at[i,'preprocessed'] = trigram_mod[bigram_mod[df.loc[i,'preprocessed']]]
    for i in range(len(df)):
        df.at[i,'preprocessed'] = (','.join(df.loc[i,'preprocessed'])).replace(','," ")
    
    print("Bi-/Tri-Grams needs:",round((datetime.datetime.now()-start_bi_trigram).total_seconds(),2),"seconds")
    
    
    # save metadata of transcriptions in DataFrame
    for i in range(len(df)):
        if 'tagesschau' in df.loc[i,'transcriptionName'] and 'vor' not in df.loc[i,'transcriptionName']:
            df.at[i,'year'] = int((df.loc[i,'transcriptionName'].split('_')[1][4:8]))
            df.at[i,'month'] = int((df.loc[i,'transcriptionName'].split('_')[1][2:4]))
            df.at[i,'day'] = int((df.loc[i,'transcriptionName'].split('_')[1][0:2]))
    
    return df


def train_lda_compute_coherence(df, corpus, texts, dictionary, n_topics, workers, no_below=20, no_above=0.5):
    global lda_models
    passes = 500
    iterations = 500
    # train LDA
    # workers: recommended is using number of pyhsical CPU cores - 1
    # for us, less cores worked better
    # minimum_probability=0 for proper usage in topic modeling over time, as you keep all probabilities.
    # chunksize with best performance when using #transcripts/workers (chunksize=transcripts, if there is enough RAM)
    # at least 500 passes and iterations to have high probability of convergence, as we have less data (~5000 transcriptions)
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=n_topics,
                                           random_state=10,
                                           chunksize=int(len(df['preprocessed'])/workers),
                                           passes=passes, 
                                           iterations=iterations,
                                           per_word_topics=True,
                                           eval_every=50, # useable for logging 
                                           workers=workers,
                                           minimum_probability=0)
                                           #callbacks=[coherence_logger]) # not supported for multicore-method
    
    
    Path(my_path+"lda/lda_"+str(n_topics)+"topics_"+str(passes)+"passes_"+str(iterations)+"iter"+str(no_below)+"no_below"+str(no_above)+"no_above").mkdir(parents=True, exist_ok=True)
    lda_model.save(my_path+"lda/lda_"+str(n_topics)+"topics_"+str(passes)+"passes_"+str(iterations)+"iter"+str(no_below)+"no_below"+str(no_above)+"no_above"+"/lda")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['preprocessed'], dictionary=id2word, coherence='c_v')
    #lda_models.append([n_topics,lda_model,coherence_model_lda])
    return coherence_model_lda.get_coherence()


def get_preprocessed_df(path_to_transcriptions):
    # set Path
    n_cpu_cores = int(os.cpu_count()/2-1)
    print("number cpu_cores to use:",n_cpu_cores)
    df_all_processed = pd.DataFrame(index=[], columns=['preprocessed'])


    #load speech2text transcriptions
    transcriptions = get_transcription_paths(path_to_transcriptions)
    keys = ['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
    years_dict = {}
    for year in keys: 
        years_dict[year] = []

    # prepare daterange of interest for lda
    daterange = pd.date_range(datetime.datetime(2006, 12, 19), datetime.datetime(2021, 11, 11))
    dates_of_interest = []
    for date in daterange:
        for i in transcriptions:
            if 'tagesschau' in i and 'vor' not in i:
                year = int((i.split('_')[1][4:8]))
                if year==date.year:
                    month = int((i.split('_')[1][2:4]))
                    if month == date.month:
                        day = int((i.split('_')[1][0:2]))
                        if day == date.day:
                            years_dict[str(year)].append(i)
                            dates_of_interest.append(i)
    print("There are ",len(dates_of_interest),"transcriptions in given daterange")

    yearsLen = []
    # preprocess every year
    for year in years_dict.keys():
        print("\npreprocess transcriptions of year:",year,", quantity:",len(years_dict[str(year)]))
        if len(years_dict[str(year)]) > 0:
            yearsLen.append(len(years_dict[str(year)]))
        if len(years_dict[str(year)]) == 0:
            print("no transcription for year:",year)
            continue
        start = datetime.datetime.now()
        transcriptions = years_dict[str(year)]
        df_processed = preprocess_transcriptions(transcriptions,number_cpu_cores=n_cpu_cores)
        df_all_processed = pd.concat([df_all_processed,df_processed],ignore_index=True)
    

          
    print("length of DataFrame:",len(df_all_processed))

    print("number of transcriptions of each year in chosen daterange:",yearsLen)

    # save lda
    df_all_processed.to_csv("saved_dataframes/df_preprocessed.csv", sep=',', index=False)

    print("\ndone")
    return df_all_processed