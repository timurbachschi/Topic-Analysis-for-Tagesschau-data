# make dataframes from txt data
# each txt contains one tagesschau transcription
import os
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA
import warnings
import time
warnings.simplefilter("ignore", DeprecationWarning)
import matplotlib.pyplot as plt
import spacy
import de_core_news_lg
nlp = de_core_news_lg.load()

lemmatizing=True # with spacy; manually added words are definetly needed
stemming=False # lemmatizing is recommended instead
stopword_removal = False # didn't work as good as I expected

os.chdir("/home/sim/all/Master/Forschungspraktikum/Tagesschau/transcripts/")
transcriptions = os.listdir()
print("Transcriptions which are used now")
print(transcriptions)
# use following line instead of the line after the folowing to use "tagesschau" only
#df = pd.DataFrame(index=[i for i in range(len(transcriptions)) if "tagessschau" in transcriptions[i]], columns=['transcriptionName','content','year','month','day'])
df = pd.DataFrame(index=[i for i in range(len(transcriptions))], columns=['transcriptionName','content','year','month','day'])
df = df.fillna(0)
for transcription in range(len(transcriptions)):
    # use following line to use "tagesschau" only
    #if "tagesschau" in str(transcriptions[transcription]):
    if 1==1: #placeholder for above line
        with open(transcriptions[transcription], "r") as f:
            df.loc[transcription,'transcriptionName'] = str(transcriptions[transcription])
            df.loc[transcription,'content'] = f.read()
df = df.reset_index()

non_nouns=[] # init array for all non-nouns found by spacy

# preprocessing for topic modelling, in this case LDA (latent dirichlet analysis)
for transcript in range(len(df)):
    # remove "newline" and punctuation
    df.loc[transcript,'content'] = df.loc[transcript,'content'].replace("\n","").replace(".","")
    
    # create token for spacy
    doc = nlp(df.loc[transcript,'content'])
    # lemmatize with help of spacy (maybe we can try another library for this)
    if lemmatizing:
        df.loc[transcript,'content'] = ((",".join([x.lemma_ for x in doc])).replace(","," "))
        doc = nlp(df.loc[transcript,'content'])
    # remove all word except nouns, for now much better than without
    non_nouns.extend([word for word in doc if word.pos_!='NOUN'])# and word.pos_!='ADJ'])
    
    # write everything lowercase
    df.loc[transcript,'content'] = df.loc[transcript,'content'].lower()
    df.at[transcript,'content'] = df.loc[transcript,'content'].split()
    
    # stemming -> deprecated, use lemmatization instead
    if stemming:
        stemmer_tr=SnowballStemmer('german',ignore_stopwords=True)
        df.at[transcript,'content'] = [stemmer_tr.stem(word) for word in df.at[transcript,'content']]
        #print(df.at[i,'content'])
    
    # stopword removal
    if stopword_removal:
        stop_words=set(stopwords.words('german'))
        df.at[transcript,'content'] = [word for word in df.at[transcript,'content'] if word not in stop_words]
    
    # compensate structure for splitting
    df.at[transcript,'content'] = ','.join(df.loc[transcript,'content'])
    df.at[transcript,'content'] = (df.loc[transcript,'content']).replace(','," ")
    

non_nouns = [str(element) for element in non_nouns]

# remove all upper-case elements, not sure atm whether we need to do that
#non_nouns = [elem for elem in non_nouns if elem.lower()==elem]

# remove duplicates
non_nouns = (list(dict.fromkeys(non_nouns)))

# we need more words than spacy finds, I think LDA needs a lot of fine tuning
wordsToReplaceManually=['erst deutsch fernseh tagesschau','dame', 'herr','heut','ganz',
                'seit','jahr','neu','mehr','deutsch','schon','gross','wurd',
                'erst','imm','mal','gibt','sieht','klein','gut','etwa','ja',
                'beid','letzt','moglich','musst','klar','kommt','sagt','frag',
               'abend','lang','gest','morg','dabei','deshalb','einfach','schnell',
               'tag','zwei','wenig','beim','dafur','alt','drei','sag','kurz','ander',
                'dramatische','einig','eng','frischen','stillen','mensch','uhr','nacht',
                'land','norden','süden','woche','ende','heut']

# convert to matching format
for transcript in range(len(df)):
    df.at[transcript,'content'] = df.loc[transcript,'content'].split()
    
# remove non_nouns
for i in range(len(df)):
    removeFromWords = []
    for j in df.loc[i,'content']:
        if j in non_nouns:
            removeFromWords.append(j)
    removeFromWords = list(dict.fromkeys(removeFromWords))
    df.at[i,'content'] = [word for word in df.loc[i,'content'] if word not in removeFromWords]
    
#remove manually chosen words
for i in range(len(df)):
    removeFromWords = []
    for j in df.loc[i,'content']:
        if j in wordsToReplaceManually:
            removeFromWords.append(j)
            #df.loc[i,'content'] = df.loc[i,'content'].replace(j.lower(),"")
    #print(len(removeFromWords))
    removeFromWords = list(dict.fromkeys(removeFromWords))
    #print(len(removeFromWords))
    df.at[i,'content'] = [word for word in df.loc[i,'content'] if word not in removeFromWords]

# convert to matching format
for transcript in range(len(df)):
    df.at[transcript,'content'] = ','.join(df.loc[transcript,'content'])
    df.at[transcript,'content'] = (df.loc[transcript,'content']).replace(','," ")


# used from tutorial to create a workcloud
all_documents_as_one_string = ','.join(list(df['content'].values))
wordcloud = WordCloud(width=800,height=400,background_color="black", max_words=50, contour_width=3, contour_color='steelblue')
wordcloud.generate(all_documents_as_one_string)
time.sleep(0.3)
display(wordcloud.to_image())
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    print((count_dict))
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    print(x_pos)
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    dfPlot = pd.DataFrame(data={'words':words,'counts':counts})
    display(dfPlot)
    sns.barplot(x=words,y=counts,data=dfPlot,palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise count vectorizer
count_vectorizer = CountVectorizer()#stop_words=stopwords.words('german'))
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(df['content'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
                      
#params for LDA
number_topics = 10 #alpha
number_words = 5 #beta

# n_jobs=-1 to use all processors, max_iter=50 randomly chosen, not sure about it
lda = LDA(n_components=number_topics, n_jobs=-1,max_iter=50)
lda.fit(count_data)
print("LDA Topics:")
print_topics(lda, count_vectorizer, number_words)
