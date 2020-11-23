from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec
import pandas as pd


def create_word_embeddings(dataframe, min_count_p=20, progress_per_p=10000, min_count_m=10, window_m=2, sample_m=6e-5,
                           alpha_m=0.03, min_alpha_m=7e-4, negative_m=20, progress_per_m=2000, epochs=30, report_delay=1):
    # TODO: Add documentation
    # Create word embeddings with preprocessed dataframe
    # Create bigrams/phrases
    sent = [row.split() for row in dataframe['content']]
    phrases = Phrases(sent, min_count=min_count_p, progress_per=progress_per_p)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    # Create word2vec model
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=min_count_m,
                         window=window_m,
                         # size=300,
                         sample=sample_m,
                         alpha=alpha_m,
                         min_alpha=min_alpha_m,
                         negative=negative_m,
                         workers=cores-1)
    # Build vocabulary
    w2v_model.build_vocab(sentences, progress_per=progress_per_m)
    # Train
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count,
                    epochs=epochs, report_delay=report_delay)
    # Freeze model
    w2v_model.init_sims(replace=True)
