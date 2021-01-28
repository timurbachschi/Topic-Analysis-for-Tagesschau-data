"""Module for creating word embeddings"""
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec
import pandas as pd
from functools import reduce
import numpy as np
import gensim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm


# Code taken from https://gist.github.com/tangert/106822a0f56f8308db3f1d77be2c7942
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        (With help from William. Thank you!)

    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """
    
    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    base_embed.init_sims()
    other_embed.init_sims()

    # make sure vocabulary and indices are aligned
    models = align_gensim_models([base_embed, other_embed], words=words)
    in_base_embed = models[0]
    in_other_embed = models[1]

    # get the embedding matrices
    base_vecs = in_base_embed.wv.syn0norm
    other_vecs = in_other_embed.wv.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.wv.vectors_norm = other_embed.wv.syn0 = (other_embed.wv.vectors_norm).dot(ortho)
    return other_embed


# Code taken from: https://gist.github.com/tangert/106822a0f56f8308db3f1d77be2c7942
def align_gensim_models(models, words=None):
    # Code originally ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
    """
    Returns the aligned/intersected models from a list of gensim word2vec models.
    Generalized from original two-way intersection as seen above.
    
    Also updated to work with the most recent version of gensim
    Requires reduce from functools
    
    In order to run this, make sure you run 'model.init_sims()' for each model before you input them for alignment.
    
    ##############################################
    ORIGINAL DESCRIPTION
    ##############################################
    
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocabs = [set(m.wv.vocab.keys()) for m in models]

    # Find the common vocabulary
    common_vocab = reduce((lambda vocab1,vocab2: vocab1&vocab2), vocabs)
    if words: common_vocab&=set(words)

    # If no alignment necessary because vocab is identical...
    
    # This was generalized from:
    # if not vocab_m1-common_vocab and not vocab_m2-common_vocab and not vocab_m3-common_vocab:
    #   return (m1,m2,m3)
    if all(not vocab-common_vocab for vocab in vocabs):
        print("All identical!")
        return models
        
    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: sum([m.wv.vocab[w].count for m in models]),reverse=True)
    
    # Then for each model...
    for m in models:
        
        # Replace old vectors_norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
                
        old_arr = m.wv.vectors_norm
                
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors_norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index,word in enumerate(common_vocab):
            old_vocab_obj=old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return models


def drawArrow(A, B, head_width=0.5):
    # Draw arrow between two points in a plot
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=head_width, length_includes_head=True, color="black", alpha=0.5, linewidth=2.5)


def tsne_plot(models, keys, years, context_len, xlim=None, arrow_head_width=0.5):
    """
    Creates TSNE plot for given list of models and keys.

            Parameters:
                    models (List): List of models used to get word vectors that are to be plotted.
                    keys (List(str)): Tokens to plot.
                    years (List(str)): Labels of the models inputted i.e. "2008" or "08-11"
                    context_len (int): Number of words closest to keys that are additionally plotted
                    xlim ([int, int]): Limit of x-axis. None by default.
                    arrow_head_width (float): Head width of arrows connecting the keys for each year. 0.5 by default. 


            Returns:
                    None
    """
    plt.figure(figsize=(18, 18))
    labels = []
    tokens = []
    # Extract word vectors for all models
    for model in models:
        for word in keys:
            tokens.append(model[word])
            labels.append(word)
            for similar_word, _ in model.most_similar(word, topn=context_len):
                labels.append(similar_word)
                tokens.append(model[similar_word])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    old_coords = {key: None for key in keys}
    # For all given models, plot the nodes and text in the tsne plot
    for i, model in enumerate(models):
        # Get all coordinates
        x = new_values[(len(keys)+context_len)*i: (len(keys)+context_len)*i+len(keys)*context_len+len(keys), 0]
        y = new_values[(len(keys)+context_len)*i: (len(keys)+context_len)*i+len(keys)*context_len+len(keys), 1]
        words_iteration = labels[(len(keys)+context_len)*i: (len(keys)+context_len)*i+len(keys)*context_len+len(keys)]
        # Plot nodes
        plt.scatter(x, y, c=colors[i], alpha=0.7, label=years[i], s=320)
        # Add text to nodes
        for j, word in enumerate(words_iteration):
            if j%(context_len+1) == 0:
                alpha=1.0
                if old_coords[word] is None:
                    old_coords[word] = (x[j], y[j])
                else:
                    new_coords = (x[j], y[j])
                    drawArrow(old_coords[word], new_coords, arrow_head_width)
                    old_coords[word] = new_coords
            else:
                # Grey out text if its not the key
                alpha=0.3
            plt.annotate(word,
                             xy=(x[j], y[j]),
                             xytext=(4, 3),
                             textcoords='offset points',
                             ha='right',
                             va='bottom',size=30, alpha=alpha)
    plt.legend(loc=4, fontsize=20)
    plt.grid(True)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show() 


def create_word_embedding(dataframe, min_count_p=50, progress_per_p=10000, min_count_m=10, window_m=2, sample_m=6e-5,
                           alpha_m=0.03, min_alpha_m=7e-4, negative_m=20, progress_per_m=2000, epochs=30, report_delay=1):
    # Create word embeddings with preprocessed dataframe
    sentences = [row for row in dataframe['content']]
    # Create word2vec model
    cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=min_count_m,
                         window=window_m,
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
    #w2v_model.init_sims(replace=True)
    return w2v_model
