from math import ceil
import re
import numpy as np


def dummy_load_data(input_file, time_window):
    data = open(input_file, 'r').read()
    chars = list(set(data))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    training_data = []
    for p in range(len(chars)):#Change to data
        if p + time_window + 1 < len(data): # throw the others, but latter will take them as well.
            inputs = [char_to_ix[ch] for ch in data[p:p + time_window]]
            targets = [char_to_ix[ch] for ch in data[p + 1:p + time_window + 1]]
            training_data.append((inputs, targets))
        p += time_window
    return training_data

def load_data(input_file, vocabulary_list):
    data = []
    tgt_data = list(open(input_file, 'r'))
    word_vocabulary = sentence_to_words(tgt_data)
    i = 1
    for k,v in word_vocabulary.items():
        vocabulary_list.append(k)
        word_vocabulary[k] = (v, i) # You have the word you want to get the frequency and index. i.e. you use this to get embeding.
        i+=1
    for sentence in tgt_data:
        data += embed_to_vector(sentence, word_vocabulary)
    return data

def sentence_to_words(tgt_data):
    word_vocabulary = {}
    words_in_sentences = []
    for sentence in tgt_data:
        words_in_sentences += [w for w in re.split(pattern=" ", string=sentence)]
    for word in words_in_sentences:
        if word not in word_vocabulary:
            word_vocabulary[word] = 1
        else:
            word_vocabulary[word] +=1
    return word_vocabulary

def embed_to_vector(input_sentence, word_vocabulary):
    output_vector = []
    for w in re.split(pattern=" ", string=input_sentence):
        output_vector.append(word_vocabulary[w][1])# We currently use only the index to present coordinates on vector space.
    return output_vector

def batch_training_data(training_data, batch_size, time_window, null_token_index):
    data_size = len(training_data)
    data_batch = [None]*(ceil(data_size/(batch_size*time_window)))
    n = pos = 0
    while pos <data_size:
        if data_size > pos + (time_window*batch_size):
            data_batch[n] = [w for w in training_data[pos:pos+(time_window*batch_size)]]
            pos += (batch_size*time_window)
            n+=1
        else:
            data_batch[n] = [training_data[pos+i] if (pos+i)<data_size else null_token_index for i in range(time_window*batch_size)]
            pos = data_size
    return data_batch

def init_model(network_length, network_depth, weight_matricies, rows, columns, rng):
    return rng.random((network_length, network_depth, weight_matricies, rows, columns))*1e-2