# Utils for sentences to author data
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np



# Takes in df and creates processed token column
def frame_process(data, no_stop = True):
    data['tokens'] = data['text'].apply(reduced_word_tokenize)


# Takes in sentence and returns normalized tokens
def norm_word_tokenize(sentence):
    tokens = [t.lower() for t in word_tokenize(sentence)]
    return tokens

# Takes in sentence and returns normalized tokens without blank or stop words
def reduced_word_tokenize(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = [t.lower().strip() for t in word_tokenize(sentence) if _is_valid_token(t)]
    return tokens

def _is_valid_token(t):
    if t.lower().strip() in stopwords.words('english'):
        return False
    return True

# Returns a sorted list vocab for the input dataframe
def create_vocab(data):
    unique = data['tokens'].explode().unique().astype(str)
    unique_list = sorted(unique.tolist())
    return unique_list

# Creates translation maps for input list
def create_translation(vocab):
    vocab.insert(0, 'UNK')
    translation = dict(zip(vocab, range(len(vocab))))
    reverse = dict(zip(range(len(vocab)), vocab))
    return translation, reverse

def get_counts(tokens, vocab):
    bag_vec = [0]
    unk_count = 0
    for v in range(len(vocab))[1:]:
        count = (tokens.count(vocab[v]))
        bag_vec.append(count)
        unk_count += count
    bag_vec[0] = len(tokens) - unk_count
    return bag_vec

def frame_counts(data, vocab):
    data['bag_vec'] = data['tokens'].apply(lambda x: get_counts(x, vocab))

# python map


# tf-idf
def total_bag_vec(data):
    arr = np.array(data['bag_vec'].tolist())
    return np.sum(arr, axis=0)

def idf_vec(data):
    arr = np.array(data['bag_vec'].tolist())
    arr[arr > 0] = 1
    idf_arr = np.sum(arr, axis=0) + 1
    idf_arr = np.log(len(data) / idf_arr)
    return idf_arr

def tfidf(bag_vec, idf_arr):
    tf = (np.array(bag_vec) + 1) / (np.sum(bag_vec) + len(bag_vec))
    # print(np.sum(bag_vec))
    tf_idf = tf * idf_arr
    return tf_idf

def frame_tfidf(data, idf_arr):
    data['tfidf_vec'] = data['bag_vec'].apply(lambda x: tfidf(x, idf_arr))
    
