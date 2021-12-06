# %%
import torch
import random
import numpy as np
import regex
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Opening and preprocessing input file
import pandas as pd
import nltk
from tqdm import tqdm
from src.preprocess import clean_text

data = pd.read_csv('train.csv', quotechar='"')
data.sample(frac=1)


# to convert authors into numbers
texts = {
    'EAP':"",
    'HPL':"", 
    'MWS':""
    
}

# lowercase, removing punctuation and tookenize sentences. Converting labels to int
for i in range(len(data)):
    texts[data['author'][i]] += " " + data['text'][i] 

    # data['text'][i] = nltk.word_tokenize(regex.sub(r'[^\w\s]', '',data['text'][i].lower()))
    # data['author'][i] = author_to_number[data['author'][i]]

print(texts['EAP'][:100])
print(texts['MWS'][:100])
print(texts['HPL'][:100])


# %% [markdown]
# # Mendenhall’s Characteristic Curves of Composition
# Literary scholar T. C. Mendenhall once wrote that an author’s stylistic signature could be found by counting how often he or she used words of different lengths.
# Plotting a graph of the word length distributions, the curves would look pretty much the same no matter what parts of the novel we had picked. Indeed, Mendenhall thought that if one counted enough words selected from various parts of a writer’s entire life’s work (say, 100,000 or so), the author’s “characteristic curve” of word length usage would become so precise that it would be constant over his or her lifetime.

# %%
authors = ('EAP','HPL','MWS')
federalist_by_author_tokens = {}
federalist_by_author_length_distributions = {}
for author in authors:
    tokens = nltk.word_tokenize(regex.sub(r'[^\w\s]', '',texts[author].lower()))

    # Filter out punctuation
    federalist_by_author_tokens[author] = tokens

    # Get a distribution of token lengths
    token_lengths = [len(token) for token in tokens]
    federalist_by_author_length_distributions[author] = nltk.FreqDist(token_lengths)
    federalist_by_author_length_distributions[author].plot(15,title=author)

# %%
# Performance metrics
def accuracy(true, pred):
    acc = 0
    den = 0
    for i in range(len(true)):
        if true[i]==pred[i]:
            acc+=1
        den +=1
    return acc/den


def binary_macro_f1(true, pred):
    f1 = None
    f1 = []

    if 0 in pred and 1 in pred and 2 in pred:
        for label in range(3):
            precision = 0
            recall = 0
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i in range(len(true)):
                if true[i] == label and pred[i] == label:
                    tp+=1
                elif true[i] != label and pred[i] == label:
                    fp+=1
                elif true[i] == label and pred[i] != label:
                    fn+=1
                else:
                    tn+=1
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

            f1.append(2*precision*recall/(precision+recall)) 
        f1 = (f1[0] + f1[1] + f1[2])/3
    else:
        return 0
    return f1

# %% [markdown]
# # Kilgariff’s Chi-Squared Method
# 
# In a 2001 paper, Adam Kilgarriff15 recommends using the chi-squared statistic to determine authorship. Readers familiar with statistical methods may recall that chi-squared is sometimes used to test whether a set of observations (say, voters’ intentions as stated in a poll) follow a certain probability distribution or pattern. This is not what we are after here. Rather, we will simply use the statistic to measure the “distance” between the vocabularies employed in two sets of texts. The more similar the vocabularies, the likelier it is that the same author wrote the texts in both sets. This assumes that a person’s vocabulary and word usage patterns are relatively constant.

# %%
from nltk.metrics import *
def split_train_val_test(df, props=[.9, .1]):
    train_df, test_df = None, None

    train_size = int(props[0] * len(df))
    test_size =train_size + int(props[1] * len(df)) 
    train_df = df.iloc[0:train_size]
    test_df = df.iloc[train_size:]
    
    return train_df, test_df

def chi_squared(candidate, disputed):
    # tokenizing
    candidate = nltk.word_tokenize(candidate)
    disputed = nltk.word_tokenize(disputed)

    joint_corpus =  candidate + disputed 
    # most freq words list
    joint_freq_dist = nltk.FreqDist(joint_corpus)
    joint_freq_dist = list(joint_freq_dist.most_common(500))

    author_share = (len(candidate)) / len(joint_corpus)

    chisqured = 0
    for word, joint_count in joint_freq_dist:

        author_count = candidate.count(word)
        disputed_count = disputed.count(word)

        adjusted_candidate_count = joint_count * author_share
        adjusted_disputed_count = joint_count * (1-author_share)

        chisqured += ((author_count-adjusted_candidate_count) *
                       (author_count-adjusted_candidate_count) /
                       adjusted_candidate_count)

        chisqured += ((disputed_count-adjusted_disputed_count) *
                       (disputed_count-adjusted_disputed_count)
                       / adjusted_disputed_count)
    
    return chisqured


from src.dataset import *

# Splitting dataset
train_df = split_train_val_test(data)
test_df = train_df[1]
train_df = train_df[0]

# to convert authors into numbers
texts = {
    'EAP':"",
    'HPL':"", 
    'MWS':""
    
}

# lowercase, removing punctuation and tookenize sentences
for i in range(len(train_df)):
    texts[data['author'][i]] += " " + regex.sub(r'[^\w\s]', '',train_df['text'].iloc[i].lower())
author_to_number = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
}
preds = []
labels = []

# Predicting author using chi_squared test

for i in range(5):
    curr_text =  regex.sub(r'[^\w\s]', '',test_df['text'].iloc[i].lower())
    curr_label = author_to_number[test_df['author'].iloc[i]]
    chis = []
    for author in texts:
        chis.append(chi_squared(texts[author],curr_text))
    preds.append(np.argmin(chis))
    labels.append(curr_label)
# labels = set(labels)
# preds = set(preds)
# print(labels)
# print(preds)

# print(f"TEST F-1: {f_measure(labels, preds)}")
print(f"TEST ACC: {accuracy(labels, preds)}")



# %%
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# %% [markdown]
# # John Burrows’ Delta Method
# 
# Burrows’ Delta is a measure of the “distance” between a text whose authorship we want to ascertain and some other corpus. Unlike chi-squared, however, the Delta Method is designed to compare an anonymous text (or set of texts) to many different authors’ signatures at the same time. More precisely, Delta measures how the anonymous text and sets of texts written by an arbitrary number of known authors all diverge from the average of all of them put together. Furthermore, the Delta Method gives equal weight to every feature that it measures, thus avoiding the problem of common words overwhelming the results, which was an issue with chi-squared tests. For all of these reasons, John Burrows’ Delta Method is usually a more effective solution to the problem of authorship.

# %%
import math
# Making a joint corpus with all training data
corpus = []
for i in range(len(train_df)):
    corpus += nltk.word_tokenize(regex.sub(r'[^\w\s]', '',train_df['text'].iloc[i].lower()))
# Frequency distribution of words
freq_dist = list(nltk.FreqDist(corpus).most_common(30))

# Calculate the presence of each word on each of the author's corpus
# for author in texts:
#     texts[author] = nltk.word_tokenize(regex.sub(r'[^\w\s]', '',texts[author].lower()))

feature_freqs = {}
features = [word for word,freq in freq_dist]
for author in texts:    
    # A dictionary for each candidate's features
    feature_freqs[author] = {}

    # A helper value containing the number of tokens in the author's subcorpus
    overall = len(texts[author])

    # Calculate each feature's presence in the subcorpus
    for feature in features:
        presence = texts[author].count(feature)
        feature_freqs[author][feature] = presence / overall

# Calculating averages and std deviations
# The data structure into which we will be storing the "corpus standard" statistics
corpus_features = {}

# For each feature...
for feature in features:
    # Create a sub-dictionary that will contain the feature's mean
    # and standard deviation
    corpus_features[feature] = {}

    # Calculate the mean of the frequencies expressed in the subcorpora
    feature_average = 0
    for author in authors:
        feature_average += feature_freqs[author][feature]
    feature_average /= len(authors)
    corpus_features[feature]["Mean"] = feature_average

    # Calculate the standard deviation using the basic formula for a sample
    feature_stdev = 0
    for author in authors:
        diff = feature_freqs[author][feature] - corpus_features[feature]["Mean"]
        feature_stdev += diff*diff
    feature_stdev /= (len(authors) - 1)
    feature_stdev = math.sqrt(feature_stdev)
    corpus_features[feature]["StdDev"] = feature_stdev

    # Calculating z-scores
    feature_zscores = {}
for author in authors:
    feature_zscores[author] = {}
    for feature in features:

        # Z-score definition = (value - mean) / stddev
        # We use intermediate variables to make the code easier to read
        feature_val = feature_freqs[author][feature]
        feature_mean = corpus_features[feature]["Mean"]
        feature_stdev = corpus_features[feature]["StdDev"]
        feature_zscores[author][feature] = ((feature_val-feature_mean) /
                                            feature_stdev)
current_test_tokens = []
# Calculate z-score of test instances
preds = []
labels = []
for i in range(len(test_df)):
    current_test_tokens = nltk.word_tokenize(regex.sub(r'[^\w\s]', '',test_df['text'].iloc[i].lower()))
    labels.append(author_to_number[test_df['author'].iloc[i]])
    overall = len(current_test_tokens)
    testcase_freqs = {}
    # Calculate the test case's features
    for feature in features:
        presence = current_test_tokens.count(feature)
        testcase_freqs[feature] = presence / overall

    # Calculate the test case's feature z-scores
    testcase_zscores = {}
    for feature in features:
        feature_val = testcase_freqs[feature]
        feature_mean = corpus_features[feature]["Mean"]
        feature_stdev = corpus_features[feature]["StdDev"]
        testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev
    deltas = []
    for author in authors:
        delta = 0
        for feature in features:
            delta += math.fabs((testcase_zscores[feature] -
                                feature_zscores[author][feature]))
            
        delta /= len(features)
        deltas.append(delta)
    preds.append(np.argmin(deltas))
        # print( "Delta score for candidate", author, "is", delta )

print(f"TEST ACC: {nltk.accuracy(labels, preds)}")



# %% [markdown]
# # Cloud Words

# %%
def generate_word_cloud(text, title):
    # Generate word cloud.
    wc = WordCloud(background_color='black', max_words=1000,
                  stopwords=STOPWORDS, max_font_size=40)
    wc.generate(" ".join(text))
    
    # Plot word cloud using matplotlib.
    plt.figure(figsize=(16, 13))
    plt.title(title, fontsize=20)
    plt.imshow(wc.recolor(colormap='Pastel2', random_state=42), alpha=0.98)
    plt.axis('off')
eap = data[data.author=="EAP"]["text"].values
hpl = data[data.author=="HPL"]["text"].values
mws = data[data.author=="MWS"]["text"].values

# generate_word_cloud(hpl, "HPL")
# generate_word_cloud(eap, "EAP")
# generate_word_cloud(mws, "MWS")


# %% [markdown]
# ### Unigram and Bigram frequencies

# %%
def ngrams(n, text:str):
    text=text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    result = []
    if n == 0:
        for index in range(0,len(text),):
            result.append(('',text[index]))

    else:
        for index in range(0,len(text)):
            if index - n >= 0:
                result.append((' '.join(text[index-n:index]),text[index]))
    return result

def count_ngrams(sentences, n):
        ''' Updates the model n-grams based on text '''
        # text = text.replace('[EOS]','')
        counts = {}
        ngramss = []

        for text in sentences:
            ngramss += ngrams(n, text.lower())
        
        for tuple in ngramss:
            if tuple[0] not in counts.keys():
                counts[tuple[0]]=1
            else: 
                counts[tuple[0]]+=1
            # if (tuple[0]+tuple[1]) not in self.counts.keys():
            #     self.counts[tuple[0]+tuple[1]]=1
            # else:
            #     self.counts[tuple[0]+tuple[1]]+=1
        return counts

def wordBarGraphFunction(word_count_dict,title,bigrams=False):
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in STOPWORDS]
    plt.figure(figsize=(16, 13))
    if bigrams:
        plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[100:150])])
        plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[100:150]))
    
    else:
        plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
        plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    
    plt.title(title)
    plt.show()
  

# wordBarGraphFunction(count_ngrams(hpl,1),'HPL',bigrams = False) 
# wordBarGraphFunction(count_ngrams(eap,1),'EAP',bigrams = False)
# wordBarGraphFunction(count_ngrams(mws,1),'MWS',bigrams = False)

# # %%
# wordBarGraphFunction(count_ngrams(hpl,2),'HPL',bigrams = False) 
# wordBarGraphFunction(count_ngrams(eap,2),'EAP',bigrams = False)
# wordBarGraphFunction(count_ngrams(mws,2),'MWS',bigrams = False)

# # %%
# wordBarGraphFunction(count_ngrams(hpl,3),'HPL',bigrams = False) 
# wordBarGraphFunction(count_ngrams(eap,3),'EAP',bigrams = False)
# wordBarGraphFunction(count_ngrams(mws,3),'MWS',bigrams = False)

# # %%
# wordBarGraphFunction(count_ngrams(hpl,4),'HPL',bigrams = False) 
# wordBarGraphFunction(count_ngrams(eap,4),'EAP',bigrams = False)
# wordBarGraphFunction(count_ngrams(mws,4),'MWS',bigrams = False)

# %%
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
# pyLDAvis.enable_notebook()
features = nltk.word_tokenize(regex.sub(r'[^\w\s]', '', texts['MWS'].lower()))
# Use tf-idf features for NMF.
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=len(set(features)), stop_words="english"
)
tfidf = tfidf_vectorizer.fit_transform(features)

# Use tf (raw term count) features for LDA.
tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, max_features=len(set(features)), stop_words="english"
)

tf = tf_vectorizer.fit_transform(features)



lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)

# nmf = NMF(n_components=10, random_state=1, alpha=0.1, l1_ratio=0.5).fit(tfidf)

tfidf_feature_names = tf_vectorizer.get_feature_names()

# tf_feature_names = tf_vectorizer_mws.get_feature_names()

pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)


