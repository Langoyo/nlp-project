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
authors = ('EAP','MWS','HPL')
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

# %% [markdown]
# # Kilgariff’s Chi-Squared Method
# 
# In a 2001 paper, Adam Kilgarriff15 recommends using the chi-squared statistic to determine authorship. Readers familiar with statistical methods may recall that chi-squared is sometimes used to test whether a set of observations (say, voters’ intentions as stated in a poll) follow a certain probability distribution or pattern. This is not what we are after here. Rather, we will simply use the statistic to measure the “distance” between the vocabularies employed in two sets of texts. The more similar the vocabularies, the likelier it is that the same author wrote the texts in both sets. This assumes that a person’s vocabulary and word usage patterns are relatively constant.

# %%
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
    texts[data['author'][i]] += " " + regex.sub(r'[^\w\s]', '',train_df['text'][i].lower())

for i in range(len(test_df)):
    curr_text =  regex.sub(r'[^\w\s]', '',test_df['text'].iloc[i].lower())
    curr_label = test_df['author'].iloc[i]
    for author in texts:
        print(chi_squared(texts[author],curr_text))




# %%
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

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

generate_word_cloud(hpl, "HPL")
generate_word_cloud(eap, "EAP")
generate_word_cloud(mws, "MWS")


# %% [markdown]
# ### Unigram frequencies

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

def wordBarGraphFunction(word_count_dict,title):
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in STOPWORDS]
    plt.figure(figsize=(16, 13))
    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    plt.show()
  

wordBarGraphFunction(count_ngrams(hpl,1),'HPL') 
wordBarGraphFunction(count_ngrams(eap,1),'EAP')
wordBarGraphFunction(count_ngrams(mws,1),'MWS')
wordBarGraphFunction(count_ngrams(hpl,2),'HPL') 
wordBarGraphFunction(count_ngrams(eap,2),'EAP')
wordBarGraphFunction(count_ngrams(mws,2),'MWS')

# %% [markdown]
# In the following cell, **instantiate the model with some hyperparameters, and select an appropriate loss function and optimizer.** 
# 
# Hint: we already use sigmoid in our model. What loss functions are availible for binary classification? Feel free to look at PyTorch docs for help!

# %%
# returns the total loss calculated from criterion
def train_loop(model, criterion, iterator):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(iterator):
        optimizer.zero_grad()

        prediction = model(x)
        prediction = torch.squeeze(prediction)
        # y = y.round()
        # y = y.long()

 
        loss = criterion(prediction,y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss

# returns:
# - true: a Python boolean array of all the ground truth values 
#         taken from the dataset iterator
# - pred: a Python boolean array of all model predictions. 
def val_loop(model, criterion, iterator):
    true, pred = [], []
    for x, y in tqdm(iterator):
    
        preds = model(x)
        preds = torch.squeeze(preds)
        for i_batch in range(len(y)):
            true.append(y[i_batch])
            pred.append(torch.argmax(preds[i_batch]))
            
    return true, pred


# %% [markdown]
# ### Part 3: Training and Evaluation [10 Points]
# The final part of this HW involves training the model, and evaluating it at each epoch. **Fill out the train and test loops below.**

# %%
# returns the total loss calculated from criterion
def train_loop(model, criterion, iterator):
    model.train()
    total_loss = 0
    
    for x, y in tqdm(iterator):
        optimizer.zero_grad()

        prediction = model(x)
        prediction = torch.squeeze(prediction)
        # y = y.round()
        # y = y.long()

 
        loss = criterion(prediction,y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss

# returns:
# - true: a Python boolean array of all the ground truth values 
#         taken from the dataset iterator
# - pred: a Python boolean array of all model predictions. 
def val_loop(model, criterion, iterator):
    true, pred = [], []
    for x, y in tqdm(iterator):
    
        preds = model(x)
        preds = torch.squeeze(preds)
        for i_batch in range(len(y)):
            true.append(y[i_batch])
            pred.append(torch.argmax(preds[i_batch]))
            
    return true, pred


# %%
# To test your eval implementation, let's see how well the untrained model does on our dev dataset.
# It should do pretty poorly.
from src.eval_utils import binary_macro_f1, accuracy
true, pred = val_loop(model, criterion, val_iterator)
# print(binary_macro_f1(true, pred))
# print(accuracy(true, pred))


# %% [markdown]
# ### Actually training the model

# %%
TOTAL_EPOCHS = 7
for epoch in range(TOTAL_EPOCHS):
    train_loss = train_loop(model, criterion, train_iterator)
    true, pred = val_loop(model, criterion, val_iterator)
    print(f"EPOCH: {epoch}")
    print(f"TRAIN LOSS: {train_loss}")
    print(f"VAL F-1: {binary_macro_f1(true, pred)}")
    print(f"VAL ACC: {accuracy(true, pred)}")


# %% [markdown]
# We can also look at the models performance on the held-out test set, using the same val_loop we wrote earlier.

# %%
true, pred = val_loop(model, criterion, test_iterator)
print(f"TEST F-1: {binary_macro_f1(true, pred)}")
print(f"TEST ACC: {accuracy(true, pred)}")


