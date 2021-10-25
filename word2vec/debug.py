# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
def split_train_val_test(df, props=[.8, .1, .1]):
    assert round(sum(props), 2) == 1 and len(props) >= 2
    train_df, test_df, val_df = None, None, None

    train_size = int(props[0] * len(df))
    val_size =  train_size + int(props[1] * len(df))
    test_size =val_size + int(props[2] * len(df)) 
    train_df = df.iloc[0:train_size]
    val_df = df.iloc[train_size:val_size]
    test_df = df.iloc[val_size:test_size]
    
    return train_df, val_df, test_df


# %%
import gensim.downloader as api

def download_embeddings(fasttetxt):
    # https://fasttext.cc/docs/en/english-vectors.html
    if fasttetxt:
      wv = api.load("fasttext-wiki-news-subwords-300")
    else:
      
      wv = api.load("word2vec-google-news-300")
      print("\nLoading complete!\n" +
            "Vocabulary size: {}".format(len(wv.vocab)))
    return wv


# %%



# %%
# Opening and preprocessing input file
import gensim.models
import pandas as pd
import nltk
from tqdm import tqdm
from src.preprocess import clean_text

data = pd.read_csv('train.csv', quotechar='"')
data.sample(frac=1)


# to convert authors into numbers
author_to_number = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
    
}

# lowercase, removing punctuation and tookenize sentences. Converting labels to int
training_text = ""
for i in range(len(data)):

    data['text'][i] = nltk.word_tokenize(regex.sub(r'[^\w\s]', '',data['text'][i].lower()))
    data['author'][i] = author_to_number[data['author'][i]]

print(data[0:10])

from src.dataset import *

# Splitting dataset and generating vocab
train_df, val_df, test_df = split_train_val_test(data)
train_vocab, reversed_vocab = generate_vocab_map(train_df)


# %%
DOWNLOAD = True
# Use fastext or word2vec
FASTTEXT = True
WINDOW_SIZE = 5

EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 1
BIDIRECTIONAL = True


# %%
# Downloading or generating word2vec embeddings

if DOWNLOAD:
    model = download_embeddings(FASTTEXT)
else:
    if FASTTEXT:
        model = gensim.models.FastText(sentences=train_df['text'], size=EMBEDDING_DIM, window=WINDOW_SIZE)
    else:
        model = gensim.models.Word2Vec(sentences=train_df['text'], size=EMBEDDING_DIM, window=WINDOW_SIZE)
                        


# %%
from src.dataset import HeadlineDataset
from torch.utils.data import RandomSampler

train_dataset = HeadlineDataset(train_vocab, train_df,model.wv, FASTTEXT)
val_dataset = HeadlineDataset(train_vocab, val_df,model.wv, FASTTEXT)
test_dataset = HeadlineDataset(train_vocab, test_df,model.wv, FASTTEXT)

# Now that we're wrapping our dataframes in PyTorch datsets, we can make use of PyTorch Random Samplers.
train_sampler = RandomSampler(train_dataset)
val_sampler = RandomSampler(val_dataset)
test_sampler = RandomSampler(test_dataset)


# %%
from torch.utils.data import DataLoader
from src.dataset import collate_fn
BATCH_SIZE = 16
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn)

for x, y in test_iterator:
    print(x,y)
    break

# %% [markdown]
# ### Modeling

# %%
from src.models import ClassificationModel

model = ClassificationModel(len(train_vocab),embedding_dim=EMBEDDING_DIM,hidden_dim = HIDDEN_DIM,num_layers = NUM_LAYERS,bidirectional = BIDIRECTIONAL)

# %% [markdown]
# In the following cell, **instantiate the model with some hyperparameters, and select an appropriate loss function and optimizer.** 
# 
# Hint: we already use sigmoid in our model. What loss functions are availible for binary classification? Feel free to look at PyTorch docs for help!

# %%
from torch.optim import AdamW

criterion, optimizer = torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

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


