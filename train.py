from typing import Text
import spacy

import torch
import torch.optim as optim
import torch.nn as nn

import torchtext
from torchtext import data

import sklearn
import wandb
import os

from models import Linear, MLP1, MLP2

wandb.init(project='meta-analogy')

spacy_en = spacy.load('en')

def tokenizer(text):
    return text.split()

TEXT = data.Field(sequential=True, tokenize="spacy")
LABEL = data.Field(sequential=True)

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='.',
    train='train-analogies.csv',
    validation='val-analogies.csv',
    test='test-analogies.csv',
    format='csv',
    skip_header=True,
    fields=[('Text', TEXT), ('Label', LABEL)]
)

vocab = data.TabularDataset(path='vocab.txt',
    fields=[('Text', TEXT)],
    format='csv',
    skip_header=True)

print('Loaded data.')

TEXT.build_vocab(vocab, vectors='glove.840B.300d')
LABEL.build_vocab(vocab, vectors='glove.840B.300d')

########################################################
# HYPERPARAMETERS
########################################################

BATCH_SIZE = 32
EMBEDDING_DIM = 900
OUTPUT_DIM = 300
INPUT_DIM = 300
N_EPOCHS = 100
wandb.config.update({"batch_size": BATCH_SIZE})

########################################################
# END HYPERPARAMETERS
########################################################

device =  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    sort_key=lambda x: len(x.Text),
    device=device)

# print(TEXT.vocab.freqs)

model = Linear(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)
# model = MLP1(INPUT_DIM, EMBEDDING_DIM, 1000, OUTPUT_DIM)
pretrained_embeddings = TEXT.vocab.vectors
wandb.watch(model)
# print(pretrained_embeddings[500])

model.embedding.weight.data = pretrained_embeddings
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='sum')

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    
    model.train()

    for batch in iterator:
        if batch.Text.size(0) != 3: # don't consider multi-word analogies
            continue
        optimizer.zero_grad()

        predictions = model(batch.Text)
        labels = model.embedding(batch.Label).squeeze(0)
        # print(predictions[0], labels[0])
        # print(predictions.shape, labels.shape) # get embeddings of labels
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            # print(batch.Text.shape)
            if batch.Text.size(0) != 3: # don't consider multi-word analogies
                continue

            predictions = model(batch.Text)
            labels = model.embedding(batch.Label).squeeze(0)
            # print(predictions.shape, batch.Label.shape)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

best_value = None

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    wandb.log({'training loss' : train_loss, 'validation loss': valid_loss})
    print(f'Train loss: {train_loss} | Validation loss: {valid_loss}')

    if (best_value is None) or (best_value > valid_loss):
        best_value = valid_loss
        save_model = True
    else:
        save_model = False

    if save_model:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))