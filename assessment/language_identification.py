import re
import datasets
import itertools
import time
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from nltk import ngrams
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from typing import Mapping, Tuple


torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download and load data set
dataset = datasets.load_dataset('wili_2018')

# Clean text, tokenize, and create vocabulary from text data
punctuations = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
    '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_',
    '`', '{', '|', '}', '~', '»', '«', '“', '”', "-",
]

def clean_text(text):
    text = re.sub(r"[-_.0-9A-Za-z]+@[-_0-9A-Za-z]+[-_.0-9A-Za-z]+", "", text)
    text = re.sub(r"https?://[-_.?&~;+=/#0-9A-Za-z]+", "", text)
    text = re.sub(r"[\+\d]?(\d{2,3}[-\.\s]??\d{2,3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})", "", text)
    text = re.sub(r'[~^0-9]', '', text)
    text = re.sub(r"([" + re.escape("".join(punctuations)) + "])", "", text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def tokenizer(text, ngram = 3):
    text = clean_text(text)
    return ["".join(k1) for k1 in list(ngrams(text, n=ngram))]

def tokenize_data(example, tokenizer):
    tokens = {'tokens': tokenizer(example['sentence'])}
    return tokens

def text_pipeline(text):
	return vocab(list(itertools.chain(['<bos>'], tokenizer(text), ['<eos>'])))

def generate_batch(data):
	batch = []
	labels = []
	for example in data:
		tokens = list(itertools.chain(['<bos>'], example['tokens'], ['<eos>']))
		data_tensor = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)
		labels.append(example['label'])
		batch.append(data_tensor)
	batch = pad_sequence(batch, padding_value=PAD_IDX)
	labels = torch.tensor(labels, dtype=torch.long)
	return batch, labels


tokenized_dataset = dataset.map(tokenize_data, remove_columns=['sentence'], fn_kwargs={'tokenizer': tokenizer})
vocab = build_vocab_from_iterator(tokenized_dataset['train']['tokens'], min_freq=3, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

PAD_IDX = vocab['<pad>']
BOS_IDX = vocab['<bos>']
EOS_IDX = vocab['<eos>']
UNK_IDX = vocab['<unk>']

vocab.set_default_index(UNK_IDX)

# Split train data into traning and validation set
train_dataset = tokenized_dataset['train']
test_dataset = tokenized_dataset['test']
num_train = int(len(train_dataset) * 0.75)
split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])


# LSTM Model for language indentification
class LanguageIndentificationModel(nn.Module):
    def __init__(self, num_class, vocab_size, embedding_dim):
        super().__init__()
        self.output_size = num_class
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX, sparse=True)
        self.fc = nn.Linear(embedding_dim, num_class)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, input):
        embedded = self.embedding(input)
        print('Embedding Shape:', embedded.shape)
        return F.log_softmax(self.fc(embedded))



# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 128 # batch size for training
VOCAB_SIZE = len(vocab) # number of vocabulary
EMB_SIZE = 512 # Embedding size
HID_DIM = 512 # Hidden dimension
NUM_LAYERS = 2 # Number of layers
NUM_CLASS = 235 # Number of output classess
DROP_RATE = 0.2


# Create model, loss function, and a optimizer for the model traning and validation
model = LanguageIndentificationModel(NUM_CLASS, VOCAB_SIZE, EMB_SIZE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(dataloader: DataLoader, epoch: int):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 10
    start_time = time.time()

    for idx, (source, target) in enumerate(dataloader):
        print('Target Shape:', source.shape)
        print('Target Shape:', len(target))
        optimizer.zero_grad()
        predicted_label = model(source)
        print('Predicted label Shape:', predicted_label.shape)
        loss = loss_fn(predicted_label, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predicted_label.argmax(1) == target).sum().item()
        total_count += target.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader: DataLoader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for _, (source, target) in enumerate(dataloader):
            predicted_label = model(source)
            _ = loss_fn(predicted_label, target)
            total_acc += (predicted_label.argmax(1) == target).sum().item()
            total_count += target.size(0)
    return total_acc/total_count


# main traning and validation loop
total_accu = None
train_dataloader = DataLoader(split_train, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(split_valid, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, epoch)
    accu_val = evaluate(valid_dataloader)

    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val

    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
    print('-' * 59)


# testing
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)

#print('Checking the results of test dataset.')
#accu_test = evaluate(test_dataloader)
#print('test accuracy {:8.3f}'.format(accu_test))


# class LanguageIndentificationModel(nn.Module):
#     def __init__(self, num_class, vocab_size, embedding_dim, hidden_dim, n_layers, batch_size=128, dropout_rate=0.5, tie_weights=False):
#         super().__init__()
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim
#         self.output_size = num_class
#         self.batch_size = batch_size
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX, sparse=True)
# 		# The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout_rate, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim, num_class)
#         self.dropout = nn.Dropout(dropout_rate)

#         self.init_weights()

#     def init_weights(self):
#         init_range = 0.1
#         self.embedding.weight.data.uniform_(-init_range, init_range)
#         self.fc.weight.data.uniform_(-init_range, init_range)
#         self.fc.bias.data.zero_()

#     def init_hidden(self):
# 		# Initiate hidden states.
#         # Shape for hidden state and cell state: num_layers * num_directions, batch, hidden_size
#         hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
#         cell = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim)
#         return (hidden, cell)

#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell

#     def forward(self, input, hidden):
# 		# Parameters
# 		# input -> sentences: padded sentences tensor. Each element of the tensor is an array of char-n-gram replaced with index
# 		# value from vocab.
#         # input = [sent len, batch size]
#         print('Input Shape:', input.shape)
#         embedding = self.dropout(self.embedding(input))
#         print('Embedding Shape:', embedding.shape)
#         # embedded = [sent len, batch size, emb dim]
# 		# hidden = [n layers, batch size, hidden dim]
#         lstm_output, hidden = self.lstm(embedding, hidden)
#         print('LSTM Shape:', lstm_output.shape)
#         # lstm output = [sent len, batch size, hidden dim * num directions]
#         output = self.dropout(lstm_output)
#         print('Output Shape:', output.shape)
# 		#hidden = [batch size, hidden dim * num directions]
#         prediction = self.fc(output)
#         # prediction = prediction.view(-1, self.output_size)
#         print('Prediction Shape:', prediction.shape)
#         return prediction, hidden
