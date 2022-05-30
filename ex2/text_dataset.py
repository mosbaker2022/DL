import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from nltk.tokenize import TweetTokenizer
import split_tokens

def tokenize_vectorize(words, unk_vect, vec, use_embs):
    tk = TweetTokenizer()
    tokens = list(tk.tokenize(words))
    tokens1 = []
    # Handle special cases with "'" (e.g i'm, he's etc)
    for t in tokens:
        if "'" in t:
            tokens1 += split_tokens.split_token(t.lower())
        else:
            tokens1.append(t)

    tokens = tokens1
    if use_embs:
        sentence_vectors = torch.zeros(len(tokens), 1).to(torch.long) # embeddings use index and not vectors as inputs
    else:
        sentence_vectors = torch.zeros(len(tokens), vec.vector_size)
    n = 0
    for t in tokens:
        if use_embs:
            if t in vec:
                i = vec.vocab[t].index + 2 # Add 2 because we added <pad> and <unk> at the beginning of the table
            elif t.lower() in vec:
                i = vec.vocab[t.lower()].index + 2
            else:
                i = 1  # index of unk is always 1
            sentence_vectors[n] = i

        else:
            if t in vec:
                v = vec[t]
            elif t.lower() in vec:
                v = vec[t.lower()]
            else:
                v = unk_vect  # this is for out of vocabulary words

            sentence_vectors[n] = torch.from_numpy(copy.deepcopy(v)).to(torch.double)
        n += 1
    return sentence_vectors

def read_data(filepath, unk_vect, vec, use_embs, label_map):
    with open(filepath, 'r') as f:  # Read file into string
        lines_data = f.readlines()
    if len(label_map) > 0: # label map already determined
        fixed_map = 1 # map is fixed
    else:
        fixed_map = 0 # build new map
    sentences = []
    tags = []
    for line1 in lines_data[1:]: # Skip header
        split_index = line1.find(',')
        lebel_str = line1[0:split_index]
        text_str = line1[split_index+1:]
        vectors = tokenize_vectorize(text_str, unk_vect, vec, use_embs)
        if len(vectors) == 0:
            continue # sentences with no text don't add anything, only confuse
        sentences.append(vectors)
        if fixed_map == 0: # build map in the loop
            if not lebel_str in label_map:
                label_map[lebel_str] = len(label_map)
        tags.append(lebel_str)
    return sentences, tags, label_map

    # Label map can come from an an argument or from the data. If label map argument is empty,
    # The learn the label map from the data.
class TextDataset(Dataset):
    def __init__(self, filepath, pad_vect, unk_vect, vec, label_map, max_allowed_seq, use_embs):
        sentences, tags, label_map= read_data(filepath, unk_vect, vec, use_embs, label_map)
        self.sentences_len = torch.Tensor([len(x) for x in sentences]).to(torch.long) # save the original lengths for the LSTM
        maxlen = min(max(np.array(self.sentences_len)), max_allowed_seq)
        self.sentences_len = torch.clamp(self.sentences_len, max = max_allowed_seq) # limit saved length values to maxlen
        # Make all sentences lengths equal to max length
        # If using embeddings, pad is just one index. If not, padd is a vector
        if use_embs:
            padd = torch.Tensor(1).to(torch.long)
        else:
            padd = torch.Tensor(pad_vect)

        for k in range(len(sentences)):
            if len(sentences[k]) < maxlen:
                sent = torch.cat((sentences[k], padd.repeat(maxlen - len(sentences[k]), 1)))  # padd
            else:
                sent = sentences[k][:maxlen]  # trim
            sentences[k] = sent
        self.sentences = sentences
        self.labels = []
        self.label_map = label_map
        for tag in tags:
            if tag in label_map: # protect against a label in the test that was not in the training
                self.labels.append(label_map[tag]) # convert to class numbers
            else:
                self.labels.append(len(label_map)) # this is an impossible class number in training
                print(f'Found a label: {tag}, which was not in the training set')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.sentences_len[idx]