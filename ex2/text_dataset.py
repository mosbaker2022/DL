import torch
from torch.utils.data import Dataset
import numpy as np
from torchtext.data import get_tokenizer

def read_data(filepath, vec):
    with open(filepath, 'r') as f:  # Read file into string
        lines_data = f.readlines()
    label_map = {}
    sentences = []
    tags = []
    tokenizer = get_tokenizer("basic_english")
    for line1 in lines_data[1:-1]: # Skip header
        split_index = line1.find(',')
        lebel_str = line1[0:split_index]
        text_str = line1[split_index+1:-1]
        if not lebel_str in label_map:
            label_map[lebel_str] = len(label_map)
        tags.append(lebel_str)
        tokens = tokenizer(text_str)
        vectors = vec.get_vecs_by_tokens(tokens)
        sentences.append(vectors)
    return sentences, tags, label_map

class TextDataset(Dataset):
    def __init__(self, filepath, vec, max_allowed_seq):
        sentences, tags, label_map= read_data(filepath, vec)
        self.sentences_len = torch.Tensor([len(x) for x in sentences]).to(torch.long) # save the original lengths for the LSTM
        maxlen = min(max(np.array(self.sentences_len)), max_allowed_seq)
        # Make all sentences lengths equal to max length
        v_len = sentences[0].shape[1]

        for k in range(len(sentences)):
            if len(sentences[k]) < maxlen:
                sent = torch.cat((sentences[k],torch.zeros(maxlen - len(sentences[k]), v_len))) # padd
            else:
                sent = sentences[k][:maxlen] # trim
            sentences[k] = sent
        # Only store file names as a list because images are too big to store
        # The image will be loaded in __getitem__ using an index according to the order of this list
        self.sentences = sentences
        self.labels = []
        self.label_map = label_map
        for tag in tags:
            self.labels.append(label_map[tag]) # convert to class numbers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.sentences_len[idx]