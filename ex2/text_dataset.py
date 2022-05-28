import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
#from torchtext.data import get_tokenizer
from nltk.tokenize import TweetTokenizer

# For debug
#--------------
total_word_count_list = []
total_oov_count_list = []
#--------------

def tokenize_vectorize(words, vec, tokenizer_use_lowercase):
    tk = TweetTokenizer()
    #tokens = list(tokenize(words, lowercase = tokenizer_use_lowercase))
    tokens = list(tk.tokenize(words))
    sentence_vectors = torch.zeros(len(tokens), vec.vector_size)
    v0 = np.zeros(vec.vector_size)
    oov_counter = 0
    n = 0
    for t in tokens:
        if t in vec:
            v = vec[t]
        else:
            v = v0  # For words that are not in the vocabulary use the fixed predefined dummy word V0
        # For debug
        # --------------
            oov_counter += 1 # count the number of out of vocab words- for debug
        total_word_count_list.append(len(tokens))
        total_oov_count_list.append(oov_counter)
        #--------------
        sentence_vectors[n] = torch.from_numpy(copy.deepcopy(v)).to(torch.long)
        n += 1
    return sentence_vectors

def read_data(filepath, vec, tokenizer_use_lowercase, label_map):
    with open(filepath, 'r') as f:  # Read file into string
        lines_data = f.readlines()
    if len(label_map) > 0: # label map already determined
        fixed_map = 1 # map is fixed
    else:
        fixed_map = 0 # build new map
    sentences = []
    tags = []
    #tokenizer = get_tokenizer("basic_english")

    for line1 in lines_data[1:-1]: # Skip header
        split_index = line1.find(',')
        lebel_str = line1[0:split_index]
        text_str = line1[split_index+1:-1]
        vectors = tokenize_vectorize(text_str, vec, tokenizer_use_lowercase)
        if len(vectors) == 0:
            continue # sentences with no text don't add anything, only confuse
        sentences.append(vectors)
        if fixed_map == 0: # build map in the loop
            if not lebel_str in label_map:
                label_map[lebel_str] = len(label_map)
        tags.append(lebel_str)

    # For debug
    # --------------
    plt.hist(total_word_count_list, bins=range(0,60), histtype='step')
    plt.hist(total_oov_count_list, bins=range(0, 60),histtype='step')
    plt.show()
    plt.hist(np.array(total_oov_count_list)/np.array(total_word_count_list), bins=np.linspace(0, 1, 30), histtype='step')
    plt.show()
    #---------------
    return sentences, tags, label_map

    # Label map can come from an an argument or from the data. If label map argument is empty,
    # The learn the label map from the data.
class TextDataset(Dataset):
    def __init__(self, filepath, vec, label_map, max_allowed_seq, tokenizer_use_lowercase):
        sentences, tags, label_map= read_data(filepath, vec, tokenizer_use_lowercase, label_map)
        self.sentences_len = torch.Tensor([len(x) for x in sentences]).to(torch.long) # save the original lengths for the LSTM
        maxlen = min(max(np.array(self.sentences_len)), max_allowed_seq)
        self.sentences_len = torch.clamp(self.sentences_len, max = max_allowed_seq) # limit saved length values to maxlen
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
            if tag in label_map: # protect against a label in the test that was not in the training
                self.labels.append(label_map[tag]) # convert to class numbers
            else:
                self.labels.append(len(label_map)) # this is an impossible class number in training
                print(f'Found a label: {tag}, which was not in the training set')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.sentences_len[idx]