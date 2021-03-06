import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# This is a class that uses external embedding- not in use in the final version
class LSTM(nn.Module):

    def __init__(self, input_size=300, hidden_size=300, num_classes = 3, num_layers = 1, bidir = True, drop_rate = 0.5):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidir)
        self.drop = nn.Dropout(p=drop_rate)
        # For bi-directional the fc dimension should be doubled (due to concatenation of the 2 directions)
        self.fc = nn.Linear((int(bidir)+1)*hidden_size, num_classes)

    def forward(self, input, data_lengths):

        # We use input which is already embeddings so no need for an embedding layer
        packed_input = pack_padded_sequence(input, data_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), data_lengths - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out = torch.cat((out_forward, out_reverse), 1)
        out = self.fc(self.drop(out))
        out = torch.squeeze(out, 1) # No need for a softmax layer because we use cross-entropy as loss function

        return out

# This is a class that uses internal embeddings, used in the final version
# We initialize with embeddings which are pre-trained (GloVe) and allow fine tunning
class LSTM_emb(nn.Module):
    def __init__(self, embs, feeze_embs, hidden_size=300, num_classes = 3, num_layers = 1, bidir = True, drop_rate = 0.5):
        super(LSTM_emb, self).__init__()
        # initialize the embedding layer with the pre-trained vectors
        self.vocab_size = embs.shape[0]
        self.embedding_dim = embs.shape[1]
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs).float(),
                                                            freeze=feeze_embs)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidir)
        self.drop = nn.Dropout(p=drop_rate)
        # For bi-directional the fc dimension should be doubled (due to concatenation of the 2 directions)
        self.fc = nn.Linear((int(bidir)+1)*hidden_size, num_classes)

    def forward(self, input, data_lengths):
        embed_out = self.embedding(torch.squeeze(input,-1))
        packed_input = pack_padded_sequence(embed_out, data_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), data_lengths - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out = torch.cat((out_forward, out_reverse), 1)
        out = self.fc(self.drop(out))
        out = torch.squeeze(out, 1) # No need for a softmax layer because we use cross-entropy as loss function

        return out