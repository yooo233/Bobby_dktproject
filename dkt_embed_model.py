import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

"""
Baseline: RNN model
1. LSTM + Attention (Global attention, Self attention, causal attention)
"2. Initialization function", for your understanding
3. Transformer with GPU. Google Colab
4. Add forgetting behavior
"""

# Deep KT with Embedding

class DKTEmbed(nn.Module):
    """Deep Knowledge tracing model"""
    def __init__(self, rnn_type, input_size, hidden_size, num_skills, nlayers, dropout=0.6, embedding_size=128, max_length=1219):

        super(DKTEmbed, self).__init__()
        # self.drop = nn.Dropout(dropout)
        #self.encoder = nn.Embedding(ntoken, ninp)

        logging.info("Running, DKT_Embed Model")
        if rnn_type in ['LSTM', 'GRU']:
            # self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            # self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
            self.rnn = nn.RNN(embedding_size, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        #self.decoder = nn.Linear(hidden_size, num_skills)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_skills)
        
        

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        # Create embedding to represent each problem
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.num_skills = num_skills
        self.input_size = input_size
        self.embedding = nn.Embedding(num_skills, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.05
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.zero_()
        #self.decoder.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.zero_()
        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer2.bias.data.zero_()
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.layer3.bias.data.zero_()
        self.layer3.weight.data.uniform_(-initrange, initrange)
        # TODO: 分析initialization function.
        # Normal distribution init. 
        # torch.nn.init.xavier_uniform_(tensor, gain=1.0)

    def forward(self, input, hidden):
        # input = self.drop(input)
        # output = self.drop(output)
        # decoded = self.decoder(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        input_emb_transform = torch.where(input==torch.LongTensor([1]))[2] % (self.num_skills-1)
        input_emb_transform = input_emb_transform.view(input.shape[0], input.shape[1], -1)

        embedded_input = self.embedding(input_emb_transform).view(input.shape[0], input.shape[1], -1)
        embedded_input = self.dropout(embedded_input)
        output, hidden = self.rnn(embedded_input, hidden)
        
        decoded = self.layer1(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        decoded = self.layer2(decoded)
        decoded = self.layer3(decoded)
        

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)