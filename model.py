import torch.nn as nn

class EmbeddingTest(nn.Module):
    """Deep Knowledge tracing model"""

    def __init__(self, rnn_type, input_size, hidden_size, num_skills, nlayers, dropout=0.6, tie_weights=False):
        super(EmbeddingTest, self).__init__()
        # self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_skills, hidden_size)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(hidden_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        # self.decoder = nn.Linear(hidden_size, num_skills)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_skills)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            pass
            # if hidden_size != input_size:
            #     raise ValueError('When using the tied flag, nhid must be equal to emsize')
            # self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.05
        #self.encoder.weight.data.uniform_(-initrange, initrange)
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
        input_emb_transform = torch.where(input==torch.LongTensor([1]))[2] % (self.num_skills-1)
        input_emb_transform = input_emb_transform.view(input.shape[0], input.shape[1], -1)
        output, hidden = self.rnn(input, hidden)
        # output = self.drop(output)
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
