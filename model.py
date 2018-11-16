import torch


class Model(torch.nn.Module):
    def __init__(self, embedding_size, layer_size, sequence_size, output_size,
                 num_layers, dropout):
        super(Model, self).__init__()
        self.embedding_size = embedding_size
        self.layer_size = layer_size
        self.sequence_size = sequence_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(output_size, embedding_size)
        self.encoder = torch.nn.Linear(embedding_size, layer_size)
        self.gru = torch.nn.GRU(layer_size, layer_size // 2, batch_first=True,
                                bidirectional=True)
        self.linear = torch.nn.Linear(layer_size * 2, layer_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm1d(sequence_size)
        self.decoder = torch.nn.Linear(layer_size * sequence_size, output_size)
    
    def forward(self, x, use_cuda=False):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.leaky_relu(x)
        hidden = self.init_hidden(len(x), use_cuda)
        x_gru = x
        for _ in range(self.num_layers):
            x_gru, hidden = self.gru(x_gru, hidden)
            x_gru = self.dropout(x_gru)
            x = torch.cat([x, x_gru], -1)
            x = self.linear(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.leaky_relu(x)
        x = x.view(len(x), -1)
        x = self.decoder(x)
        return x

    def init_hidden(self, batch_size, use_cuda):
        hidden = torch.zeros(2, batch_size, self.layer_size // 2)
        torch.nn.init.uniform_(hidden, 0, 0.1)
        if use_cuda:
            return hidden.cuda()
        return hidden
