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
        self.linear = torch.nn.Linear(layer_size * 2, layer_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm1d(sequence_size)
        self.decoder = torch.nn.Linear(layer_size * sequence_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dropout(x)
        x_gru = x
        for _ in range(self.num_layers):
            x_gru, self.hidden = self.gru(x_gru, self.hidden)
            x = torch.cat([x, x_gru], -1)
            x = self.linear(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.leaky_relu(x)
            x = self.dropout(x)
        x = x.view(len(x), -1)
        x = self.decoder(x)
        return x

    def init_hidden(self, batch_size, init_range=0.1):
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(2, batch_size, self.layer_size // 2)
        torch.nn.init.uniform_(self.hidden, 0, init_range)
