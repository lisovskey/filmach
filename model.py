import torch


class LayerNormGRUCell(torch.nn.modules.rnn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, ln_preact=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias,
                                               num_chunks=3)

        self.ln_preact = ln_preact
        if ln_preact:
            self.ln_ih = torch.nn.LayerNorm(3*self.hidden_size)
            self.ln_hh = torch.nn.LayerNorm(3*self.hidden_size)
        self.ln_in = torch.nn.LayerNorm(self.hidden_size)
        self.ln_hn = torch.nn.LayerNorm(self.hidden_size)

    def forward(self, x, hx=None):
        self.check_forward_input(x)
        if hx is None:
            hx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(x, hx)
        
        ih = x @ self.weight_ih.t() + self.bias_ih
        hh = hx @ self.weight_hh.t() + self.bias_hh
        if self.ln_preact:
            ih = self.ln_ih(ih)
            hh = self.ln_hh(hh)

        i_r, i_z, i_n = ih.chunk(3, dim=1)
        h_r, h_z, h_n = hh.chunk(3, dim=1)
        i_n = self.ln_in(i_n)
        h_n = self.ln_hn(h_n)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r*h_n)
        h = (1 - z)*n + z*hx

        return h


class LayerNormGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0):
        super(LayerNormGRU, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.cells = torch.nn.ModuleList([
            LayerNormGRUCell(input_size if i == 0 else hidden_size,
                             hidden_size, bias)
            for i in range(self.num_layers)
        ])

    def forward(self, x, hx):
        seq_dim = 1 if self.batch_first else 0
        for i, cell in enumerate(self.cells):
            y = []
            for xc in x.chunk(x.size(seq_dim), dim=seq_dim):
                hx[i] = cell(xc.squeeze(seq_dim), hx[i].clone())
                y.append(hx[i].unsqueeze(0))
            x = torch.stack(y, dim=seq_dim + 1).squeeze(0)
            if self.dropout > 0 and i != self.num_layers - 1:
                x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x, hx


class Model(torch.nn.Module):
    def __init__(self, embedding_size, layer_size, sequence_size, output_size,
                 num_layers=1, dropout=0):
        super(Model, self).__init__()

        self.embedding_size = embedding_size
        self.layer_size = layer_size
        self.sequence_size = sequence_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(output_size, embedding_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = LayerNormGRU(embedding_size, layer_size, num_layers,
                                batch_first=True, dropout=dropout)
        self.decoder = torch.nn.Linear(
            (layer_size + embedding_size) * sequence_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x_gru, self.hidden = self.gru(x, self.hidden)
        x = torch.cat([x, x_gru], -1)
        x = x.view(len(x), -1)
        x = self.decoder(x)
        return x

    def init_hidden(self, batch_size=1, init_range=0.1):
        weight = next(self.parameters())
        self.hidden = weight.new_zeros(self.num_layers, batch_size,
                                       self.layer_size)
