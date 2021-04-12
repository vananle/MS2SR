from typing import List

import torch


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()


class GraphConvNet(torch.nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = torch.nn.Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = torch.nn.functional.dropout(h, self.dropout, training=self.training)
        return h


def parallel_rnn(seq, len, cell, h):
    output = []
    for i in range(len):
        _in = seq[..., i]
        h = cell(_in, h)

        output.append(h)
    output = torch.stack(output, dim=-1)  # [b, h, len]

    return output


class GCRINT(torch.nn.Module):
    def __init__(self, args):
        super(GCRINT, self).__init__()

        self.seq_len = args.seq_len_x
        self.nSeries = args.nSeries
        # self.nNodes = args.nNodes
        self.lstm_hidden = args.lstm_hidden
        self.gcn_indim = args.in_dim
        # self.gcn_hidden = args.gcn_hidden
        self.dropout = args.dropout
        self.num_layers = args.n_lstm
        self.device = args.device
        self.apt_size = args.apt_size
        self.residual_channels = args.residual_channels
        self.verbose = args.verbose

        self.fixed_supports = []

        self.start_conv = torch.nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                          out_channels=self.residual_channels,
                                          kernel_size=(1, 1))
        self.cat_feature_conv = torch.nn.Conv2d(in_channels=self.gcn_indim - 1,
                                                out_channels=self.residual_channels,
                                                kernel_size=(1, 1))

        self.cell_fw = torch.nn.ModuleList(
            [torch.nn.LSTM(input_size=self.residual_channels,
                           hidden_size=self.lstm_hidden, bias=True, batch_first=True, dropout=0.2, bidirectional=True)
             for _ in range(self.num_layers)])

        # # only first layer has backward LSTM
        # self.lstm_cell_bw = torch.nn.LSTMCell(input_size=self.residual_channels,
        #                                       hidden_size=self.lstm_hidden, bias=True)

        self.supports_len = len(self.fixed_supports)
        nodevecs = torch.randn(self.nSeries, self.apt_size), torch.randn(self.apt_size, self.nSeries)
        self.supports_len += 1
        self.nodevec1, self.nodevec2 = [torch.nn.Parameter(n.to(self.device), requires_grad=True) for n in nodevecs]

        self.graph_convs = torch.nn.ModuleList(
            [GraphConvNet(self.lstm_hidden * 2, self.residual_channels, self.dropout, support_len=self.supports_len)
             for _ in range(self.num_layers)])
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.residual_channels) for _ in range(self.num_layers)])

        self.end_conv_1 = torch.nn.Conv2d(self.residual_channels, self.lstm_hidden * 2, (1, 1), bias=True)
        self.end_conv_2 = torch.nn.Conv2d(self.lstm_hidden * 2, self.seq_len, (1, 1), bias=True)

        self.linear_out = torch.nn.Linear(in_features=int(self.seq_len / (2 ** (self.num_layers - 1))), out_features=1)

    def lstm_layer(self, x, cell):
        # input x [bs, rc, n, s]
        # output (torch) [bs, hidden, n, s]

        x = x.transpose(1, 3)  # [bs, s, n, rc]

        futures: List[torch.jit.Future[torch.Tensor]] = []
        for k in range(self.nSeries):
            futures.append(torch.jit.fork(cell, x[:, :, k, :]))

        outputs = []
        for future in futures:
            outputs.append(torch.jit.wait(future)[0])

        outputs = torch.stack(outputs, dim=2)  # [b, h, n, len]
        outputs = outputs.transpose(1, 3)
        return outputs

    def feature_concat(self, input_tensor, mask):
        """
        input_tensor: [bs, s, n, f]
        mask: [bs, s, n ,1]
        """
        x = torch.cat([input_tensor, mask], dim=-1)  # [b, s, n, 2]
        x = x.transpose(1, 3)  # [b, 2, n, s]

        if self.verbose:
            print('input x: ', x.shape)

        f1, f2 = x[:, [0], :, :], x[:, 1:, :, :]
        x1 = self.start_conv(f1)
        # x2 = torch.nn.functional.leaky_relu(self.cat_feature_conv(f2))
        x2 = self.cat_feature_conv(f2)
        x = x1 + x2  # [b, rc, n, s]

        return x

    def forward(self, input_tensor, mask):

        # Input: x [b, s, n]
        # w: mask [b, s, n]

        x = self.feature_concat(input_tensor, mask)  # [b, rc, n, s]
        # x_bw = self.feature_concat(input_tensor_bw, mask_bw)  # [b, rc, n, s]

        if self.verbose:
            print('After startconv x = ', x.shape)

        # calculate the current adaptive adj matrix once per iteration
        adp = torch.nn.functional.softmax(torch.nn.functional.relu(torch.mm(self.nodevec1, self.nodevec2)),
                                          dim=1)  # the learnable adj matrix
        adjacency_matrices = self.fixed_supports + [adp]

        if self.verbose:
            print('Adjmx: ', adjacency_matrices[0].shape)

        outputs = 0
        for l in range(self.num_layers):

            in_lstm = x  # [b, rc, n, s]
            len = in_lstm.size(-1)

            if self.verbose:
                print('layer {} input = {}'.format(l, in_lstm.shape))

            gcn_in = self.lstm_layer(in_lstm, self.cell_fw[l])  # fw lstm  [b, h, n, len]

            # if l == 0:
            #     in_lstm_bw = x_bw  # [b, rc, n, s]
            #
            #     gcn_in_bw = self.lstm_layer(in_lstm_bw, self.lstm_cell_bw)  # bw lstm layer [b, h, n, len]
            #     gcn_in_bw = torch.flip(gcn_in_bw, dims=[-1])  # flip bw output
            #     gcn_in = (gcn_in + gcn_in_bw) / 2.0  # combine 2 outputs

            gcn_in = torch.tanh(gcn_in)

            if self.verbose:
                print('gcn in = ', gcn_in.shape)

            graph_out = self.graph_convs[l](gcn_in, adjacency_matrices)  # [b, rc, n, len]
            if self.verbose:
                print('gcn out = ', graph_out.shape)

            try:
                outputs = outputs[:, :, :, -graph_out.size(3):]
            except:
                outputs = 0
            outputs = graph_out + outputs

            if self.verbose:
                print('skip outputs = ', outputs.shape)

            x = graph_out
            x = x[..., 0:len:2]  # [b, rc, n, s/2 ]
            x = self.bn[l](x)

            if self.verbose:
                print('---------------------------------')

        if self.verbose:
            print('final skip outputs = ', outputs.shape)

        outputs = torch.nn.functional.relu(outputs)  # [b, gcn_hidden, n, seq/L]
        outputs = self.end_conv_1(outputs)  # [b, h', n, s/L]
        outputs = torch.nn.functional.relu(outputs)  # [b, h', n, s/L]
        outputs = self.end_conv_2(outputs)  # [b, s, n, s/L]
        if self.verbose:
            print('outputs end_conv = ', outputs.shape)

        outputs = self.linear_out(outputs)  # [b, s, n, 1]
        outputs = outputs.squeeze(dim=-1)  # [b, s, n]

        if self.verbose:
            print('final outputs = ', outputs.shape)
            print('*******************************************************************')

        return outputs
