import math

import numpy as np
import torch
from symmetrizer.symmetrizer.nn import BasisLinear
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu

class EquivariantGRUCell(nn.Module):
    def __init__(self, input_length, hidden_length, input_group=None, hidden_group=None):
        super(EquivariantGRUCell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        if input_group is None:  # for sanity check
            assert (hidden_group is None)
            self.linear_reset_w1 = nn.Linear(input_length, hidden_length, bias=True)
            self.linear_reset_r1 = nn.Linear(hidden_length, hidden_length, bias=True)

            self.linear_reset_w2 = nn.Linear(input_length, hidden_length, bias=True)
            self.linear_reset_r2 = nn.Linear(hidden_length, hidden_length, bias=True)

            # update gate components
            self.linear_gate_w3 = nn.Linear(input_length, hidden_length, bias=True)
            self.linear_gate_r3 = nn.Linear(hidden_length, hidden_length, bias=True)
        else:
            self.linear_reset_w1 = BasisLinear(input_length, hidden_length, input_group, bias=True)
            self.linear_reset_r1 = BasisLinear(hidden_length, hidden_length, hidden_group, bias=True)

            self.linear_reset_w2 = BasisLinear(input_length, hidden_length, input_group, bias=True)
            self.linear_reset_r2 = BasisLinear(hidden_length, hidden_length, hidden_group, bias=True)

            # update gate components
            self.linear_gate_w3 = BasisLinear(input_length, hidden_length, input_group, bias=True)
            self.linear_gate_r3 = BasisLinear(hidden_length, hidden_length, hidden_group, bias=True)

        self.activation_1 = nn.Sigmoid()
        self.activation_2 = nn.Sigmoid()
        self.activation_3 = nn.Tanh()

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_1(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_2( h_2 + x_2)
        return z

    def update_component(self, x, h,r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.linear_gate_r3(h)
        gate_update = self.activation_3(x_3+h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1-z) * n + z * h

        return h_new

class EquivariantGRU(nn.Module):
    def __init__(self, input_size, hidden_size, input_group=None, hidden_group=None):
        super(EquivariantGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_group = input_group
        self.cell = EquivariantGRUCell(input_size, hidden_size, input_group, hidden_group)

    def forward(self, input):
        # outputs = [] Note: for efficiency, we don't collect the outputs as they aren't used in our setting
        if self.input_group is None:
            # Take all batch dimensions remove sequence length and replace input size by hidden size.
            h_t = ptu.zeros(list(input.shape[:-2]) + [self.hidden_size])
        else:
            # BasisLinear doesn't seem to be able to handle multiple batch dimensions
            old_dim = list(input.shape[:-2])
            input = input.reshape(-1, input.shape[-2], input.shape[-1])
            h_t = ptu.zeros(list(input.shape[:-2]) + [self.hidden_size, self.input_group.repr_size_out])

        for i, input_t in enumerate(input.chunk(input.size(-2), dim=-2)):
            if self.input_group is None:
                h_t = self.cell(input_t[..., 0, :], h_t)  # removing the time dimension that was split on
            else:
                # here, the dimension would also be removed but a new input size dimension would be added at the same
                # position
                h_t = self.cell(input_t, h_t)
            # outputs += [h_t]

        if self.input_group is not None:
            h_t = h_t.reshape(old_dim + list(h_t.shape[-2:]))

        # we also don't use prediction of the future although it should work
        # for _ in range(future):
        #     h_t = self.cell(input_t, h_t)
        # outputs = torch.stack(outputs, 1).squeeze(2)
        # return outputs
        return None, h_t[None, ...]  # None and added dimension to make same interface as for torch GRU


class Sequence(nn.Module):
    def __init__(self, custom=True):
        super(Sequence, self).__init__()
        if custom:
            print("Custom GRU cell implementation...")
            self.rnn1 = EquivariantGRUCell(1, 51)
            self.rnn2 = EquivariantGRUCell(51, 51)
        else:
            print("Official PyTorch GRU cell implementation...")
            self.rnn1 = nn.GRUCell(1, 51)
            self.rnn2 = nn.GRUCell(51, 51)

        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t = self.rnn1(input_t, h_t)
            h_t2 = self.rnn2(h_t, h_t2)

            output = self.linear(h_t2)
            outputs += [output]

        # if we should predict the future
        for i in range(future):
            h_t = self.rnn1(input_t, h_t)
            h_t2 = self.rnn2(h_t, h_t2)

            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class GRUPredictor(nn.Module):
    """
    Just for testing purposes, this adds a final fully connected layer to the sequence in order to e.g. predict
    parameters
    """
    def __init__(self, input_size, hidden_size, output_size, custom=True):
        super(GRUPredictor, self).__init__()
        if custom:
            self.gru = EquivariantGRU(input_size, hidden_size)
        else:  # for sanity checks
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, gru_out = self.gru(input)
        print(gru_out.shape)
        return self.linear(gru_out[0, ...])

def test_get_sin_params(custom=True, steps=20):
    res = 10.0  # points per unit
    L = 100  # sequence length
    N_train = 100
    N_test = 50
    N = N_train + N_test
    data = ptu.empty((N, L))
    data[:] = ptu.arange(L)[None, :] / res
    params = ptu.rand(N, 3)  # a*sin(bx+c)
    params[:, 0] = params[:, 0] * 5  # a in [0, 5)
    params[:, 1] = 1  # *= 5  # b in [0, 5)
    params[:, 2] = 0  # *= 2 * math.pi  # c in [0, 2pi)
    data = params[:, 0:1] * torch.sin(data * params[:, 1:2] + params[:, 2:3])
    data = data[:, :, None]  # add input dimension

    net = GRUPredictor(1, 50, 3, custom=custom).to(ptu.device)
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    print("="*10 + " parameters " + "="*10)
    sum_trainable = 0
    sum_nontrainable = 0
    for p in net.parameters():
        print(("trainable: " if p.requires_grad else "non-trainable: ") + str(p.numel()))
        if p.requires_grad:
            sum_trainable += p.numel()
        else:
            sum_nontrainable += p.numel()
    print("total trainable: " + str(sum_trainable) + ", total nontrainable: " + str(sum_nontrainable))
    print("=" * 32)
    optimizer = optim.LBFGS(net.parameters()) # optim.LBFGS(net.parameters())
    # begin to train
    for i in range(steps):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = net(data[N_test:, ...])
            loss = criterion(out, params[N_test:, ...])
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = net(data[:N_test, ...])
            loss = criterion(pred, params[:N_test, ...])
            print('test loss:', loss.item())
            for i in range(3):
                print("e.g." + str(pred[i, :]) + " for " + str(params[i, :]))

def train_and_test_sin(custom=True, steps=20):
    data = torch.load('traindata.pt')
    input = ptu.from_numpy(data[3:, :-100]).double()
    print(input.shape, input.device)
    target = ptu.from_numpy(data[3:, 100:]).double()
    test_input = ptu.from_numpy(data[:3, :-100]).double()
    test_target = ptu.from_numpy(data[:3, 100:]).double()

    seq = Sequence(custom=custom)  #.cuda()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # begin to train
    for i in range(steps):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict_' + ('custom_' if custom else 'torch_') + str(i) + '.png')
        plt.close()

def generate_sin_data():
    T = 20
    L = 1000
    N = 200

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    torch.save(data, open('traindata.pt', 'wb'))

if __name__ == "__main__":
    ptu.set_gpu_mode(True, 0)
    # generate_sin_data()
    # train_and_test_sin(steps=20, custom=True)
    test_get_sin_params()