import torch
import numpy as np
import snntorch as snn


class MotionDetectionLifSNN(torch.nn.Module):

    def __init__(
              self, width: int, height: int, input_pooling: int,
              tau_mem_1: float, tau_mem_2: float, train_mem: bool = False):
        super().__init__()

        self.f_target = 3
        out_shape = torch.tensor([width, height], dtype=int)
        sg = snn.surrogate.fast_sigmoid(slope=50)

        # Down sample 
        self.pool0 = torch.nn.AvgPool2d(input_pooling)
        out_shape = (out_shape / input_pooling).to(int)
        # conv block 2
        self.linear1 = torch.nn.Linear(
            int(out_shape[0]) * int(out_shape[1]), 200, bias=False)
        self.lif = snn.Leaky(
            beta=tau_mem_1, output=True, spike_grad=sg, learn_beta=train_mem)
        # ff block
        self.linear2 = torch.nn.Linear(200, 1, bias=False)
        self.li = snn.Leaky(beta=tau_mem_2, threshold=100000)

    def regularize(self):
        reg_loss = 0.
        reg_loss += torch.mean((self.f_target - self.z.sum(1))**2)
        return reg_loss

    def forward(self, x):
        self.lif.reset_mem()
        self.lif.beta.data = torch.min(torch.tensor(0.999), self.lif.beta.data)
        self.li.reset_mem()
        self.li.beta.data = torch.min(torch.tensor(0.999), self.li.beta.data)

        zs, vs, ys = [], [], []
        for ts in range(x.shape[1]):
            # Down sample
            x0 = self.pool0(x[:, ts])
            # lif
            g1 = self.linear1(torch.flatten(x0, start_dim=1))
            z, v = self.lif(g1)
            zs.append(z)
            vs.append(v)
            # ff block
            g2 = self.linear2(z)
            _, y = self.li(g2)
            ys.append(y)

        self.z = torch.stack(zs, dim=0).transpose(1, 0)
        self.v = torch.stack(vs, dim=0).transpose(1, 0)
        self.y = torch.stack(ys, dim=0).transpose(1, 0)

        return self.y