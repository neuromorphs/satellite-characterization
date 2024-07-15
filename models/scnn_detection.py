import torch
import numpy as np
import snntorch as snn


class MotionDetectionSNN(torch.nn.Module):

    def __init__(
              self, width: int, height: int, input_pooling: int,
              tau_mem_1: float, tau_mem_2: float, tau_mem_3: float,
              record: bool = False, train_mem: bool = False):
        super().__init__()

        self.record = record
        kernel_size1 = 5
        kernel_size2 = 5
        pooling_size = 4
        stride = 1

        out_shape = torch.tensor([width, height], dtype=int)

        sg = snn.surrogate.fast_sigmoid(slope=50)

        # Down sample 
        self.pool0 = torch.nn.AvgPool2d(input_pooling)
        out_shape = (out_shape / input_pooling).to(int)

        # conv block 1
        self.conv1 = torch.nn.Conv2d(
            1, 8, kernel_size1, padding=int(kernel_size1//2), stride=stride)
        self.pool1 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.lif1 = snn.Leaky(
            beta=tau_mem_1, output=True, spike_grad=sg, learn_beta=train_mem)

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            8, 8, kernel_size2, padding=int(kernel_size2//2), stride=stride)
        self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.lif2 = snn.Leaky(
            beta=tau_mem_2, output=True, spike_grad=sg, learn_beta=train_mem)
        print(out_shape)

        # ff block
        self.linear = torch.nn.Linear(
            8 * int(out_shape[0]) * int(out_shape[1]), 1, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, threshold=100000)

    def regularize(self):
        reg_loss = 0.
        reg_loss += torch.mean(self.z1.sum(1)**2)
        reg_loss += torch.mean(self.z2.sum(1)**2)
        return reg_loss

    def forward(self, x):
        self.lif1.reset_mem()
        self.lif1.beta.data = torch.max(torch.tensor(0.99), self.lif1.beta.data)
        self.lif2.reset_mem()
        self.lif2.beta.data = torch.max(torch.tensor(0.99), self.lif2.beta.data)
        self.li.reset_mem()

        zs1, vs1, zs2, vs2, ys = [], [], [], [], []
        for ts in range(x.shape[1]):

            # Down sample
            x0 = self.pool0(x[:, ts])

            # conv block 1
            g1 = self.conv1(x0)
            z1, v1 = self.lif1(g1)
            p1 = self.pool1(z1)
            zs1.append(z1)
            vs1.append(v1)

            # conv block 2
            g2 = self.conv2(p1)
            z2, v2 = self.lif2(g2)
            p2 = self.pool2(z2)
            zs2.append(z2)
            vs2.append(v2)

            # ff block
            g3 = self.linear(torch.flatten(p2, start_dim=1))
            _, y = self.li(g3)
            ys.append(y)

        self.z1 = torch.stack(zs1, dim=0).transpose(1, 0)
        self.z2 = torch.stack(zs2, dim=0).transpose(1, 0)
        if self.record:
            self.v1 = torch.stack(vs1, dim=0).transpose(1, 0)
            self.v2 = torch.stack(vs2, dim=0).transpose(1, 0)
            self.y = torch.stack(ys, dim=0).transpose(1, 0)

        return torch.stack(ys, dim=0).transpose(1, 0)