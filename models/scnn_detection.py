import torch
import numpy as np
import snntorch as snn


class MotionDetectionSNN(torch.nn.Module):

    def __init__(
              self, width: int, height: int, tau_mem_1: float, tau_mem_2: float,
              tau_mem_3: float):
        super().__init__()
        kernel_size = 5
        pooling_size = 2
        stride = 4

        out_shape = torch.tensor([width, height])

        # conv block 1
        self.conv1 = torch.nn.Conv2d(
            2, 4, kernel_size, padding=int(kernel_size//2), stride=stride)
        self.pool1 = torch.nn.AvgPool2d(pooling_size)
        out_shape = out_shape / stride / pooling_size
        self.lif1 = snn.Leaky(beta=tau_mem_1, output=True)

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            4, 20, kernel_size, padding=int(kernel_size//2), stride=stride)
        self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = out_shape / stride / pooling_size
        self.lif2 = snn.Leaky(beta=tau_mem_2, output=True)

        # ff block
        self.linear = torch.nn.Linear(
            20 * int(out_shape[0]) * int(out_shape[1]), 1, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, threshold=100000)

    def forward(self, x):
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.li.reset_mem()

        zs1, vs1, zs2, vs2, ys = [], [], [], [], []
        for ts in range(x.shape[1]):
            # conv block 1
            g1 = self.conv1(x[:, ts])
            p1 = self.pool1(g1)
            z1, v1 = self.lif1(p1)
            zs1.append(z1)
            vs1.append(v1)

            # conv block 2
            g2 = self.conv2(z1)
            p2 = self.pool2(g2)
            z2, v2 = self.lif2(p2)
            zs2.append(z2)
            vs2.append(v2)

            # ff block
            g3 = self.linear(torch.flatten(z2, start_dim=1))
            _, y = self.li(g3)
            ys.append(y)

        return torch.stack(ys, dim=0).transpose(1, 0)