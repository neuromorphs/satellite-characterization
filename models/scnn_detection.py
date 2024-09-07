import torch
import numpy as np
import snntorch as snn

from models.gauss_lif_detection import get_filter


class MotionDetectionSNN(torch.nn.Module):

    def __init__(
              self, width: int, height: int, input_pooling: int,
              tau_mem_1: float, tau_mem_2: float, tau_mem_3: float,
              v_th_1: float, v_th_2: float, v_th_3: float,
              record: bool = False, train_mem: bool = False,
              learn_threshold: bool = False):
        super().__init__()

        self.record = record
        kernel_size1 = 5
        kernel_size2 = 5
        pooling_size = 2
        stride = 1
        self.f_target_conv1 = 3
        self.f_target_conv2 = 3
        self.f_target_lif = 3

        out_shape = torch.tensor([width, height], dtype=int)

        sg = snn.surrogate.fast_sigmoid(slope=50)

        # Down sample 
        #self.pool0 = torch.nn.AvgPool2d(input_pooling)
        out_shape = (out_shape / input_pooling).to(int)

        # conv block 1
        self.conv1 = torch.nn.Conv2d(
            1, 8, kernel_size1, padding=int(kernel_size1//2), stride=stride,
            bias=False)
        self.pool1 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.lif1 = snn.Leaky(
            beta=tau_mem_1, threshold=v_th_1, output=True, spike_grad=sg,
            learn_beta=train_mem, learn_threshold=learn_threshold,
            reset_mechanism="zero")

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            8, 32, kernel_size2, padding=int(kernel_size2//2), stride=stride,
            bias=False)
        self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.lif2 = snn.Leaky(
            beta=tau_mem_2, threshold=v_th_2, output=True, spike_grad=sg,
            learn_beta=train_mem, learn_threshold=learn_threshold,
            reset_mechanism="zero")

        # conv block 2
        self.linear1 = torch.nn.Linear(
            32 * int(out_shape[0]) * int(out_shape[1]), 100, bias=False)
        self.lif3 = snn.Leaky(
            beta=tau_mem_2, threshold=v_th_3, output=True, spike_grad=sg,
            learn_beta=train_mem, learn_threshold=learn_threshold,
            reset_mechanism="zero")

        # ff block
        self.linear2 = torch.nn.Linear(100, 1, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, threshold=100000)

    def regularize(self):
        reg_loss = 0.
        reg_loss += torch.mean((self.f_target_conv1 - self.z1.sum(1))**2)
        reg_loss += torch.mean((self.f_target_conv2 - self.z2.sum(1))**2)
        reg_loss += torch.mean((self.f_target_lif - self.z3.sum(1))**2)
        return reg_loss

    def forward(self, x):
        self.lif1.reset_mem()
        self.lif1.beta.data = torch.min(torch.tensor(0.99), self.lif1.beta.data)
        self.lif2.reset_mem()
        self.lif2.beta.data = torch.min(torch.tensor(0.99), self.lif2.beta.data)
        self.lif3.reset_mem()
        self.lif3.beta.data = torch.min(torch.tensor(0.99), self.lif3.beta.data)
        self.li.reset_mem()

        zs1, vs1, zs2, vs2, zs3, vs3, ys = [], [], [], [], [], [], []
        #for ts in range(x.shape[1]):

        # Down sample
        x0 = x #self.pool0(x[:, ts])

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

        # lif
        g3 = self.linear1(torch.flatten(p2, start_dim=1))
        z3, v3 = self.lif3(g3)
        zs3.append(z3)
        vs3.append(v3)

        # ff block
        print("LINEAR NUMBER 2 COMING")
        g4 = self.linear2(z3)
        _, y = self.li(g4)
        #ys.append(y)

        self.z1 = torch.stack(zs1, dim=0).transpose(1, 0)
        self.z2 = torch.stack(zs2, dim=0).transpose(1, 0)
        self.z3 = torch.stack(zs3, dim=0).transpose(1, 0)
        if self.record:
            self.v1 = torch.stack(vs1, dim=0).transpose(1, 0)
            self.v2 = torch.stack(vs2, dim=0).transpose(1, 0)
            self.v3 = torch.stack(vs3, dim=0).transpose(1, 0)
            self.y = torch.stack(ys, dim=0).transpose(1, 0)

        return y #torch.stack(ys, dim=0).transpose(1, 0)


class MotionDetectionDualSNN(torch.nn.Module):

    def __init__(
              self, width: int, height: int, input_pooling: int,
              tau_mem_1: float, tau_mem_2: float, tau_mem_3: float,
              record: bool = False, train_mem: bool = False,
              learn_threshold: bool = False):
        super().__init__()

        self.record = record
        kernel_size1 = 5
        kernel_size2 = 5
        pooling_size = 4
        stride = 1
        self.f_target_conv1 = 3
        self.f_target_conv2 = 3
        self.f_target_lif = 3

        out_shape = torch.tensor([width, height], dtype=int)

        sg = snn.surrogate.fast_sigmoid(slope=50)

        # Down sample 
        self.pool0 = torch.nn.AvgPool2d(input_pooling)
        out_shape = (out_shape / input_pooling).to(int)

        # conv block 1
        self.conv1 = torch.nn.Conv2d(
            1, 8, kernel_size1, padding=int(kernel_size1//2), stride=stride,
            bias=False)
        self.pool1 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.lif1 = snn.Leaky(
            beta=tau_mem_1, output=True, spike_grad=sg, learn_beta=train_mem,
            learn_threshold=learn_threshold)

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            8, 32, kernel_size2, padding=int(kernel_size2//2), stride=stride,
            bias=False)
        self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.lif2 = snn.Leaky(
            beta=tau_mem_2, output=True, spike_grad=sg, learn_beta=train_mem,
            learn_threshold=learn_threshold)

        # conv block 2
        self.linear1 = torch.nn.Linear(
            32 * int(out_shape[0]) * int(out_shape[1]), 100, bias=False)
        self.lif3 = snn.Leaky(
            beta=tau_mem_2, output=True, spike_grad=sg, learn_beta=train_mem,
            learn_threshold=learn_threshold)

        # ff block
        self.linear2 = torch.nn.Linear(100, 2, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, threshold=100000)

    def regularize(self):
        reg_loss = 0.
        reg_loss += torch.mean((self.f_target_conv1 - self.z1.sum(1))**2)
        reg_loss += torch.mean((self.f_target_conv2 - self.z2.sum(1))**2)
        reg_loss += torch.mean((self.f_target_lif - self.z3.sum(1))**2)
        return reg_loss

    def forward(self, x):
        self.lif1.reset_mem()
        self.lif1.beta.data = torch.min(torch.tensor(0.99), self.lif1.beta.data)
        self.lif2.reset_mem()
        self.lif2.beta.data = torch.min(torch.tensor(0.99), self.lif2.beta.data)
        self.lif3.reset_mem()
        self.lif3.beta.data = torch.min(torch.tensor(0.99), self.lif3.beta.data)
        self.li.reset_mem()

        zs1, vs1, zs2, vs2, zs3, vs3, ys = [], [], [], [], [], [], []
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

            # lif
            g3 = self.linear1(torch.flatten(p2, start_dim=1))
            z3, v3 = self.lif3(g3)
            zs3.append(z3)
            vs3.append(v3)

            # ff block
            g4 = self.linear2(z3)
            _, y = self.li(g4)
            ys.append(y)

        self.z1 = torch.stack(zs1, dim=0).transpose(1, 0)
        self.z2 = torch.stack(zs2, dim=0).transpose(1, 0)
        self.z3 = torch.stack(zs3, dim=0).transpose(1, 0)
        if self.record:
            self.v1 = torch.stack(vs1, dim=0).transpose(1, 0)
            self.v2 = torch.stack(vs2, dim=0).transpose(1, 0)
            self.v3 = torch.stack(vs3, dim=0).transpose(1, 0)
            self.y = torch.stack(ys, dim=0).transpose(1, 0)

        return torch.stack(ys, dim=0).transpose(1, 0)