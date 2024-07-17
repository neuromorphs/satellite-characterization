import torch
import snntorch as snn


class MotionTrackerDynamicSNN(torch.nn.Module):

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
        self.relu1 = torch.Relu()

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            4, 20, kernel_size, padding=int(kernel_size//2), stride=stride)
        self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = out_shape / stride / pooling_size
        self.relu2 = torch.Relu()

        # conv block 3
        self.conv3 = torch.nn.Conv2d(
            20, 40, kernel_size, padding=int(kernel_size//2), stride=stride)
        self.pool3 = torch.nn.AvgPool2d(pooling_size)
        out_shape = out_shape / stride / pooling_size
        self.relu3 = torch.Relu()

        # ff block
        self.linear = torch.nn.Linear(
            20 * int(out_shape[0]) * int(out_shape[1]), 2, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, threshold=100000)

    def forward(self, x):
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.li.reset_mem()

        as1, as2, as3 = [], [], []
        zs1, vs1, zs2, vs2, ys = [], [], [], [], []
        for ts in range(x.shape[0]):
            # stationary conv
            g1 = self.conv1(x[ts])
            p1 = self.pool1(g1)
            a1 = self.relu1(p1)
            as1.append(a1)

            g2 = self.conv2(a1)
            p2 = self.pool2(g2)
            a2 = self.relu2(p2)
            as2.append(a2)

            g3 = self.conv3(a2)
            p3 = self.pool3(g3)
            a3 = self.relu3(p3)
            as3.append(a3)

            # SNN dynamic block
            g4 = self.linear(torch.flatten(a3, start_dim=1))
            z1, v1 = self.lif1(g4)

            g5 = self.linear(z1)
            z2, v2 = self.lif1(g4)

            zs1.append(z1)
            vs1.append(v1)
            zs2.append(z2)
            vs2.append(v2)

            # ff block
            g6 = self.linear(g5)
            _, y = self.li(g6)
            ys.append(y)

        return torch.stack(ys, dim=0)


class MotionTrackerStaticSNN(torch.nn.Module):

    def __init__(
              self, height: int, width: int, tau_mem_1: float, tau_mem_2: float,
              tau_mem_3: float, train_mem: bool = False):
        super().__init__()
        kernel_size = 5
        pooling_size = 2
        stride = 2

        out_shape = torch.tensor([height, width])
        sg = snn.surrogate.fast_sigmoid(slope=50)

        # conv block 1
        self.conv1 = torch.nn.Conv2d(
            2, 4, kernel_size, padding=int(kernel_size//2), stride=stride)
        #self.pool1 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride).to(int)
        self.lif1 = snn.Leaky(
            beta=tau_mem_1, output=True, spike_grad=sg, learn_beta=train_mem)

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            4, 20, kernel_size, padding=int(kernel_size//2), stride=stride)
        #self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride).to(int)
        self.lif2 = snn.Leaky(
            beta=tau_mem_2, output=True, spike_grad=sg, learn_beta=train_mem)
        
        # conv block 3
        self.conv3 = torch.nn.Conv2d(
            20, 1, kernel_size, padding=int(kernel_size//2), stride=1)
        #self.pool3 = torch.nn.AvgPool2d(pooling_size)
        #out_shape = (out_shape / stride).to(int)
        #self.relu3 = torch.Relu()
        # ff block
        #self.linear = torch.nn.Linear(
        #    20 * int(out_shape[0]) * int(out_shape[1]), 1, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, reset_mechanism='none')

    def regularize(self):
        reg_loss = 0.
        reg_loss += torch.mean(self.z1.sum(1)**2)
        reg_loss += torch.mean(self.z2.sum(1)**2)
        return reg_loss
    
    def forward(self, x):
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.li.reset_mem()

        as1, as2, as3 = [], [], []
        zs1, vs1, zs2, vs2, ys = [], [], [], [], []
        for ts in range(x.shape[1]):
            x_t = x[:,ts]
            # stationary conv
            g1 = self.conv1(x_t)
            z1, v1 = self.lif1(g1)
            #p1 = self.pool1(g1)
            #as1.append(a1)

            g2 = self.conv2(z1)
            z2, v2 = self.lif2(g2)
            #p2 = self.pool2(g2)
            #as2.append(a2)

            # SNN dynamic block
            #g4 = self.linear(torch.flatten(a3, start_dim=1))
            g3 = self.conv3(z2)
            #z3, v3 = self.lif1(g3)
            #p3 = self.conv3(z3)

            #l4 = self.linear(z2)
            _, y = self.li(g3)

            zs1.append(z1)
            vs1.append(v1)
            zs2.append(z2)
            vs2.append(v2)

            ys.append(y)

        return torch.stack(ys, dim=0)
