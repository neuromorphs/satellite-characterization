import torch


class MotionDetectionLSTM(torch.nn.Module):

    def __init__(
              self, width: int, height: int, input_pooling: int):
        super().__init__()

        kernel_size1 = 5
        kernel_size2 = 5
        pooling_size = 4
        stride = 1

        out_shape = torch.tensor([width, height], dtype=int)

        # Down sample 
        self.pool0 = torch.nn.AvgPool2d(input_pooling)
        out_shape = (out_shape / input_pooling).to(int)

        # conv block 1
        self.conv1 = torch.nn.Conv2d(
            1, 8, kernel_size1, padding=int(kernel_size1//2), stride=stride)
        self.pool1 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.relu1 = torch.nn.ReLU()

        # conv block 2
        self.conv2 = torch.nn.Conv2d(
            8, 32, kernel_size2, padding=int(kernel_size2//2), stride=stride)
        self.pool2 = torch.nn.AvgPool2d(pooling_size)
        out_shape = (out_shape / stride / pooling_size).to(int)
        self.relu2 = torch.nn.ReLU()

        # lstm
        self.lstm = torch.nn.LSTM(
            input_size=32 * int(out_shape[0]) * int(out_shape[1]),
            hidden_size=100)

        self.linear2 = torch.nn.Linear(100, 1, bias=False)

    def forward(self, x):
        zs1, zs2, zs3, ys = [], [], [], []
        for ts in range(x.shape[1]):

            # Down sample
            x0 = self.pool0(x[:, ts])

            # conv block 1
            g1 = self.conv1(x0)
            z1 = self.relu1(g1)
            p1 = self.pool1(z1)
            zs1.append(z1)

            # conv block 2
            g2 = self.conv2(p1)
            z2 = self.relu2(g2)
            p2 = self.pool2(z2)
            zs2.append(z2)

            # lif
            z3, _ = self.lstm(torch.flatten(p2, start_dim=1))
            zs3.append(z3)

            # ff block
            y = self.linear2(z3)
            ys.append(y)

        self.z1 = torch.stack(zs1, dim=0).transpose(1, 0)
        self.z2 = torch.stack(zs2, dim=0).transpose(1, 0)
        self.z3 = torch.stack(zs3, dim=0).transpose(1, 0)
        self.y = torch.stack(ys, dim=0).transpose(1, 0)

        return self.y