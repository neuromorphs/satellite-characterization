import torch
import numpy as np
import snntorch as snn
import norse



def gaussian_kernel(
        size: int, c: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
        domain: int = 8) -> torch.Tensor:
    ci = torch.linalg.inv(c)
    cd = torch.linalg.det(c)
    fraction = 1 / (2 * torch.pi * torch.sqrt(cd))
    a = torch.linspace(-domain, domain, size)
    xs, ys = torch.meshgrid(a, a, indexing="xy")
    xs = xs - x
    ys = ys - y
    coo = torch.stack([xs, ys], dim=2)
    b = torch.einsum("bimj,jk->bik", -coo.unsqueeze(2), ci)
    a = torch.einsum("bij,bij->bi", b, coo)
    return fraction * torch.exp(a / 2)


def covariance_matrix(
        sigma1: torch.Tensor, sigma2: torch.Tensor, phi: torch.Tensor) \
            -> torch.Tensor:
    lambda1 = torch.as_tensor(sigma1) ** 2
    lambda2 = torch.as_tensor(sigma2) ** 2
    phi = torch.as_tensor(phi)
    cxx = lambda1 * phi.cos() ** 2 + lambda2 * phi.sin() ** 2
    cxy = (lambda1 - lambda2) * phi.cos() * phi.sin()
    cyy = lambda1 * phi.sin() ** 2 + lambda2 * phi.cos() ** 2
    cov = torch.ones(2, 2, device=phi.device)
    cov[0][0] = cxx
    cov[0][1] = cxy
    cov[1][0] = cxy
    cov[1][1] = cyy
    return cov


def get_filter(kernel_size):
    count = 0
    kernels = []
    for i in range(4):
        for j in range(4):
            c = covariance_matrix(1, 4, count * 2 * np.pi / 32)
            kernels.append(
                gaussian_kernel(kernel_size, c, x=0, y=0))
            count+=1
    return kernels


class MotionDetectionGaussLifSNN(torch.nn.Module):

    def __init__(
              self, width: int, height: int, tau_mem_1: float, tau_mem_2: float,
              tau_mem_3: float, v_th_1: float = 1., v_th_2: float = 1.,
              train_mem: bool = False, learn_threshold: bool = False):
        super().__init__()

        self.f_target1 = 1
        self.f_target2 = 1
        sg = snn.surrogate.fast_sigmoid(slope=50)
        out_shape = torch.tensor([width, height], dtype=int)
        print(out_shape)

        # Conv - LIF
        self.kernel_size = 9
        self.kernels = torch.stack(get_filter(self.kernel_size)).unsqueeze(1)
        print(self.kernels.shape)
        self.lif1 = snn.Leaky(
            beta=tau_mem_1, threshold=v_th_1, output=True, spike_grad=sg,
            learn_beta=train_mem, learn_threshold=learn_threshold,
            reset_mechanism="zero")

        # Hidden LIF
        self.linear1 = torch.nn.Linear(
            16 * int(out_shape[0]) * int(out_shape[1]), 200, bias=False)
        self.lif2 = snn.Leaky(
            beta=tau_mem_2, threshold=v_th_2, output=True, spike_grad=sg,
            learn_beta=train_mem, learn_threshold=learn_threshold,
            reset_mechanism="zero")

        # ff block
        self.linear2 = torch.nn.Linear(200, 1, bias=False)
        self.li = snn.Leaky(beta=tau_mem_3, threshold=100000)

    def regularize(self):
        reg_loss = 0.
        reg_loss += torch.mean((self.f_target1 - self.z1.sum(1))**2)
        reg_loss += torch.mean((self.f_target2 - self.z2.sum(1))**2)
        return reg_loss

    def forward(self, x):
        self.lif1.reset_mem()
        self.lif1.beta.data = torch.min(torch.tensor(0.999), self.lif1.beta.data)
        self.lif2.reset_mem()
        self.lif2.beta.data = torch.min(torch.tensor(0.999), self.lif2.beta.data)
        self.li.reset_mem()
        self.li.beta.data = torch.min(torch.tensor(0.999), self.li.beta.data)

        zs1, zs2, vs1, vs2, ys = [], [], [], [], []
        for ts in range(x.shape[1]):
            # lif
            g1 = torch.nn.functional.conv2d(
                x[:, ts], self.kernels.to(x.device), stride=1,
                padding=int(self.kernel_size//2))
            z1, v1 = self.lif1(g1)
            zs1.append(z1)
            vs1.append(v1)

            g2 = self.linear1(torch.flatten(z1, start_dim=1))
            z2, v2 = self.lif2(g2)
            zs2.append(z2)
            vs2.append(v2)

            # ff block
            g3 = self.linear2(z2)
            _, y = self.li(g3)
            ys.append(y)

        self.z1 = torch.stack(zs1, dim=0).transpose(1, 0)
        self.z2 = torch.stack(zs2, dim=0).transpose(1, 0)
        self.v1 = torch.stack(vs1, dim=0).transpose(1, 0)
        self.v2 = torch.stack(vs2, dim=0).transpose(1, 0)
        self.y = torch.stack(ys, dim=0).transpose(1, 0)

        return self.y