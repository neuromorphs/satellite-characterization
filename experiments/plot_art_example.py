""" Train motion detection algoritm """
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from models.scnn_detection import MotionDetectionSNN, MotionDetectionGaussSNN
from models.gauss_lif_detection import MotionDetectionGaussLifSNN
from dataset import EgoMotionDataset
from matplotlib.animation import FuncAnimation, PillowWriter


# Membrane time constant
tau_mem1 = 0.9
tau_mem2 = 0.9
tau_mem3 = 0.95

n_kernel1 = 16
n_kernel2 = 8

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
dtype = torch.float

# Target size for tonic down sampling
tau_mem1 = 0.8
tau_mem2 = 0.8
tau_mem3 = 0.9

v_th1 = 0.1
v_th2 = 0.5
v_th3 = 0.4

period_sim = 1000
period = 100
width = 250
height = 138
target_size = [80, 120]


index=2


# Additional down sampling
input_avg_pooling = 1

# data
path = Path("./art_data/trained_gauss_scnn_no_learning")

# Fix random seed
np.random.seed(0)
torch.manual_seed(0)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

  
def leaky_filter(input, decay: float = 0.7):
    output = [input[0]]
    frame = input[0] 
    print(input.shape)
    for ts in range(1, input.shape[0]):
        frame = decay * frame + input[ts]
        output.append(frame)
    return torch.stack(output)


def plot_sample(z):
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros_like(z[0, 0]), vmin=-1, vmax=1, cmap="bwr")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def animate(step):
        im.set_data(z[step, 0])
        return [im]

    ani = FuncAnimation(
        fig, animate, interval=100, blit=True, repeat=True, frames=z.shape[0])    
    ani.save(path / "sample.gif", dpi=300, writer=PillowWriter(fps=10))


def plot_conv1(v, z):
    print(z.shape, v.shape)
    n_kernel1 = int(16//2)
    fig, axs = plt.subplots(
        ncols=n_kernel1, nrows=2, figsize=(15,4), sharex=True,
        sharey=True)
    ims =\
        [axs[0, i].imshow(
            v[0, i], vmin=-0.1, vmax=0.1, cmap="bwr") for i in range(n_kernel1)]\
      + [axs[1, i].imshow(
          z[0, i], vmin=-1, vmax=1, cmap="bwr") for i in range(n_kernel1)]

    i = 0 
    for col in axs[0]:
        col.set_title(f"v(t), kernel {i}")
        col.set_xlabel("x")
        col.set_ylabel("y")
        i += 1
    for col in axs[1]:
        col.set_title(f"s(t)")
        col.set_xlabel("x")
        col.set_ylabel("y")

    def animate(step):
        # im.set_data(sample[step, 0])
        for i in range(n_kernel1):
            ims[i].set_data(v[step, i])
            ims[i].set_clim(vmin=-0.1, vmax=0.1)
        for i in range(n_kernel1):
            ims[i + n_kernel1].set_data(z[step, i])
            ims[i + n_kernel1].set_clim(vmin=-1, vmax=1)
        return ims

    ani = FuncAnimation(
        fig, animate, interval=500, blit=True, repeat=True, frames=z.shape[0])    
    fig.tight_layout()
    ani.save(path / "conv1_1.gif", dpi=300, writer=PillowWriter(fps=100))

    fig, axs = plt.subplots(ncols=n_kernel1, nrows=2, figsize=(15,4))
    ims =\
        [axs[0, i].imshow(
            v[0, i + 8], vmin=-0.1, vmax=0.1, cmap="bwr") for i in range(n_kernel1)]\
      + [axs[1, i].imshow(
          z[0, i + 8], vmin=-1, vmax=1, cmap="bwr") for i in range(n_kernel1)]

    i = 0 
    for col in axs[0]:
        col.set_title(f"v(t), kernel {i + 8}")
        col.set_xlabel("x")
        col.set_ylabel("y")
        i += 1
    for col in axs[1]:
        col.set_title(f"s(t)")
        col.set_xlabel("x")
        col.set_ylabel("y")

    def animate(step):
        # im.set_data(sample[step, 0])
        for i in range(n_kernel1):
            ims[i].set_data(v[step, i + 8])
            ims[i].set_clim(vmin=-0.1, vmax=0.1)
        for i in range(n_kernel1):
            ims[i + n_kernel1].set_data(z[step, i + 8])
            ims[i + n_kernel1].set_clim(vmin=-1, vmax=1)
        return ims

    ani = FuncAnimation(
        fig, animate, interval=500, blit=True, repeat=True, frames=z.shape[0])    
    fig.tight_layout()
    ani.save(path / "conv1_2.gif", dpi=300, writer=PillowWriter(fps=100))

def plot_conv2(v, z):
    # conv2
    fig, axs = plt.subplots(ncols=n_kernel2, nrows=2, figsize=(15,4))
    ims =\
        [axs[0, i].imshow(
            v[0, i], vmin=-10, vmax=10, cmap="bwr") for i in range(n_kernel2)]\
      + [axs[1, i].imshow(
          z[0, i], vmin=-1, vmax=1, cmap="bwr") for i in range(n_kernel2)]

    def animate(step):
        # im.set_data(sample[step, 0])
        for i in range(n_kernel2):
            ims[i].set_data(v[step, i])
            ims[i].set_clim(vmin=-1, vmax=1)
        for i in range(n_kernel2):
            ims[i + n_kernel2].set_data(z[step, i])
            ims[i + n_kernel2].set_clim(vmin=-1, vmax=1)
        return ims

    ani = FuncAnimation(
        fig, animate, interval=500, blit=True, repeat=True, frames=z.shape[0])    
    fig.tight_layout()
    ani.save(path / "conv2.gif", dpi=300, writer=PillowWriter(fps=100))

    print("Conv2 done")

def plot(path):
    dataset = EgoMotionDataset(
            10000, width, height, velocity=(
                (period_sim / period) / np.array(
                [period_sim / period * 1.2, period_sim / period])),
                n_objects=np.random.randint(15,high=30),label=1)
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Lambda(lambda input: leaky_filter(input)),
    ])
    dataset = TransformedSubset(dataset, transforms)

    # Model
    # model = MotionDetectionSNN(
    #     *target_size, input_avg_pooling, tau_mem1, tau_mem2, tau_mem3,
    #     v_th1, v_th2, v_th3, train_mem=False, learn_threshold=False, record=True)
    # model = MotionDetectionGaussSNN(
    #     *target_size, tau_mem1, tau_mem2, tau_mem2, v_th1, v_th2,
    #     train_mem=False, learn_threshold=False)
    model = MotionDetectionGaussSNN(
        *target_size, input_avg_pooling, tau_mem1, tau_mem2, tau_mem3,
        v_th1, v_th2, v_th3, train_gauss=True, train_mem=False,
        learn_threshold=False)
    model.load_state_dict(torch.load(path / "model.pt"))
    # model.lif1._reset_mechanism = "zero"
    # model.lif2._reset_mechanism = "zero"
    # model.lif3._reset_mechanism = "zero"
    model.to(device)
    print(model.lif1.threshold)
    print(model.lif2.threshold)
    # print(model.lif3.threshold)
    print(model.lif1.beta)
    print(model.lif2.beta)
    # print(model.lif3.beta)

    for i in range(index):
        data, y = dataset[0]
    out = model(data.unsqueeze(0).to(dtype).to(device))

    z1 = model.z1.detach().cpu()[0].numpy()
    v1 = model.v1.detach().cpu()[0].numpy()
    z2 = model.z2.detach().cpu()[0].numpy()
    v2 = model.v2.detach().cpu()[0].numpy()
    # z3 = model.z3.detach().cpu()[0].numpy()
    # v3 = model.v3.detach().cpu()[0].numpy()
    y = model.y.detach().cpu()[0]
    ysum = np.cumsum(y)

    # conv1
    plot_sample(data.numpy())
    # conv1
    plot_conv1(v1, z1)
    # conv2
    plot_conv2(v2, z2)

    # lif
    spikes = torch.nonzero(torch.tensor(z2))
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5,4))
    # print(v3.shape)
    # axs[0].plot(v3)
    print(spikes)
    try:
        axs[1].scatter(spikes[:, 0].numpy(), spikes[:, 1].numpy())
        axs[1].set_xlim(0, 20)
    except:
        pass
    fig.savefig(path / "lif.png")

    # out
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5,4))
    axs[0].set_ylim(-0.25, 0.25)
    axs[1].set_ylim(-3, 3)
    axs[0].set_ylabel("v(t)")
    axs[1].set_ylabel("sum(v(t))")
    axs[0].set_xlabel("time")
    axs[1].set_xlabel("time")

    line1, = axs[0].plot(np.arange(y.shape[0]), y)
    line2, = axs[1].plot(np.arange(y.shape[0]), ysum)

    def animate(step):
        sample = animate.sample
        line1.set_data(np.arange(0, step), sample[0][:step])
        line2.set_data(np.arange(0, step), sample[1][:step])
        animate.sample = sample
        return [line1, line2]
    animate.sample = (y, ysum)

    ani = FuncAnimation(
        fig, animate, interval=500, blit=True, repeat=True, frames=y.shape[0])    
    fig.tight_layout()
    ani.save(path / "out.gif", dpi=300, writer=PillowWriter(fps=10))

if __name__ == "__main__":
    plot(path)