""" Train motion detection algoritm """
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from astrosite_dataset import BinaryClassificationAstrositeDataset
from models.scnn_detection import MotionDetectionSNN
from experiments.motion_detection import TransformedSubset
from tonic.slicers import SliceByTime
from tonic import SlicedDataset, transforms
from matplotlib.animation import FuncAnimation, PillowWriter


# Membrane time constant
tau_mem1 = 0.9
tau_mem2 = 0.9
tau_mem3 = 0.95

n_kernel1 = 8
n_kernel2 = 8

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
dtype = torch.float

# Target size for tonic down sampling
target_size = [256, 144]
# Additional down sampling
input_avg_pooling = 1

# data
path = Path("./data/low_pass")
dataset_path = '/ley/users/earnold/datasets/astrosite/recordings'
target_list = [
    '50574', '47851', '37951', '39533', '43751', '32711', '27831', '45465',
    '46826', '42942', '42741', '41471', '43873', '40982', '41725', '43874',
    '27711', '40892', '50005', '44637']

  
def leaky_filter(input, decay: float = 0.9):
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

    def animate(step):
        im.set_data(z[step, 0])
        return [im]

    ani = FuncAnimation(
        fig, animate, interval=100, blit=True, repeat=True, frames=z.shape[0])    
    ani.save(path / "sample.gif", dpi=300, writer=PillowWriter(fps=10))


def plot_conv1(v, z):
    fig, axs = plt.subplots(ncols=n_kernel1, nrows=2, figsize=(15,4))
    ims =\
        [axs[0, i].imshow(
            v[0, i], vmin=-10, vmax=10, cmap="bwr") for i in range(n_kernel1)]\
      + [axs[1, i].imshow(
          z[0, i], vmin=-1, vmax=1, cmap="bwr") for i in range(n_kernel1)]

    def animate(step):
        # im.set_data(sample[step, 0])
        for i in range(n_kernel1):
            ims[i].set_data(v[step, i])
            ims[i].set_clim(vmin=-10, vmax=10)
        for i in range(n_kernel1):
            ims[i + n_kernel1].set_data(z[step, i])
            ims[i + n_kernel1].set_clim(vmin=-1, vmax=1)
        return ims

    ani = FuncAnimation(
        fig, animate, interval=500, blit=True, repeat=True, frames=z.shape[0])    
    fig.tight_layout()
    ani.save(path / "conv1.gif", dpi=300, writer=PillowWriter(fps=100))

    print("Conv1 done")

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
            ims[i].set_clim(vmin=-10, vmax=10)
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
    dataset = BinaryClassificationAstrositeDataset(
        dataset_path, split=target_list)

    # Slice dataset into time frames
    slicer = SliceByTime(time_window=10e6, include_incomplete=False)
    downsample = transforms.Compose([
        # transforms.Denoise(filter_time=1e5),
        transforms.Downsample(
            sensor_size=dataset.sensor_size, target_size=target_size),
        transforms.ToFrame(
            sensor_size=target_size + [2], n_time_bins=50,
            include_incomplete=True)])
    dataset = SlicedDataset(
        dataset, slicer=slicer, metadata_path='metadata/1',
        transform=downsample)

    # Split into train and test set
    # TODO: Use val set instead of test set
    size = len(dataset)
    indices = torch.arange(size)
    val_indices = indices[int(size * 0.8):]
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda input: torch.tensor(input)),
        torchvision.transforms.Lambda(lambda input: input[:, 1]),
        torchvision.transforms.Lambda(lambda input: input.unsqueeze(1)),
        torchvision.transforms.Lambda(lambda input: leaky_filter(input)),
    ])
    val_subset = torch.utils.data.Subset(dataset, indices=val_indices)
    val_set = TransformedSubset(
        val_subset, transform=val_transforms)

    # Model
    model = MotionDetectionSNN(
        *target_size, input_avg_pooling,  tau_mem1, tau_mem2, tau_mem3, True)
    model.load_state_dict(torch.load(path / "model.pt"))
    model.to(device)

    data, y = val_set[1]
    out = model(data.unsqueeze(0).to(dtype).to(device))

    z1 = model.z1.detach().cpu()[0]
    v1 = model.v1.detach().cpu()[0].numpy()
    z2 = model.z2.detach().cpu()[0].numpy()
    v2 = model.v2.detach().cpu()[0].numpy()
    z3 = model.z3.detach().cpu()[0].numpy()
    v3 = model.v3.detach().cpu()[0].numpy()
    y = model.y.detach().cpu()[0]
    ysum = np.cumsum(y)

    # conv1
    plot_sample(data.numpy())
    # conv1
    plot_conv1(v1, z1)
    # conv2
    plot_conv2(v2, z2)

    # lif
    spikes = torch.nonzero(torch.tensor(z3))
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5,4))
    axs[0].plot(v3)
    print(spikes)
    try:
        axs[1].scatter(spikes[:, 0].numpy(), spikes[:, 1].numpy())
        axs[1].set_xlim(0, 50)
    except:
        pass
    fig.savefig(path / "lif.png")

    # out
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5,4))
    axs[0].set_ylim(-1, 1)
    axs[1].set_ylim(-5, 10)
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