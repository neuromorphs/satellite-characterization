""" Train motion detection algoritm """
import torch
import torchvision
import numpy as np
from pathlib import Path
from tqdm import tqdm
from astrosite_dataset import BinaryClassificationAstrositeDataset
from models.scnn_detection import MotionDetectionSNN
from tonic.slicers import SliceByTime
from tonic import SlicedDataset, transforms


base_path = Path("./data/dynamic_tau")
data_path = Path("data.csv")
model_path = Path("model.pt")
base_path.mkdir(exist_ok=True, parents=True)
with open(base_path / data_path, "w") as file:
    file.write("epochs;train_loss;train_acc;val_loss;val_acc\n")

lr = 1e-5
reg_strength = 1e-6
epochs = 100
batch_size = 64 
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
dtype = torch.float
loss_fn = torch.nn.BCEWithLogitsLoss()

# Membrane time constant
tau_mem1 = 0.9
tau_mem2 = 0.9
tau_mem3 = 0.95
train_mem = False

# Target size for tonic down sampling
target_size = [256, 144]
# Additional down sampling
input_avg_pooling = 1

# data
dataset_path = '/ley/users/earnold/datasets/astrosite/recordings'
target_list = [
    '50574', '47851', '37951', '39533', '43751', '32711', '27831', '45465',
    '46826', '42942', '42741', '41471', '43873', '40982', '41725', '43874',
    '27711', '40892', '50005', '44637']

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


def do_epoch(model, data_loader, optimizer, training: bool):
    model.train(training)

    # Minibatch training loop
    accs, losses = [], []
    samples = 0
    pbar = tqdm(total=len(data_loader), unit="batch")
    for data, target in data_loader:

        data = data.to(device).to(dtype)
        target = target.to(device).to(dtype)

        if training:
            optimizer.zero_grad()

        # forward pass (one polarity)
        y = model(data)
        # sum-over-time
        y_sum = y.sum(1).reshape(-1)

        #  loss
        loss = loss_fn(y_sum, target) + reg_strength * model.regularize()
        acc = torch.sum((y_sum >= 0) == target)

        # Gradient calculation + weight update
        if training:
            loss.backward()
            optimizer.step()

        # Store loss history for future plotting
        losses.append(loss.detach())
        accs.append(acc.detach())

        # count samples
        samples += data.shape[0]

        pbar.set_postfix(
            loss=f"{loss/data.shape[0]:.4f}", acc=f"{acc/data.shape[0]:.4f}")
        pbar.update()

    pbar.close()

    loss = torch.stack(losses).sum() / samples
    acc = torch.stack(accs).sum() / samples

    return loss, acc


def main():
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
    train_indices = indices[:int(size * 0.8)]
    val_indices = indices[int(size * 0.8):]
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda input: torch.tensor(input)),
        torchvision.transforms.Lambda(lambda input: input[:, 1]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.Lambda(lambda input: input.unsqueeze(1)),
    ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda input: torch.tensor(input)),
        torchvision.transforms.Lambda(lambda input: input[:, 1]),
        torchvision.transforms.Lambda(lambda input: input.unsqueeze(1)),
    ])
    train_subset = torch.utils.data.Subset(dataset, indices=train_indices)
    train_set = TransformedSubset(
        train_subset, transform=train_transforms)
    val_subset = torch.utils.data.Subset(dataset, indices=val_indices)
    val_set = TransformedSubset(
        val_subset, transform=val_transforms)

    # Data
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, batch_size=batch_size, num_workers=1)

    # Model
    model = MotionDetectionSNN(
        *target_size, input_avg_pooling,  tau_mem1, tau_mem2, tau_mem3,
        train_mem=train_mem)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999))

    # Train and test
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    pbar = tqdm(total=epochs, unit="epoch")
    for epoch in range(epochs):
        # Train and evaluate
        train_loss, train_acc = do_epoch(
            model, train_loader, optimizer, True)
        val_loss, val_acc = do_epoch(
            model, val_loader, optimizer, False)

        with open(base_path / data_path, "a") as file:
            file.write(
                f"{epoch};{train_loss.item()};{train_acc.item()};"
                + f"{val_loss.item()};{val_acc.item()}\n")

        pbar.set_postfix(
            loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")
        pbar.update()

        # Keep
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save model
        torch.save(model.to("cpu").state_dict(), base_path / model_path)
        model.to(device)
    
    pbar.close()


if __name__ == "__main__":
    main()