""" Train motion detection algoritm """
import torch
import numpy as np
from tqdm import tqdm
from astrosite_dataset import BinaryClassificationAstrositeDataset
from models.scnn_detection import MotionDetectionSNN
from tonic.slicers import SliceByTime
from tonic import SlicedDataset, transforms


lr = 1e-3
epochs = 10
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
dtype = torch.float
loss_fn = torch.nn.BCEWithLogitsLoss()

# Membrane time constant
tau_mem1 = 10e-3
tau_mem2 = 10e-3
tau_mem3 = 20e-3

# data
dataset_path = '/ley/users/earnold/datasets/astrosite/recordings'
target_list = [
    '50574', '47851', '37951', '39533', '43751', '32711', '27831', '45465',
    '46826', '42942', '42741', '41471', '43873', '40982', '41725', '43874',
    '27711', '40892', '50005', '44637']

# Fix random seed
np.random.seed(0)
torch.manual_seed(0)


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

        # forward pass
        y = model(data)
        # sum-over-time
        y_sum = y.sum(1).reshape(-1)

        #  loss
        loss = loss_fn(y_sum, target)
        acc = torch.sum((y_sum > 0.5) == target)

        # Gradient calculation + weight update
        if training:
            loss.backward()
            optimizer.step()

        # Store loss history for future plotting
        losses.append(loss.detach())
        accs.append(acc.detach())

        # count samples
        samples += data.shape[1]

        pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
        pbar.update()

    pbar.close()

    loss = torch.stack(losses).sum() / samples
    acc = torch.stack(accs).sum() / samples

    return loss, acc


def main():
    dataset = BinaryClassificationAstrositeDataset(
        dataset_path, split=target_list)

    # Slice dataset into time frames
    slicer = SliceByTime(time_window=1e6, include_incomplete=False)
    frame_transform = transforms.ToFrame(
        sensor_size=dataset.sensor_size, time_window=1e5,
        include_incomplete=True)
    dataset = SlicedDataset(
        dataset, slicer=slicer, metadata_path='metadata/1',
        transform=frame_transform)

    # Split into train and test set
    # TODO: Use val set instead of test set
    size = len(dataset)
    indices = torch.arange(size)
    train_indices = indices[:int(size * 0.8)]
    val_indices = indices[int(size * 0.8):]
    train_set = torch.utils.data.Subset(dataset, indices=train_indices)
    val_set = torch.utils.data.Subset(dataset, indices=val_indices)

    # Data
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, batch_size=batch_size, num_workers=1)

    # Model
    model = MotionDetectionSNN(720, 1280, tau_mem1, tau_mem2, tau_mem3)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999))

    # Train and test
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    pbar = tqdm(total=epochs, unit="epoch")
    for _ in range(epochs):
        # Train and evaluate
        train_loss, train_acc = do_epoch(
            model, train_loader, optimizer, True)
        val_loss, val_acc = do_epoch(
            model, val_loader, optimizer, False)

        pbar.set_postfix(
            loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")
        pbar.update()

        # Keep
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    
    pbar.close()


if __name__ == "__main__":
    main()