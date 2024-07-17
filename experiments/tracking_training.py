""" Train motion detection algoritm """
import sys
sys.path.append('/home/jules/Documents/satellite-characterization')
import torch
import torchvision
import numpy as np
from pathlib import Path
from tqdm import tqdm
from astrosite_dataset import build_merge_dataset, MergedDataset
from models.scnn_tracker import MotionTrackerStaticSNN
from tonic.slicers import SliceByTime
from tonic import SlicedDataset, transforms
import norse

base_path = Path("./data/dynamic_tau")
data_path = Path("data.csv")
model_path = Path("model_1sec_sample.pt")
base_path.mkdir(exist_ok=True, parents=True)
with open(base_path / data_path, "w") as file:
    file.write("epochs;train_loss;train_acc;val_loss;val_acc\n")

lr = 1e-3
reg_strength = 1e-6
epochs = 50
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
print("Using device:", device)
dtype = torch.float
loss_fn = torch.nn.BCEWithLogitsLoss()

# Membrane time constant
tau_mem1 = 0.2
tau_mem2 = 0.4
tau_mem3 = 0.6
train_mem = False

# Target size for tonic down sampling
input_size = [256,144]
output_size = [15,10]
# Additional down sampling
input_avg_pooling = 1

# data
dataset_path = '/home/jules/Documents/filtered_dataset/recordings'
target_list = [
    '50574', '47851', '37951', '39533', '43751', '32711', '27831', '45465',
    '46826', '42942', '42741', '41471', '43873', '40982', '41725', '43874',
    '27711', '40892', '50005', '44637']

# Fix random seed
np.random.seed(0)
torch.manual_seed(0)

def get_keypoint(heatmap): #returns (argmax x, argmax y) for input (y,x)
    return np.flip(np.unravel_index(np.argmax(heatmap), heatmap.shape))

def heatmapLoss(pred, target, batch_size):
    # [64, 7, 48, 48]
    
    #print(pred.shape, target.shape)
    #logging.info(pred.shape)
    #logging.info(target.shape)
    #heatmaps_pred = pred.reshape((batch_size, pred.shape[1], -1)).split(1, 1)
    # #对tensor在某一dim维度下，根据指定的大小split_size=int，或者list(int)来分割数据，返回tuple元组
    #print(len(heatmaps_pred), heatmaps_pred[0].shape)#7 torch.Size([64, 1, 48*48]
    #heatmaps_gt = target.reshape((batch_size, pred.shape[1], -1)).split(1, 1)
    #print(heatmaps_gt[0].shape)
    #print(len(heatmaps_pred))
    loss = 0
    #for idx in range(pred.shape[0]):
    heatmap_pred = pred#.squeeze(1)#[64, 40*40]
    
    heatmap_gt = target#.squeeze(1)
    loss += centernetfocalLoss(heatmap_pred, heatmap_gt)
    loss /= pred.shape[1]
    return loss #self.myMSEwithWeight(pred, target)

def centernetfocalLoss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pred = pred.clamp(1e-4,1-1e-4)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        #if len(x.shape)==1:
        #    x = torch.zeros((10,2,40,60),dtype=torch.short)
        try :
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
        except Exception as e :
            x = np.zeros((10,2,input_size[1], input_size[0]),dtype=np.int8)
            if self.transform:
                x = self.transform(x)
                y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.subset)


def do_epoch(model, data_loader, optimizer, training: bool):
    model.train(training)

    # Minibatch training loop
    accs, losses = [], []
    samples = 0
    pbar = tqdm(total=len(data_loader), unit="batch")
    test_data, test_target = next(iter(data_loader))
    for data, target in data_loader:

        data = data.to(device).to(dtype)
        target = target.to(device).to(dtype)

        if training:
            optimizer.zero_grad()

        # forward pass (one polarity)
        y = model(data)
        # sum-over-time
        y_mean= torch.mean(y,dim=0)
        #  loss
        loss = heatmapLoss(y_mean,target, batch_size) #loss_fn(y_sum, target) + reg_strength * model.regularize()
        acc = 0
        for idx in range(y_mean.shape[0]):
            y_pred, x_pred = get_keypoint(y_mean[idx][0].cpu().detach().numpy())
            y_gt, x_gt = get_keypoint(target[idx][0].cpu().detach().numpy())
            acc += np.linalg.norm((y_pred-y_gt, x_pred-x_gt))
        # Gradient calculation + weight update
        if training:
            loss.backward()
            optimizer.step()

        # Store loss history for future plotting
        losses.append(loss.detach())
        accs.append(acc)

        # count samples
        samples += data.shape[0]

        pbar.set_postfix(
            loss=f"{loss/data.shape[0]:.4f}", acc=f"{acc/data.shape[0]:.4f}")
        pbar.update()

    pbar.close()

    loss = torch.stack(losses).sum() / samples
    acc = np.stack(accs).sum() / samples

    return loss, acc


def main():
    dataset = build_merge_dataset(
        dataset_path, split=target_list)

    # Split into train and test set
    # TODO: Use val set instead of test set
    size = len(dataset.dataset1)
    print(size)
    input,target = dataset[0]
    print(input.shape)
    print(target.shape)
    indices = torch.arange(size)
    train_indices = indices[:int(size * 0.8)]
    val_indices = indices[int(size * 0.8):]
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda input: torch.from_numpy(input)),
        #torchvision.transforms.Lambda(lambda input: input[:, 1]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        #torchvision.transforms.Lambda(lambda input: input.unsqueeze(1)),
    ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda input: torch.from_numpy(input)),
        #torchvision.transforms.Lambda(lambda input: input[:, 1]),
        #torchvision.transforms.Lambda(lambda input: input.unsqueeze(1)),
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
    model = MotionTrackerStaticSNN(
        input_size[1], input_size[0],  tau_mem1, tau_mem2, tau_mem3,
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