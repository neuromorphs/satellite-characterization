""" Train motion detection algoritm """
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.parent.resolve())
import argparse
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from models.scnn_detection import (
    MotionDetectionGaussSNN, MotionDetectionSNN, MotionDetectionDualSNN)
from models.lstm_detection import MotionDetectionLSTM
from models.lif_detection import MotionDetectionLifSNN
from models.gauss_lif_detection import MotionDetectionGaussLifSNN
from dataset import EgoMotionDataset


def get_parser() -> argparse.ArgumentParser:
    """ Returns an argument parser with all the options """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-path", type=str, default="./results")

    # training
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--period-sim", type=int, default=1000)
    parser.add_argument("--period", type=int, default=100)

    # dataset
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--low-pass-filter", default=False, action="store_true")
    parser.add_argument('--target-size', nargs='+', type=int, default=[80, 120])

    # model
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--input-avg-pooling", type=int, default=1)
    parser.add_argument("--reg-strength", type=float, default=1e-4)
    parser.add_argument(
        '--tau-mem', nargs='+', type=float, default=[0.8, 0.8, 0.9])
    parser.add_argument(
        '--v-th', nargs='+', type=float, default=[0.1, 0.4, 0.4])
    parser.add_argument("--train-tau-mem", default=False, action="store_true")
    parser.add_argument("--train-v-th", default=False, action="store_true")
    parser.add_argument("--train-gauss", default=False, action="store_true")

    return parser


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


def get_model(args):
    if args.model_name == "lif":
        model = MotionDetectionLifSNN(
            *args.target_size,
            args.input_avg_pooling,
            *args.tau_mem,
            args.train_tau_mem)
    if args.model_name == "gauss_lif":
        model = MotionDetectionGaussLifSNN(
            *args.target_size,
            *args.tau_mem,
            *args.v_th,
            train_mem=args.train_tau_mem,
            learn_threshold=args.train_v_th)
    if args.model_name == "lstm":
        model = MotionDetectionLSTM(
            *args.target_size,
            args.input_avg_pooling)
    if args.model_name == "scnn":
        model = MotionDetectionSNN(
            *args.target_size,
            args.input_avg_pooling,
            *args.tau_mem,
            *args.v_th,
            train_mem=args.train_tau_mem,
            learn_threshold=args.train_v_th)
    if args.model_name == "gauss_scnn":
        model = MotionDetectionGaussSNN(
            *args.target_size,
            args.input_avg_pooling,
            *args.tau_mem,
            *args.v_th,
            train_gauss=args.train_gauss,
            train_mem=args.train_tau_mem,
            learn_threshold=args.train_v_th)
    if args.model_name == "dual_scnn":
        model = MotionDetectionDualSNN(
            *args.target_size,
            args.input_avg_pooling,
            *args.tau_mem,
            train_mem=args.train_tau_mem,
            learn_threshold=args.train_v_th)
    return model


def leaky_filter(input, decay: float = 0.7):
    output = [input[:, 0]]
    frame = input[:, 0] 
    for ts in range(1, input.shape[1]):
        frame = decay * frame + input[:, ts]
        output.append(frame)
    return torch.stack(output).transpose(1, 0)


def do_epoch(args, model, loss_fn, data_loader, optimizer, training: bool):
    model.train(training)

    # Minibatch training loop
    accs, losses = [], []
    samples = 0
    pbar = tqdm(total=len(data_loader), unit="batch")
    for data, target in data_loader:
        data = data.to(model.device).to(float)
        target = target.to(model.device).to(float)

        if training:
            optimizer.zero_grad()

        # forward pass (one polarity)
        y = model(data)
        # sum-over-time
        y_sum = y.sum(1).reshape(-1)

        #  loss
        loss = loss_fn(y_sum, target)
        if hasattr(model, "regularize"):
            reg_loss = args.reg_strength * model.regularize()
        else:
            reg_loss = torch.tensor(0.)
        acc = torch.sum((y_sum >= 0) == target)

        # Gradient calculation + weight update
        if training:
            (loss + reg_loss).backward()
            optimizer.step()

        # Store loss history for future plotting
        losses.append(loss.detach())
        accs.append(acc.detach())

        # count samples
        samples += data.shape[0]

        pbar.set_postfix(
            loss=f"{loss.detach().mean():.4f}",
            reg_loss=f"{reg_loss.detach().mean():.4f}",
            acc=f"{acc.detach() / data.shape[0]:.4f}")
        pbar.update()

    pbar.close()

    loss = torch.stack(losses).sum() / samples
    acc = torch.stack(accs).sum() / samples

    return loss, acc


def main(args: argparse.Namespace):
    print(vars(args))
    """ Train a model """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dir to save data
    base_path = Path(args.base_path)
    base_path.mkdir(exist_ok=True, parents=True)
    data_path = base_path / "data.csv"
    model_path = base_path / "model.pt"

    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_set = EgoMotionDataset(
            10000, args.width, args.height, velocity=(
                (args.period_sim / args.period) / np.array(
                [args.period_sim / args.period * 1.2,
                 args.period_sim / args.period])),
                n_objects=np.random.randint(15, high=30), label=1)
    val_set = EgoMotionDataset(
            1000, args.width, args.height, velocity=(
                (args.period_sim / args.period) / np.array(
                [args.period_sim / args.period * 1.2,
                 args.period_sim / args.period])),
                n_objects=np.random.randint(15, high=30), label=1)

    if args.low_pass_filter:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda input: leaky_filter(input)),
        ])
        train_set = TransformedSubset(train_set, transforms)
        val_set = TransformedSubset(val_set, transforms)

    # Data
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, batch_size=args.batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, batch_size=args.batch_size, num_workers=1)

    # Model
    model = get_model()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Train and test
    data = np.zeros((4, args.epochs))
    pbar = tqdm(total=args.epochs, unit="epoch")
    for epoch in range(args.epochs):
        # Train and evaluate
        train_loss, train_acc = do_epoch(
            args, model, loss_fn, train_loader, optimizer, True)
        val_loss, val_acc = do_epoch(
            args, model, loss_fn, val_loader, optimizer, False)

        data[0, epoch] = train_loss.item()
        data[1, epoch] = train_acc.item()
        data[2, epoch] = val_loss.item()
        data[3, epoch] = val_acc.item()

        pbar.set_postfix(
            loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")
        pbar.update()

        # Save model
        torch.save(model.to("cpu").state_dict(), base_path / model_path)
        with open(data_path, "wb") as file:
            np.save(file, {"data": data, "args": vars(args)})
        model.to(device)
    
    pbar.close()


if __name__ == "__main__":
    main(get_parser().parse_args())
