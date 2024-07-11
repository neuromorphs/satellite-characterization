import torch
import numpy as np

def coordinate_loss(output, label) :
    #(x,y) coordinates of the satellite, rate encoding or voltage values
    tuples = torch.nonzero(label[:,1], as_tuple=True)
    cy,cx = torch.mean(tuples[1][1:].type(torch.float)), torch.mean(tuples[2][1:].type(torch.float))
    return (output[0]-cy)**2 + (output[1]-cx)**2


def print_batch_accuracy(data, targets, batch_size, net, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")