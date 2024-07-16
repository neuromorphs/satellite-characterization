import torch.nn as nn
import torch
import lightning as pl
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import torchvision


class SpectrogramCNN(pl.LightningModule):
    def __init__(self, num_classes=8, class_names=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 6)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(32, 64, kernel_size=(3, 6)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(64, 128, kernel_size=(4, 8)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f'Class {i}' for i in range(num_classes)]
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("val_loss", loss)
        _, predicted = torch.max(y_hat, dim=1)
        accuracy = torch.sum(predicted == y).item() / len(y)
        self.log("val_acc", accuracy, prog_bar=True)
        self.confusion_matrix.update(y_hat.argmax(dim=1), y)
        return loss

    def on_validation_epoch_end(self):
        cm = self.confusion_matrix.compute().cpu().numpy()
        self.confusion_matrix.reset()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

        buf = BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)

        self.logger.experiment.add_image("Confusion Matrix", im, self.current_epoch)
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
