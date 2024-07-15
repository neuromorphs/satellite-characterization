import torchvision
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class ClassificationDataset(LightningDataModule):
    def __init__(
        self,
        batch_size=16,
        dataset_path="data/astrosite/spectrograms",
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = torchvision.datasets.ImageFolder(
            dataset_path, transform=transform, target_transform=target_transform
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Define your split ratio
        train_ratio = 0.8  # 80% training, 20% testing

        # Calculate the sizes of the training and testing sets
        train_size = int(train_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size

        # Split the dataset
        train_dataset, test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_dataset = train_dataset
        self.val_dataset = test_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )
