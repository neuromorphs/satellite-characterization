import argparse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from classification_dataset import ClassificationDataset
from classification_model import SpectrogramCNN


def main(args):
    data_module = ClassificationDataset(batch_size=args.batch_size, dataset_path=args.dataset_path)

    class_names = ["39483", "39485", "39624", "40921", "43751", "43752", "50574", "51102"]
    model = SpectrogramCNN(num_classes=args.num_classes, class_names=class_names)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    logger = TensorBoardLogger("tb_logs", name="spectrogram_cnn")

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Spectrogram CNN with PyTorch Lightning')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (default: 32)')
    parser.add_argument('--dataset_path', type=str, default="data/astrosite/spectrograms", help='Path to dataset (default: data/astrosite/spectrograms_training)')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes (default: 5)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs (default: 10)')

    args = parser.parse_args()
    main(args)