import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollatorWithPadding

from parallel_cross_encoder.dataset import load_train_data
from parallel_cross_encoder.models.lightning_model import (LightningModel,
                                                           load_model)

from .configs import TrainConfigs


class EpochElapsedTime(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed_time = time.perf_counter() - self.start
        print()
        print(f"It took {elapsed_time} seconds to complete the epoch")
        print()


def train_cmd(configs: TrainConfigs) -> str:
    """
    Train command.

    Launches a cross encoder training based on the TrainConfigs passed as argument.

    Args:
        configs (TrainConfigs): training configurations

    Returns:
        str: best trained model path
    """
    print("\n\n", "*" * 32, "\n*\tBeginning Training\t*\n", "*" * 32)

    print("TRAIN CONFIGS :", configs)

    if configs.parallelize_cross_encoder:
        name = "parallelized_cross_encoder"
    else:
        name = "vanilla_cross_encoder"

    name = name + "_" + configs.model_type

    logger = TensorBoardLogger(configs.output_dir, name=name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best-checkpoint",
        save_weights_only=True,
        save_top_k=1,
        verbose=True,
        monitor="val_f1",
        mode="max",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_f1",
        patience=configs.early_stopping_epochs,
        mode="max",
        verbose=True,
    )

    epoch_elapsed_time = EpochElapsedTime()
    callbacks = [early_stopping_callback, checkpoint_callback, epoch_elapsed_time]
    if configs.no_cuda:
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=checkpoint_callback,
            callbacks=callbacks,
            max_epochs=configs.epochs,
            accelerator="cpu",
            log_every_n_steps=configs.logging_steps,
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=checkpoint_callback,
            callbacks=callbacks,
            max_epochs=configs.epochs,
            gpus=-1,
            accelerator="auto",
            log_every_n_steps=configs.logging_steps,
        )
    model = LightningModel(configs)
    print("MODEL :", model)
    tokenizer = model.tokenizer
    ds_train, ds_val = load_train_data(tokenizer, configs)
    train_dataloader = DataLoader(
        ds_train,
        batch_size=configs.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=2),
        shuffle=True,
        drop_last=True,
        num_workers=configs.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        ds_val,
        batch_size=configs.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=2),
        shuffle=False,
        drop_last=False,
        num_workers=configs.dataloader_num_workers,
    )
    # setup line:
    model.steps_per_epoch = len(train_dataloader)
    print(f"Training for {configs.epochs} epochs ...")
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    return checkpoint_callback.best_model_path
