import time

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from parallel_cross_encoder.configs import TestConfigs
from parallel_cross_encoder.dataset import load_test_data
from parallel_cross_encoder.models.lightning_model import LightningModel


def evaluate_model(
    model: LightningModel, test_data: torch.utils.data.Dataset, configs: TestConfigs
) -> tuple:
    """

    Evaluate a cross encoder model given a test dataset

    Args:
        model (LightningModel): Pytorch Lightning model to evaluate
        test_data (torch.utils.data.Dataset): dataset for evaluation
        configs (TestConfigs): testing configurations

    Returns:
        Tuple(float, float): F1 and inference elapsed time
    """
    start = time.perf_counter()

    y_true = test_data.targets
    y_pred = model.predict(
        dataset=test_data,
        batch_size=configs.batch_size,
        dataloader_num_workers=configs.dataloader_num_workers,
        device=configs.device,
    )

    elapsed_time = time.perf_counter() - start

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )
    acc = accuracy_score(y_true, y_pred)

    print(
        f"F1: {f1:.03f}. Precision: {precision:.03f}. Recall: {recall:.03f}. Accuracy: {acc:.03f}. Time: {elapsed_time:.02f}",
        end="\n\n",
    )

    return f1, elapsed_time


def test_cmd(configs: TestConfigs) -> tuple:
    """
    Test/Eval command

    Launches a cross encoder test based on the TestConfigs passed as argument.

    Args:
        configs (TestConfigs): testing configuration

    Returns:
        Tuple(float, float): F1 and inference elapsed time
    """
    print("TEST CONFIGS :", configs)
    model = LightningModel.load_from_checkpoint(configs.model_path)
    model = model.to(configs.device)

    test_data = load_test_data(model.tokenizer, configs, model.configs)

    return evaluate_model(model, test_data, configs)
