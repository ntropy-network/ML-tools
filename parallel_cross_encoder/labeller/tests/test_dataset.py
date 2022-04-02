from datasets import load_dataset

from labeller.src.configs import TrainConfigs
from labeller.src.dataset import (
    LABELS,
    NewsDatasetInferenceMultiLabel,
    NewsDatasetSingleLabel,
    NewsDatasetTrainMultiLabel,
    load_train_data,
)

from transformers.models.auto.tokenization_auto import AutoTokenizer


def test_news_dataset_multilabel():
    newsgroups_data = load_dataset("ag_news")["train"]
    tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
    configs = TrainConfigs(no_cuda=True)
    print("c", configs)
    d = NewsDatasetTrainMultiLabel(
        samples=newsgroups_data["text"],
        targets=newsgroups_data["label"],
        target_names=LABELS,
        tokenizer=tokenizer,
        configs=configs,
    )
    print("d", d[1])
    assert isinstance(d[1], dict)


def test_news_dataset_singlelabel():
    newsgroups_data = load_dataset("ag_news")["train"]
    tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
    configs = TrainConfigs(no_cuda=True)
    print("c", configs)
    d = NewsDatasetSingleLabel(
        samples=newsgroups_data["text"],
        targets=newsgroups_data["label"],
        target_names=LABELS,
        tokenizer=tokenizer,
        model_type=configs.model_type,
        max_length=configs.max_length,
    )
    print("d", d[1])
    assert isinstance(d[1], dict)


def test_dataset_inference():
    newsgroups_data = load_dataset("ag_news")["train"]
    tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
    d = NewsDatasetInferenceMultiLabel(
        samples=newsgroups_data["text"],
        targets=newsgroups_data["label"],
        target_names=LABELS,
        tokenizer=tokenizer,
        num_labels_per_sample=4,
        batch_size=16,
        model_type="bart",
    )
    print("d", d[1])
    assert isinstance(d[1], dict)


def test_dataset_inference_no_labels():
    newsgroups_data = load_dataset("ag_news")["train"]
    tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")
    d = NewsDatasetInferenceMultiLabel(
        samples=newsgroups_data["text"],
        targets=None,
        target_names=LABELS,
        tokenizer=tokenizer,
        num_labels_per_sample=4,
        batch_size=2,
        model_type="bart",
    )
    print("d", d[1])
    assert isinstance(d[1], dict)


def test_load_data():
    tokenizer = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")
    params = {
        "task": "business",
        "model_type": "bart",
        "negative_labels_per_sample": 3,
        "logging_dir": "./logs",
        "logging_steps": 10,
        "output_dir": "./output",
        "batch_size": 32,
        "weight_decay": 0.0001,
        "dataloader_num_workers": 4,
        "dataloader_drop_last": False,
        "early_stopping_epochs": 6,
    }
    c = TrainConfigs(**params)
    dataset_train, dataset_test = load_train_data(tokenizer, c)
    print("train", dataset_train)
    print("test", dataset_test)
    assert isinstance(dataset_train[1], dict)
    assert isinstance(dataset_test[1], dict)
