import math
import os
import random
from typing import List

import numpy as np
import torch
from datasets import load_dataset

from .configs import TestConfigs, TrainConfigs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LABELS = [
    "world",
    "sports",
    "business",
    "science and technology",
]

HYPOTHESIS = "This example is about"


def load_train_data(tokenizer, configs: TrainConfigs):
    """
    Load AG News training data

    Args:
        tokenizer: transformers tokenizer
        configs (TrainConfigs): training configurations

    Returns:
        tuple : Train and Val datasets
    """
    newsgroups_data = load_dataset("ag_news")["train"]
    newsgroups_data = newsgroups_data.shuffle(seed=42)
    split = int(len(newsgroups_data) * (1 - configs.validation_data_ratio))
    train_data, val_data = newsgroups_data[:split], newsgroups_data[split:]

    if configs.parallelize_cross_encoder:
        ds_train = NewsDatasetTrainMultiLabel(
            samples=train_data["text"],
            targets=train_data["label"],
            target_names=LABELS,
            tokenizer=tokenizer,
            configs=configs,
        )

        ds_val = NewsDatasetTrainMultiLabel(
            samples=val_data["text"],
            targets=val_data["label"],
            target_names=LABELS,
            tokenizer=tokenizer,
            configs=configs,
        )
    else:
        ds_train = NewsDatasetSingleLabel(
            samples=train_data["text"],
            targets=train_data["label"],
            target_names=LABELS,
            tokenizer=tokenizer,
            model_type=configs.model_type,
            max_length=configs.max_length,
        )

        ds_val = NewsDatasetSingleLabel(
            samples=val_data["text"],
            targets=val_data["label"],
            target_names=LABELS,
            tokenizer=tokenizer,
            model_type=configs.model_type,
            max_length=configs.max_length,
        )

    return ds_train, ds_val


def load_test_data(tokenizer, configs: TestConfigs, train_configs: TrainConfigs):
    """
    Load AG News test data

    Args:
        tokenizer (_type_): transformers tokenizer
        configs (TestConfigs): testing configurations
        train_configs (TrainConfigs): model trained configurations

    Returns:
        dataset: test data
    """
    newsgroups_data = load_dataset("ag_news")["test"]

    if train_configs.parallelize_cross_encoder:
        dataset = NewsDatasetInferenceMultiLabel(
            samples=newsgroups_data["text"],
            targets=newsgroups_data["label"],
            target_names=LABELS,
            batch_size=configs.batch_size,
            tokenizer=tokenizer,
            num_labels_per_sample=configs.num_labels_per_sample,
            model_type=train_configs.model_type,
            max_length=train_configs.max_length,
        )
    else:
        dataset = NewsDatasetSingleLabel(
            samples=newsgroups_data["text"],
            targets=newsgroups_data["label"],
            target_names=LABELS,
            tokenizer=tokenizer,
            model_type=train_configs.model_type,
            max_length=train_configs.max_length,
        )
    return dataset


def make_hypothesis(
    model_type: str,
    tokenizer,
    labels: list,
) -> str:
    """
    Format transformers 2nd part of the input to include multiple classes to infer in parallel.

    Args:
        model_type (str): bart or bert
        tokenizer (_type_): transformers tokenizer
        labels (list): list of labels we want to infer on

    Raises:
        ValueError: If model type unknown

    Returns:
        str: formated hypothesis
    """
    if model_type == "bert":
        labels = "".join([f" {tokenizer.cls_token} {label}" for label in labels])
        hypothesis = f"{tokenizer.sep_token} {HYPOTHESIS} {labels}"
    elif model_type == "bart":
        labels = "".join([f" {label} {tokenizer.cls_token}" for label in labels])
        hypothesis = (
            f"{tokenizer.sep_token} {HYPOTHESIS} {tokenizer.sep_token} {labels}"
        )
    else:
        raise ValueError("Use bert or bart for model_type")
    return hypothesis


class NewsDatasetTrainMultiLabel(torch.utils.data.Dataset):
    """
    Train dataset for the parallelized transformers, that can infer multiple labels per sample.
    We can tune the number of negative labels to use per sample (there is only 1 positive label per sample).
    """

    def __init__(
        self,
        samples,
        targets,
        target_names,
        tokenizer,
        configs: TrainConfigs,
        entailment_id=1,
        contradiction_id=0,
    ):
        self.samples = samples
        self.targets = targets
        self.target_names = target_names

        self.model_type = configs.model_type
        self.negative_labels_per_sample = configs.negative_labels_per_sample
        assert self.negative_labels_per_sample < len(
            self.target_names
        ), "Too much negatives per sample given the number of targets"
        self.max_length = configs.max_length
        self.entailment_id = entailment_id
        self.contradiction_id = contradiction_id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt = self.samples[idx]
        target_idx = self.targets[idx]

        pos_label = self.target_names[target_idx]

        neg_labels = random.sample(
            [l for l in self.target_names if l != pos_label],
            self.negative_labels_per_sample,
        )

        labels = [pos_label] + neg_labels
        targets = [self.entailment_id] + [self.contradiction_id] * len(neg_labels)

        temp = list(zip(labels, targets))
        random.shuffle(temp)
        labels, targets = zip(*temp)
        labels, targets = list(labels), list(targets)

        hypothesis = make_hypothesis(self.model_type, self.tokenizer, labels)

        # print("train prompt", prompt)
        # print("train hypothesis", hypothesis)
        enc = self.tokenizer(
            prompt,
            hypothesis,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc["labels"] = torch.tensor(targets, dtype=torch.long)

        return {k: v.squeeze() for k, v in enc.items()}


class NewsDatasetSingleLabel:
    """
    This is the basic dataset for the vanilla cross-encoder (1 label per sample)
    """

    def __init__(
        self,
        samples,
        targets,
        target_names,
        tokenizer,
        model_type,
        max_length,
        entailment_id=1,
        contradiction_id=0,
    ):
        self.samples = samples
        self.targets = targets
        self.target_names = target_names

        self.model_type = model_type
        self.max_length = max_length
        self.entailment_id = entailment_id
        self.contradiction_id = contradiction_id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples) * len(self.target_names)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.target_names)
        label_idx = idx % len(self.target_names)
        prompt = self.samples[sample_idx]
        target_idx = self.targets[sample_idx]

        label = self.target_names[label_idx]
        if target_idx == label_idx:
            # pos label
            target = [self.entailment_id]
        else:
            # neg label
            target = [self.contradiction_id]

        hypothesis = f"{HYPOTHESIS} {label}"
        enc = self.tokenizer(
            prompt,
            hypothesis,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(target, dtype=torch.long)
        enc["sample_idxs"] = torch.tensor(sample_idx, dtype=torch.long)

        return {k: v.squeeze() for k, v in enc.items()}


class NewsDatasetInferenceMultiLabel(torch.utils.data.Dataset):
    """
    Here we test every possible label for each sample.
    This version infer several labels at the time for inference.
    It also make sure to fill the batch size.
    It uses extra variables (labels_mask and sample_idxs) to help retrieving the actual logits and
    discard the padding logits (in case there is some).
    """

    def __init__(
        self,
        samples,
        target_names,
        tokenizer,
        targets=None,
        num_labels_per_sample=16,
        batch_size=32,
        model_type: str = "bart",
        max_length: int = 256,
        entailment_id=1,
        contradiction_id=0,
    ):
        super().__init__()

        self.samples = samples
        self.targets = targets
        self.target_names = target_names

        self.num_labels_per_sample = num_labels_per_sample
        self.batch_size = batch_size
        self.entailment_id = entailment_id
        self.contradiction_id = contradiction_id
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length

        C = len(self.target_names)
        self.nb_samples = C // self.num_labels_per_sample
        self.last_sample_nb_labels = C % self.num_labels_per_sample
        if self.last_sample_nb_labels > 0:
            self.nb_samples += 1
            self.nb_pad_labels = self.num_labels_per_sample - self.last_sample_nb_labels
        else:
            self.nb_pad_labels = 0
        self.padded_target_names = self.target_names + (["none"] * self.nb_pad_labels)

        assert len(self.padded_target_names) % self.num_labels_per_sample == 0
        print("Inference Dataset: labels w/ padding:", self.padded_target_names)

    def __len__(self):
        return math.ceil(len(self.samples) * self.nb_samples / self.batch_size)

    def _get_tokenizer_inputs(self, prompt, labels_idx_start, pos_labels):
        nb_labels = len(self.padded_target_names) - labels_idx_start
        nb_samples = nb_labels // self.num_labels_per_sample
        tokenizer_input_1 = [f"{prompt}"] * nb_samples
        tokenizer_input_2 = []
        targets = []
        targets_mask = []
        for i in range(nb_samples):
            start = labels_idx_start + i * self.num_labels_per_sample
            end = min(
                len(self.padded_target_names),
                labels_idx_start + (i + 1) * self.num_labels_per_sample,
            )
            curr_topics = self.padded_target_names[start:end]
            t = np.zeros(len(curr_topics))
            if pos_labels is not None:
                for l in pos_labels:
                    if l in curr_topics:
                        t[curr_topics.index(l)] = 1
            targets.append(list(t))
            targets_mask.append([0 if l == "none" else 1 for l in curr_topics])
            hypothesis = make_hypothesis(self.model_type, self.tokenizer, curr_topics)
            tokenizer_input_2.append(hypothesis)
        return tokenizer_input_1, tokenizer_input_2, targets, targets_mask

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError("Auto batching. Must set batch_size=1 in dataloader.")

        sample_idxs = []

        labels_idx_start = (
            idx
            * (self.batch_size * self.num_labels_per_sample)
            % len(self.padded_target_names)
        )

        sample_idx = (
            idx
            * (self.batch_size * self.num_labels_per_sample)
            // len(self.padded_target_names)
        )

        tokenizer_input_1 = []
        tokenizer_input_2 = []

        targets = []
        targets_mask = []
        while True:
            if sample_idx >= len(self.samples):
                # reached end of dataset :)
                break
            prompt = self.samples[sample_idx]
            if self.targets is not None:
                target_idx = self.targets[sample_idx]
                pos_label = [self.target_names[target_idx]]
            else:
                pos_label = None
            (
                add_tok_1,
                add_tok_2,
                add_targets,
                add_targets_mask,
            ) = self._get_tokenizer_inputs(prompt, labels_idx_start, pos_label)
            tokenizer_input_1.extend(add_tok_1)
            tokenizer_input_2.extend(add_tok_2)
            targets.extend(add_targets)
            targets_mask.extend(add_targets_mask)
            sample_idxs.extend([sample_idx] * len(add_tok_1))
            if len(tokenizer_input_1) >= self.batch_size:
                tokenizer_input_1 = tokenizer_input_1[: self.batch_size]
                tokenizer_input_2 = tokenizer_input_2[: self.batch_size]
                targets = targets[: self.batch_size]
                targets_mask = targets_mask[: self.batch_size]
                sample_idxs = sample_idxs[: self.batch_size]
                break
            else:
                labels_idx_start = 0
                sample_idx += 1

        encodings = self.tokenizer(
            tokenizer_input_1,
            tokenizer_input_2,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        if len(tokenizer_input_1) > 1:
            items = {k: torch.squeeze(v) for k, v in encodings.items()}
        else:
            items = encodings
        items["sample_idxs"] = torch.tensor(sample_idxs, dtype=torch.long)
        if self.targets is not None:
            items["labels"] = torch.tensor(targets, dtype=torch.long)
            items["labels_mask"] = torch.tensor(targets_mask, dtype=torch.long)
        return items
