from copy import deepcopy
from typing import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch

from parallel_cross_encoder.configs import TrainConfigs
from parallel_cross_encoder.dataset import NewsDatasetInferenceMultiLabel
from parallel_cross_encoder.models.bart_model import BartForMultiSequenceClassification
from parallel_cross_encoder.models.bert_model import DistilBertForMultiSequenceClassification

from psutil import cpu_count
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    BartForSequenceClassification,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from transformers.data.data_collator import DataCollatorWithPadding


class LightningModel(pl.LightningModule):
    """
    Lightning Model for the vanilla/parallelized cross-encoders
    for faster transformers training and inference.
    """

    def __init__(self, configs: TrainConfigs):
        super().__init__()
        if isinstance(configs, dict):
            configs = TrainConfigs(**configs)
        self.configs = configs
        print("MODEL TRAINED CONFIGS :", configs)
        self.save_hyperparameters(configs.dict())
        self.model, self.tokenizer = load_model(configs)
        self.steps_per_epoch = None  # Need to be setup after init, manually

    def forward(self, input_ids, attention_mask, labels=None):
        if len(input_ids.shape) == 3:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            if labels is not None:
                labels = labels.squeeze(0)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # take only the entailment output in case of non-parallel model:
        if not self.configs.parallelize_cross_encoder:
            outputs.logits = outputs.logits[:, 1]
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", outputs.loss.item(), prog_bar=True, logger=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True)
        return {
            "loss": outputs.loss,
            "logits": outputs.logits.view(-1),
            "labels": labels.view(-1),
        }

    def validation_epoch_end(self, out):
        logits = torch.cat([x["logits"] for x in out])
        labels = torch.cat([x["labels"] for x in out])
        probs = torch.sigmoid(logits)
        preds = probs > self.configs.validation_threshold
        labels = labels.detach().cpu()
        preds = preds.detach().cpu()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = round(accuracy_score(labels, preds), 4)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        self.log("val_accuracy", acc)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        print(f"Validation :: acc: {acc}, pre: {precision}, rec: {recall}, f1: {f1}")
        return out

    def configure_optimizers(self):
        """Prepare optimizer and scheduler (linear warmup and weight decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.configs.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.configs.learning_rate,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.steps_per_epoch * self.configs.warmup_epochs,
            num_training_steps=self.steps_per_epoch * self.configs.epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def predict(
        self, dataset, batch_size=32, dataloader_num_workers=-1, device="cpu"
    ) -> list:
        """Predict labels on a given dataset.
        Works for vanilla or parallelized cross-encoders.

        Args:
            dataset (torch.Dataset): any torch NLP dataset
            batch_size (int, optional): Batch size. Defaults to 32.
            dataloader_num_workers (int, optional): Num workers (-1 for MAX). Defaults to -1.
            device (str, optional): (device cuda or cpu). Defaults to "cpu".

        Returns:
            list: predictions
        """

        self.eval()
        if dataloader_num_workers == -1:
            dataloader_num_workers = cpu_count()
        scores = [[] for _ in range(len(dataset.samples))]

        if isinstance(dataset, NewsDatasetInferenceMultiLabel):
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=dataloader_num_workers,
                batch_size=1,
                drop_last=False,
            )
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=dataloader_num_workers,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer, pad_to_multiple_of=2
                ),
                batch_size=batch_size,
                drop_last=False,
            )

        for batch in tqdm(loader):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sample_idxs = batch["sample_idxs"].detach().cpu().numpy()
                out = self(input_ids, attention_mask)
                out = out.logits.detach().cpu().numpy()
                yhat = 1 / (1 + np.exp(-out))
                if len(yhat.shape) == 1:
                    yhat = np.expand_dims(yhat, axis=1)
                if isinstance(dataset, NewsDatasetInferenceMultiLabel):
                    labels_mask = batch["labels_mask"].detach().cpu().numpy()[0]
                    for i, idx in enumerate(sample_idxs[0]):
                        yhat_no_padding = yhat[i][labels_mask[i] != 0]
                        scores[idx].extend(list(yhat_no_padding))
                else:
                    [scores[idx].append(yhat[i]) for i, idx in enumerate(sample_idxs)]

        preds = [np.argmax(s) for s in scores]

        return preds


def load_model(configs: TrainConfigs) -> tuple:
    """
    Load a Pretrained transformers model (bert or bart).
    Also we can choose how many layers we want to keep on both the encoder and decoder.
    Indeed, by taking only the few firsts layers of each modules, we can keep a high accuracy and make
    the forward pass faster.

    Args:
        configs (TrainConfigs): training configurations

    Raises:
        ValueError: only bert or bart available

    Returns:
        tuple: model, tokenizer
    """
    model_type = configs.model_type
    multilabel = configs.parallelize_cross_encoder
    encoder_hidden_layers = configs.encoder_hidden_layers
    decoder_hidden_layers = configs.decoder_hidden_layers

    if model_type == "bert":
        model_name = "typeform/distilbert-base-uncased-mnli"
        if multilabel:
            model_class = DistilBertForMultiSequenceClassification
        else:
            model_class = DistilBertForSequenceClassification
        aux = AutoModel.from_pretrained(model_name)
        config = deepcopy(aux.config)

        if encoder_hidden_layers == -1:
            encoder_hidden_layers = config.num_hidden_layers

        config._num_labels = 2
        config.label2id = {"contradiction": 0, "entailment": 1}
        config.id2label = {0: "contradiction", 1: "entailment"}
        config.classif_dropout = 0.2
        config.classifier_dropout = 0.2
        config.num_hidden_layers = encoder_hidden_layers
        model = model_class(config)

    elif model_type == "bart":
        model_name = "valhalla/distilbart-mnli-12-1"
        if multilabel:
            model_class = BartForMultiSequenceClassification
        else:
            model_class = BartForSequenceClassification

        aux = AutoModel.from_pretrained(model_name)
        config = deepcopy(aux.config)

        if encoder_hidden_layers == -1:
            encoder_hidden_layers = config.encoder_layers
        if decoder_hidden_layers == -1:
            decoder_hidden_layers = config.decoder_layers

        config._num_labels = 2
        config.label2id = {"contradiction": 0, "entailment": 1}
        config.id2label = {0: "contradiction", 1: "entailment"}
        config.encoder_layers = encoder_hidden_layers
        config.decoder_layers = decoder_hidden_layers
        config.classif_dropout = 0.2
        config.classifier_dropout = 0.2
        model = model_class(config)
    else:
        raise ValueError(f"model type {model_type} not supported")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.cls_token_id = tokenizer.cls_token_id
    model.pad_token_id = tokenizer.pad_token_id
    model.sep_token_id = tokenizer.sep_token_id

    aux_state_dict = aux.state_dict()

    tag = "model."

    new_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if k[len(tag) :] in aux_state_dict:
            if "classif" not in k:
                new_dict[k] = deepcopy(aux_state_dict[k[len(tag) :]])
            else:
                new_dict[k] = v
        else:
            new_dict[k] = v
    model.load_state_dict(new_dict)

    del aux
    return model, tokenizer
