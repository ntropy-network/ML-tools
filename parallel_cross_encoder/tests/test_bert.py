import pytest
from parallel_cross_encoder.models.bert_model import DistilBertForMultiSequenceClassification
from transformers import AutoTokenizer


@pytest.fixture
def bert_model():
    model = DistilBertForMultiSequenceClassification.from_pretrained(
        "typeform/distilbert-base-uncased-mnli", ignore_mismatched_sizes=True
    )
    model.requires_grad_(False)
    model.eval()
    return model


@pytest.fixture
def bert_tokenizer():
    return AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")


@pytest.fixture
def bert_inputs(bert_tokenizer):
    sentence_1 = [
        "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken"
    ]
    sentence_2 = [
        f"{bert_tokenizer.sep_token}This example is about{bert_tokenizer.cls_token}food{bert_tokenizer.cls_token}business{bert_tokenizer.cls_token}science{bert_tokenizer.cls_token}sports{bert_tokenizer.cls_token}technology{bert_tokenizer.pad_token}{bert_tokenizer.pad_token}{bert_tokenizer.pad_token}"
    ]
    return sentence_1, sentence_2


@pytest.fixture
def bert_inputs_shorter(bert_tokenizer):
    sentence_1 = [
        "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken"
    ]
    sentence_2 = [
        f"{bert_tokenizer.sep_token}This example is about{bert_tokenizer.cls_token}science{bert_tokenizer.cls_token}food{bert_tokenizer.pad_token}{bert_tokenizer.pad_token}{bert_tokenizer.pad_token}"
    ]
    return sentence_1, sentence_2


def test_bert_forward(bert_model, bert_tokenizer, bert_inputs, bert_inputs_shorter):
    print("inputs", bert_inputs)
    bert_model.cls_token_id = bert_tokenizer.cls_token_id
    bert_model.pad_token_id = bert_tokenizer.pad_token_id
    encoded = bert_tokenizer(
        bert_inputs[0],
        bert_inputs[1],
        truncation="only_first",
        max_length=100,
        return_tensors="pt",
        add_special_tokens=False,
    )
    print("encoded", encoded)
    out1 = bert_model(**encoded)
    print("out1", out1.logits)
    print("inputs_shorter", bert_inputs_shorter)
    encoded = bert_tokenizer(
        bert_inputs_shorter[0],
        bert_inputs_shorter[1],
        truncation="only_first",
        max_length=100,
        return_tensors="pt",
        add_special_tokens=False,
    )
    print("encoded", encoded)
    out2 = bert_model(**encoded)
    print("out2", out2.logits)
    assert out1.logits[0, 0].item() == pytest.approx(
        out2.logits[0, 1].item(), 0.0001
    ), "Food label got different logit"
    assert out1.logits[0, 2].item() == pytest.approx(
        out2.logits[0, 0].item(), 0.0001
    ), "Intra account transfer label got different logit"
