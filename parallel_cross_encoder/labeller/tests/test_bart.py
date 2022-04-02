import pytest
from labeller.src.models.bart_model import BartForMultiSequenceClassification
from transformers import AutoTokenizer


@pytest.fixture
def bart_model():
    model = BartForMultiSequenceClassification.from_pretrained(
        "valhalla/distilbart-mnli-12-3", ignore_mismatched_sizes=True
    )
    model.requires_grad_(False)
    model.eval()
    return model


@pytest.fixture
def bart_tokenizer():
    return AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-3")


@pytest.fixture
def bart_inputs(bart_tokenizer):
    sentence_1 = [
        "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken"
    ]
    sentence_2 = [
        f"{bart_tokenizer.sep_token}This example is about{bart_tokenizer.sep_token}food{bart_tokenizer.cls_token}business{bart_tokenizer.cls_token}science{bart_tokenizer.cls_token}sports{bart_tokenizer.cls_token}technology{bart_tokenizer.cls_token}{bart_tokenizer.pad_token}{bart_tokenizer.pad_token}{bart_tokenizer.pad_token}"
    ]
    return sentence_1, sentence_2


@pytest.fixture
def bart_inputs_shorter(bart_tokenizer):
    sentence_1 = [
        "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken"
    ]
    sentence_2 = [
        f"{bart_tokenizer.sep_token}This example is about{bart_tokenizer.sep_token}science{bart_tokenizer.cls_token}{bart_tokenizer.pad_token}{bart_tokenizer.pad_token}{bart_tokenizer.pad_token}"
    ]
    return sentence_1, sentence_2


def test_bart_forward(bart_model, bart_tokenizer, bart_inputs, bart_inputs_shorter):
    print("inputs", bart_inputs)
    bart_model.cls_token_id = bart_tokenizer.cls_token_id
    bart_model.pad_token_id = bart_tokenizer.pad_token_id
    encoded = bart_tokenizer(
        bart_inputs[0],
        bart_inputs[1],
        truncation="only_first",
        max_length=100,
        return_tensors="pt",
        add_special_tokens=False,
    )
    print("encoded", encoded)
    out1 = bart_model(**encoded)
    print("out1", out1.logits)
    print("inputs_shorter", bart_inputs_shorter)
    encoded = bart_tokenizer(
        bart_inputs_shorter[0],
        bart_inputs_shorter[1],
        truncation="only_first",
        max_length=100,
        return_tensors="pt",
        add_special_tokens=False,
    )
    print("encoded", encoded)
    out2 = bart_model(**encoded)
    print("out2", out2.logits)
    # assert out1.logits[0, 0].item() == pytest.approx(
    #     out2.logits[0, 1].item(), 0.0001
    # ), "Food label got different logit"
    assert out1.logits[0, 2].item() == pytest.approx(
        out2.logits[0, 0].item(), 0.0001
    ), "Intra account transfer label got different logit"
