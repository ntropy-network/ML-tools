import torch
from parallel_cross_encoder.configs import TrainConfigs
from parallel_cross_encoder.models.lightning_model import LightningModel


def test_pl_model():
    input = "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken. [SEP] This example is about [CLS] war [CLS] sport [CLS] shopping [CLS] internet [CLS] software [CLS] crypto [CLS] world"
    params = {
        "model_type": "bert",
        "negative_labels_per_sample": 10,
        "logging_steps": 10,
        "output_dir": "./output",
        "batch_size": 32,
        "weight_decay": 0.0001,
        "dataloader_num_workers": 4,
        "early_stopping_epochs": 6,
    }
    configs = TrainConfigs(**params)
    model = LightningModel(configs)

    enc_input = model.tokenizer(
        input, truncation=True, padding=True, add_special_tokens=False
    )
    print("enc_input", enc_input)
    print(model.tokenizer.decode(enc_input["input_ids"]))
    output = model.forward(
        **{k: torch.tensor(v).unsqueeze(0) for k, v in enc_input.items()}
    )
    print("logits", output.logits)
    assert len(torch.flatten(output.logits)) == 7  # 7 labels => 7 logits
