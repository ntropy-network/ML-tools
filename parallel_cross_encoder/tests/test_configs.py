from parallel_cross_encoder.configs import TrainConfigs


def test_config():
    params = {
        "model_type": "bart",
        "num_labels_per_sample": 10,
        "output_dir": "./output",
        "batch_size": 32,
        "weight_decay": 0.0001,
        "dataloader_num_workers": 4,
        "dataloader_drop_last": False,
        "early_stopping_epochs": 6,
    }
    c = TrainConfigs(**params)
    print(c)
    assert c.model_type == "bart"
    assert c.batch_size == 32
