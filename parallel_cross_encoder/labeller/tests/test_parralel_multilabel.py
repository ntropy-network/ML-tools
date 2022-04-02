import atexit
from numpy import dtype
import torch
from labeller.src.models.multilabel_helper import BertMultiLabelHelper


def test_attention_mask():
    position_mask = torch.Tensor([[0, 0, 0, 1, 1, 2, 2, -1]])
    attention = BertMultiLabelHelper.get_attention_mask(position_mask)
    print(attention)
    assert torch.equal(
        attention,
        torch.Tensor(
            [
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            ]
        ),
    )


def test_position_ids():
    position_mask = torch.Tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, -1], [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, -1]]
    )
    position_ids = BertMultiLabelHelper.get_position_ids(position_mask)
    print("position_ids:")
    print(position_ids)
    assert torch.equal(
        position_ids,
        torch.Tensor(
            [[0, 1, 2, 3, 4, 5, 3, 4, 3, 4, 0], [0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 0]],
        ).to(torch.long),
    )


def test_segmentation():
    cls_token_id = 0
    pad_token_id = 1
    input_ids = torch.Tensor(
        [
            [
                1509,
                5016,
                17971,
                2,
                255,
                43026,
                59,
                1437,
                0,
                689,
                1437,
                0,
                689,
                1437,
                0,
                689,
                1437,
                1,
                1,
                1,
                1,
            ]
        ]
    )
    segmentation = BertMultiLabelHelper.get_segmentation(
        input_ids, cls_token_id, pad_token_id
    )
    print(segmentation)
    assert torch.equal(
        segmentation,
        torch.Tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, -1, -1, -1, -1]]
        ).to(torch.long),
    )
