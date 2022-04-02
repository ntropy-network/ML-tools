import torch


class BartMultiLabelHelper:
    PAD = -1

    @staticmethod
    def get_segmentation(inputs_id, cls_token_id, sep_token_id, pad_token_id):
        """
        For a given encoding, returns the splitting of text input and each labels
        ex:
        Total gas station [SEP] This example is about [SEP] gas station [CLS] food and drinks [CLS] PAD PAD
        =>
        0     0   0        0      0     0     0   0      0    1    1       1    2   2     2    2  -1  -1
        Note: help to calculate the pos encoding and the attention mask logits
        """

        def reverse_cumsum(x):
            return x + torch.sum(x, dim=1, keepdims=True) - torch.cumsum(x, dim=1)

        # topics
        cls_pos = (inputs_id == cls_token_id).int()
        segments = reverse_cumsum(cls_pos)
        # transaction segment
        cls_seg = (inputs_id == sep_token_id).int()
        mask_transaction = reverse_cumsum(cls_seg)
        mask_transaction = mask_transaction >= 1
        segments[mask_transaction] = 0
        # padding
        segments[inputs_id == pad_token_id] = BartMultiLabelHelper.PAD
        return segments

    @staticmethod
    def get_attention_mask(position_mask, position_mask_2=None, fill="diag"):
        """
        fill parameter:
            - 'diag' -> fill the diagonal matrix with 0 to have at least a token to attend for padding token in encoding
            - 'first' -> fill the first col matrix with 0 to have at least the first token to attend for padding token in decoder_encoder_attention
        Retrieve the attention mask of size (Batch size, sequence length, sequence_length), for which
        the tokens of the text input see each other and the labels tokens sees also the text input tokens + their own
        labels tokens.
        """
        bs = position_mask.shape[0]
        if position_mask_2 is None:
            position_mask_2 = position_mask
        # init with attention enabled between input text tokens (first part of the sequences)
        res = (
            ((position_mask != -1).float())
            .reshape(bs, -1, 1)
            .matmul((position_mask_2 == 0).float().reshape(bs, 1, -1))
        )
        nb_labels = int(torch.max(position_mask).item())
        # for each label, we add the label tokens specific attentions
        for label in range(1, nb_labels + 1):
            res += (
                ((position_mask == label).float())
                .reshape(bs, -1, 1)
                .matmul((position_mask_2 == label).float().reshape(bs, 1, -1))
            )

        # Important to not have only -infs in the padding attention mask (would result in nans)
        if fill == "diag":
            res = res.logical_or(
                torch.eye(res.shape[1]).expand(res.shape).to(res.device)
            ).float()
        elif fill == "first":
            res[:, :, 0] = 1
        else:
            raise RuntimeError(
                f"parameter fill should be 'diag' or 'first' and not {fill}"
            )
        res.masked_fill_(res == 0, float("-inf"))
        res.masked_fill_(res == 1, 0)
        res = res.unsqueeze(1)
        return res

    @staticmethod
    def get_position_ids(position_mask):
        """
        Helper function that convert a position mask into positions for pos encoding
        ex:
        position mask ::
               [[0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, -1, -1, -1],
                [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, -1]]
        out ::
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 6, 7, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 0]])
        """
        seq_length = position_mask.size(1)
        res = torch.zeros_like(
            position_mask, dtype=torch.long, device=position_mask.device
        )
        nb_labels = int(torch.max(position_mask).item())
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=position_mask.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            position_mask
        )  # (bs, max_seq_length)
        res[position_mask == 0] = position_ids[
            position_mask == 0
        ]  # (bs, max_seq_length)
        for label in range(nb_labels, 0, -1):
            curr_position_ids = position_ids - torch.sum(
                (position_mask > label), dim=-1
            ).reshape(-1, 1)
            res[position_mask == label] = curr_position_ids[position_mask == label]

        return res


class BertMultiLabelHelper:
    PAD = -1

    @staticmethod
    def get_segmentation(inputs_id, cls_token_id, pad_token_id):
        """
        For a given encoding, returns the splitting of text input and each labels
        ex:
        Ant colony hits Australia [SEP] This example is about [CLS] sport [CLS] food PAD PAD

        ==>

        0     0         0     0     0     0     0     0   0     1     1     2     2   -1  -1
        Note: help to calculate the pos encoding and the attention mask logits
        """
        cls_pos = (inputs_id == cls_token_id).int()
        segments = torch.cumsum(cls_pos, axis=1)
        segments[inputs_id == pad_token_id] = BertMultiLabelHelper.PAD
        return segments

    @staticmethod
    def get_attention_mask(segments_mask):
        """
        Retrieve the attention mask of size (Batch size, sequence length, sequence_length), for which
        the tokens of the text input see each other and the labels tokens sees also the input text tokens + their own
        labels tokens.
        """
        bs = segments_mask.shape[0]
        # init with attention enabled between input text tokens (first part of the sequences)
        res = (
            ((segments_mask != -1).float())
            .reshape(bs, -1, 1)
            .matmul((segments_mask == 0).float().reshape(bs, 1, -1))
        )
        nb_labels = int(torch.max(segments_mask).item())
        # for each label, we add the label tokens specific attentions
        for label in range(1, nb_labels + 1):
            res += (
                ((segments_mask == label).float())
                .reshape(bs, -1, 1)
                .matmul((segments_mask == label).float().reshape(bs, 1, -1))
            )

        # Important to not have only -infs in the padding attention mask (would result in nans)
        res = res.logical_or(
            torch.eye(res.shape[1]).expand(res.shape).to(res.device)
        ).float()
        return res

    @staticmethod
    def get_position_ids(segments_mask):
        """
        Helper function that convert a segment mask into positions for pos encoding
        ex:
        position mask ::
               [[0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, -1, -1, -1],
                [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, -1]]
        out ::
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 6, 7, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 0]])
        """
        seq_length = segments_mask.size(1)
        res = torch.zeros_like(
            segments_mask, dtype=torch.long, device=segments_mask.device
        )
        nb_labels = int(torch.max(segments_mask).item())
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=segments_mask.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            segments_mask
        )  # (bs, max_seq_length)
        res[segments_mask == 0] = position_ids[
            segments_mask == 0
        ]  # (bs, max_seq_length)
        for label in range(1, nb_labels + 1):
            res[segments_mask == label] = position_ids[segments_mask == label]
            position_ids = position_ids - torch.sum(
                segments_mask == label, dim=1
            ).reshape(-1, 1)
        return res
