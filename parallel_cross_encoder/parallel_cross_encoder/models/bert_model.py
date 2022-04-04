import copy
import math

import torch
from parallel_cross_encoder.models.multilabel_helper import BertMultiLabelHelper
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.models.distilbert.modeling_distilbert import (
    BaseModelOutput, DistilBertForSequenceClassification, DistilBertModel,
    Embeddings, MultiHeadSelfAttention, SequenceClassifierOutput, Transformer,
    TransformerBlock)


class MultiLabelsEmbeddings(Embeddings):
    """
    Positional embedding for the case of a transaction followed by multiples labels
    (and also multiples [CLS] tokens)
    """

    def forward(self, input_ids, position_mask):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        position_ids = BertMultiLabelHelper.get_position_ids(position_mask)
        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(MultiHeadSelfAttention):
    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)
        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        if head_mask is not None:
            head_mask = head_mask == 0
            scores.masked_fill_(head_mask, -float("inf"))
        else:
            scores.masked_fill_(
                mask, -float("inf")
            )  # (bs, n_heads, q_length, k_length)
        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class TransformerBlock(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)
        assert config.dim % config.n_heads == 0
        self.attention = MultiHeadSelfAttention(config)


class Transformer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state,
                attn_mask=attn_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class DistilBertModel(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = Transformer(config)  # Encoder
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class DistilBertForMultiSequenceClassification(DistilBertForSequenceClassification):
    """
    BertMultiLabel model, can predict (0/1) on several labels on a single forward pass
    """

    REDUCTION_DIM = 128

    def __init__(self, config):
        config.num_labels = 1
        super(DistilBertForMultiSequenceClassification, self).__init__(config)
        self.config.use_cache = False
        self.config.label2id = {"contradiction": 0, "entailment": 1}
        self.config.id2label = {0: "contradiction", 1: "entailment"}

        if (
            not hasattr(self.config, "classif_dropout")
            or self.config.classif_dropout is None
        ):
            self.config.classif_dropout = 0.3
        D = self.config.dim
        if not hasattr(self.config, "reduction_dim"):
            self.config.reduction_dim = (
                DistilBertForMultiSequenceClassification.REDUCTION_DIM
            )
        H = self.config.reduction_dim

        self.embeddings = MultiLabelsEmbeddings(config)  # Embeddings

        self.distilbert = DistilBertModel(config)

        self.pre_classifier = nn.Linear(D, D)
        self.reduce_dim = nn.Linear(D, H)
        self.classifier = nn.Linear(H, 1)
        self.cls_token_id = 101
        self.pad_token_id = 0
        self.loss_fct = BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None or attention_mask is None:
            raise RuntimeError(
                "Bad arguments for forward function of BertMultiLabel (input_ids & attention_mask required)"
            )

        position_mask = BertMultiLabelHelper.get_segmentation(
            input_ids, self.cls_token_id, self.pad_token_id
        )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, position_mask)
        # we automatically construct head_mask from the position_mask
        assert (
            head_mask is None
        ), "head mask has to be None, it is auto computed from the position_mask argument"
        head_mask = BertMultiLabelHelper.get_attention_mask(
            position_mask
        )  # (bs x seq length x seq length)

        # let's expand this tensor into a 4d tensor for the bert Attention class ([batch x num_heads x seq_length x seq_length])
        head_mask = head_mask.unsqueeze(1).expand(
            -1, self.config.num_attention_heads, -1, -1
        )

        distilbert_output = self.distilbert(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        cls_mask = input_ids == self.cls_token_id
        if len(torch.unique(cls_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <cls> tokens.")
        pooled_output = hidden_state[cls_mask].view(
            hidden_state.shape[0], -1, hidden_state.shape[-1]
        )  # (bs, nb_labels, dim)

        x = self.pre_classifier(pooled_output)  # (bs * nb_labels, dim)
        x = nn.Tanh()(x)  # (bs * nb_labels, dim)
        x = self.dropout(x)  # (bs * nb_labels, dim)
        hidden_state = self.reduce_dim(
            x + pooled_output
        )  # (bs * nb_labels, reduced_dim)
        logits = self.classifier(hidden_state).squeeze(-1)  # (bs * nb_labels)

        if logits.shape[-1] == 1:
            logits = torch.squeeze(logits)

        # note: BCE expect labels to be float
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.float().view(-1))
        else:
            loss = None

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
