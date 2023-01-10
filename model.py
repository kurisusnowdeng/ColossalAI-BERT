"""
Adapted from huggingface modeling_bert.py. Change the necessary part to use Colossolai.
"""
import math

import torch
from colossalai import nn as col_nn
from colossalai.kernel.cuda_native.scaled_softmax import AttnMaskType, FusedScaleMaskSoftmax
from colossalai.kernel.cuda_native.flash_attention import MemoryEfficientAttention
from colossalai.nn.layer.utils import divide
from colossalai.utils import get_current_device
from einops import rearrange
from flash_attn.flash_attention import FlashAttention
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput)


@torch.jit.script
def bias_gelu_fwd(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def bias_gelu_bwd(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bias):
        ctx.save_for_backward(x, bias)
        return bias_gelu_fwd(bias, x)

    @staticmethod
    def backward(ctx, grad_output):
        x, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, bias, x)
        return tmp, tmp


bias_gelu = GeLUFunction.apply


@torch.jit.script
def bias_dropout_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    out = torch.nn.functional.dropout(torch.add(x, bias), p=prob, training=training)
    out = residual + out
    return out


def core_attention(
    attention_scores: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout_prob: float,
    attention_head_size: int,
    training: bool,
) -> torch.Tensor:

    attention_scores = attention_scores / math.sqrt(attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = torch.nn.functional.dropout(attention_probs, p=dropout_prob, training=training)

    return attention_probs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = col_nn.Linear(
            config.hidden_size,
            config.hidden_size,
            skip_bias_add=True,
        )
        self.dropout = config.hidden_dropout_prob
        # self.dropout = col_nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = col_nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states, bias = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states) + input_tensor
        hidden_states = bias_dropout_add(hidden_states, bias, input_tensor, self.dropout, self.training)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                             f"heads ({config.num_attention_heads})")
        self.flash_attn = config.flash_attention
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = divide(config.hidden_size, config.num_attention_heads)

        self.query_key_value = col_nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size * 3)

        if self.flash_attn:
            self.attention_func = FlashAttention(softmax_scale=math.sqrt(self.attention_head_size),
                                                 attention_dropout=config.attention_probs_dropout_prob)
        else:
            self.attention_func = FusedScaleMaskSoftmax(input_in_fp16=True,
                                                        input_in_bf16=False,
                                                        attn_mask_type=AttnMaskType.padding,
                                                        scaled_masked_softmax_fusion=True,
                                                        mask_func=None,
                                                        softmax_in_fp32=True,
                                                        scale=math.sqrt(self.attention_head_size))
            self.dropout = col_nn.Dropout(config.attention_probs_dropout_prob)
            # self.attention_func = MemoryEfficientAttention(config.hidden_size, self.num_attention_heads, 0.0)

    def forward(self, hidden_states, attention_mask=None):
        qkv = self.query_key_value(hidden_states)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = divide(all_head_size, self.attention_head_size)
        if self.flash_attn:
            qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=num_attention_heads)
            context, _ = self.attention_func(qkv, key_padding_mask=attention_mask, causal=False)
            context = rearrange(context, 'b s h d -> b s (h d)')
        else:
            new_qkv_shape = qkv.shape[:-1] + \
                (num_attention_heads, 3 * self.attention_head_size)
            qkv = qkv.view(new_qkv_shape)
            qkv = qkv.permute((0, 2, 1, 3))
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            attention_scores = torch.matmul(q, k.transpose(-1, -2))

            attention_probs = self.attention_func(
                attention_scores,
                attention_mask,
            )
            attention_probs = self.dropout(attention_probs)

            context = torch.matmul(attention_probs, v)

            context = context.permute(0, 2, 1, 3).contiguous()
            new_context_shape = context.size()[:-2] + (all_head_size, )
            context = context.view(new_context_shape)

            # context = self.attention_func(q, k, v, None)

        return context


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = col_nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            skip_bias_add=True,
        )
        # self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states, bias = self.dense(hidden_states)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = bias_gelu(hidden_states, bias)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = col_nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            skip_bias_add=True,
        )
        # self.dropout = col_nn.Dropout(config.hidden_dropout_prob)
        self.dropout = config.hidden_dropout_prob
        self.LayerNorm = col_nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states, bias = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states) + input_tensor
        hidden_states = bias_dropout_add(hidden_states, bias, input_tensor, self.dropout, self.training)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.gradient_checkpointing = False

    def layer_forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output

    def forward(self, hidden_states, attention_mask=None):
        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.layer_forward,
                hidden_states,
                attention_mask,
            )
        else:
            hidden_states = self.layer_forward(hidden_states, attention_mask)

        return hidden_states


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = col_nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = col_nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = col_nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = col_nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = col_nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = torch.arange(config.max_position_embeddings, device=get_current_device()).expand((1, -1))
        self.token_type_ids = torch.zeros(self.position_ids.size(), dtype=torch.long, device=get_current_device())

    def forward(self, input_ids, token_type_ids=None):
        input_shape = input_ids.size()
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length].expand(batch_size, -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = col_nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        if attention_mask is not None:
            attention_mask = col_nn.partition_batch(attention_mask)

            if self.config.flash_attention:
                attention_mask = attention_mask.bool()
            else:
                batch_size = attention_mask.shape[0]
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask_b1s = attention_mask.unsqueeze(1)
                # [b, s, 1]
                attention_mask_bs1 = attention_mask.unsqueeze(2)
                # [b, s, s]
                attention_mask_bss = attention_mask_b1s * attention_mask_bs1
                # [b, 1, s, s]
                extended_attention_mask = attention_mask_bss.unsqueeze(1)
                extended_attention_mask = (extended_attention_mask < 0.5)
                attention_mask = extended_attention_mask.to(dtype=embedding_output.dtype)

        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask)

        return BaseModelOutput(last_hidden_state=encoder_outputs)


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = col_nn.Linear(
            config.hidden_size,
            config.hidden_size,
            gather_output=True,
            skip_bias_add=True,
        )
        # self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = col_nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states, bias = self.dense(hidden_states)
        # hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = bias_gelu(hidden_states, bias)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, weight):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = col_nn.Classifier(config.hidden_size, config.vocab_size, weight=weight, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertMaskedLMLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = col_nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # Flatten the tokens
        return self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))


class BertForMaskedLM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.bert = BertModel(config)
        self.cls = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        prediction_scores = self.cls(outputs.last_hidden_state)

        return MaskedLMOutput(logits=prediction_scores)


class BertForSequenceClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.pooler = BertPooler(config)

        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = col_nn.Dropout(classifier_dropout)
        self.classifier = col_nn.Classifier(config.hidden_size, config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.pooler(outputs.last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(logits=logits)
