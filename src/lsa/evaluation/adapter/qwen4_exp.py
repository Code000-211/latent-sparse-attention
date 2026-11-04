"""Qwen3.8-Flash-Next (`qwen4_exp`) - QSA on every fourth layer, four hyper-connection streams and a hashed n-gram table.

Four things differ from a standard HF decoder.

The global layers' attention is Qwen3.5's - a query projection that carries the output gate, QK-norm, and 64 of 256
head dims rotated - so `project` is Qwen3.5's and the index space is the generic one: the 192 position-free dims of each
of the 2 key heads side by side, and each group's 12 query heads summed beside them, 384 dims.

Every global layer reads through QSA, Qwen's own indexer. It mean-pools the keys into blocks of four, scores each closed
block with four query heads, and reads the 512 best blocks and the tokens of the open one - 2,051 keys a query.
transformers' indexer loops over the queries in Python and hands the attention a dense `[queries, keys]` mask, which is
hours of loop at 64k and 128 GiB of bool for a 64k chunk of a 2M document. `load` binds `select_blocks` and
`attend_through_selection` in their place - the same selection, scored a block of query rows at a time, and the same
attention read by gathering the selected keys - and switches off the mask nothing reads any more. Both assume a causal,
unpadded batch, which is every document a teacher reads.

The residual stream is four hyper-connection streams wide. A decoder layer mixes them down to one `hidden_size` input
before its attention, which is what an extraction hooks and keeps as `x`, and a span carries all four
`[1, S, 4 * hidden_size]` across a cut.

One layer adds a hashed n-gram embedding (PLE) that reads the token ids, so a span that holds it is handed the ids as
well as the streams. Its 51B-parameter table lives in that layer, so it loads with that span alone.

The decoder layer hands its attention the rotary tables of every position the cache holds rather than the chunk's,
because the indexer rotates each block key at its block's first position.
"""

import math
from importlib import import_module
from types import MethodType
from typing import Any

import torch

from lsa.evaluation.adapter.model import ModelAdapter
from lsa.evaluation.adapter.qwen3_5 import Qwen35SparseAttention
from lsa.evaluation.config import ModelConfig
from lsa.evaluation.constants import Qwen4Exp


def rotary(module: Any) -> Any:
    """Return the `apply_rotary_pos_emb` of the modelling file a module comes from.

    Args:
        module: A module of the `qwen4_exp` modelling file.

    Returns:
        Its rotary function, which takes `unsqueeze_dim` and rotates the queries alone when no keys are given.
    """
    return import_module(type(module).__module__).apply_rotary_pos_emb


def unpadded(attention_mask: Any) -> None:
    """Refuse a mask, which only a padded batch or a load that was not switched to the selection would bring.

    Args:
        attention_mask: What the decoder layer handed the attention.

    Raises:
        ValueError: There is a mask.
    """
    if attention_mask is not None:
        raise ValueError(
            "the read through QSA's selection is causal and unpadded - it takes no mask, so load the model through "
            "Qwen4ExpAdapter, which builds none, and do not pad"
        )


def top_blocks(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Return the `count` best blocks of each row, a tie going to the most recent block.

    A block every head scores below zero scores exactly zero after the rectifier, so ties are common, and `topk` breaks
    them however its kernel does - differently on another device or at another row length, so a document read a chunk
    at a time would pick other blocks than the same document read whole. Upstream leaves them to `topk` too; here the
    blocks strictly above the cut are kept, and the cut's own score fills the rest from the most recent block back.

    Args:
        scores: Block scores [B, R, N] fp32, `-inf` for a block a row does not see.
        count: Blocks to keep a row, at most N.

    Returns:
        Kept block ids [B, R, count] int64, in no particular order.
    """
    cut = scores.topk(count, dim=-1).values[..., -1:]
    above = scores > cut
    tied = scores == cut
    room = count - above.sum(dim=-1, keepdim=True, dtype=torch.int32)
    # [B, R, N] how many tied blocks sit at or after each one
    later_ties = tied.flip(-1).cumsum(dim=-1, dtype=torch.int32).flip(-1)
    kept = above | (tied & (later_ties <= room))
    # Exactly `count` blocks are kept, each marked by its own id + 1, so the top `count` of the marks are those
    marks = kept.int() * torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=torch.int32)
    return marks.topk(count, dim=-1).indices


@torch.no_grad()
def select_blocks(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any,
    past_key_values: Any,
) -> torch.Tensor:
    """Select each query's keys the way `Qwen4ExpTextQSAIndexer.forward` does, a block of query rows at a time.

    A query at position `p` sees `p + 1` tokens, of which the first `4 * ((p + 1) // 4)` make its closed blocks and the
    rest are the open one. A closed block's key is the mean of its four raw keys, normed and rotated at the block's
    first position, the same for every query that sees it - so every block key is built once here, where the upstream
    loop builds them again for each query. A query keeps its `budget / 4` best closed blocks, scored as the sum over
    its heads of the rectified dot product, and every token of its open block. Tied scores go to the most recent block
    (`top_blocks`), where upstream's are left to `topk`.

    Args:
        self: The layer's own `Qwen4ExpTextQSAIndexer`, since this is bound as its method.
        hidden_states: The attention's input [B, S, hidden_size].
        position_embeddings: The `(cos, sin)` tables of every position the cache holds, the chunk's last.
        attention_mask: None - the read is causal and unpadded.
        past_key_values: The transformers cache, or None for a document in one pass.

    Returns:
        Selected token ids [B, S, budget + 3] int32 - the kept blocks' tokens, then the open block's, `-1` where
        nothing is.

    Raises:
        ValueError: A mask arrived, the indexer has more than one key head, or the tables do not cover the cache.
    """
    unpadded(attention_mask)
    if self.index_kv_heads != 1:
        raise ValueError(f"QSA's indexer pools one key head - this one has {self.index_kv_heads}")
    apply_rotary = rotary(self)
    batch_size, query_count = hidden_states.shape[:2]
    device = hidden_states.device
    head_dim, ratio = self.index_head_dim, self.compress_ratio
    full_cos, full_sin = position_embeddings

    query_dim = self.index_n_heads * head_dim
    query, raw_keys = self.index_qk_proj(hidden_states).split([query_dim, head_dim], dim=-1)
    # [B, S, heads, head_dim] rotated queries
    query = self.q_layernorm(query.view(batch_size, query_count, -1, head_dim))
    query = apply_rotary(
        query, cos=full_cos[:, -query_count:], sin=full_sin[:, -query_count:], unsqueeze_dim=2
    ).float()
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
    key_count = raw_keys.shape[1]
    if full_cos.shape[1] != key_count:
        raise ValueError(
            f"the rotary tables cover {full_cos.shape[1]} positions where the cache holds {key_count}"
        )

    # [S] what each query sees, and how many closed blocks that makes
    visible = torch.arange(key_count - query_count, key_count, device=device) + 1
    closed = visible // ratio
    width = self.token_budget + ratio - 1
    selected = torch.full((batch_size, query_count, width), -1, dtype=torch.int32, device=device)

    # [S, ratio - 1] the open block's tokens, after the room the kept blocks take
    tail = closed[:, None] * ratio + torch.arange(ratio - 1, device=device)
    selected[..., self.block_topk * ratio :] = torch.where(tail < visible[:, None], tail, -1).int()

    block_count = key_count // ratio
    if block_count == 0:
        return selected
    # [B, N, head_dim] each closed block's key, pooled in fp32 and rotated at its first token, as upstream builds it
    pooled = raw_keys[:, : block_count * ratio].reshape(batch_size, block_count, ratio, head_dim)
    pooled = self.k_layernorm(pooled.float().mean(dim=2).to(raw_keys.dtype))
    starts = torch.arange(block_count, device=device) * ratio
    block_keys = apply_rotary(
        pooled[:, :, None], cos=full_cos[:, starts], sin=full_sin[:, starts], unsqueeze_dim=2
    )[:, :, 0]
    # [B, 1, head_dim, N] the key side of every score, built once rather than once per row
    block_keys = block_keys.float().transpose(1, 2)[:, None]

    kept_count = min(self.block_topk, block_count)
    blocks = torch.arange(block_count, device=device)
    offsets = torch.arange(ratio, device=device)
    rows = max(1, min(query_count, Qwen4Exp.INDEXER_SCORE_CELLS // (self.index_n_heads * block_count)))
    for start in range(0, query_count, rows):
        stop = min(start + rows, query_count)
        # [B, rows, N] summed over heads, upstream's order
        scores = torch.matmul(query[:, start:stop], block_keys).relu().sum(dim=2) / math.sqrt(head_dim)
        seen = closed[start:stop, None]
        scores = scores.masked_fill(blocks[None, None] >= seen, float("-inf"))
        picked = top_blocks(scores, kept_count)
        # [B, rows, kept * ratio] each kept block's tokens, `-1` for a pick past what the row has closed
        tokens = (picked[..., None] * ratio + offsets).masked_fill((picked >= seen)[..., None], -1)
        selected[:, start:stop, : kept_count * ratio] = tokens.flatten(-2).int()
    return selected


@torch.no_grad()
def attend_through_selection(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attend the way `Qwen4ExpTextAttention.forward` does, reading each query's selected keys alone.

    Upstream adds the indexer's `[queries, keys]` mask to the causal one and runs dense attention under it. Here each
    query gathers the 2,051 keys and values it selected and attends to those, which is the same attention, taken in
    fp32 rather than in the checkpoint's dtype.

    Args:
        self: The layer's own `Qwen4ExpTextAttention`, since this is bound as its method.
        hidden_states: The attention's input [B, S, hidden_size].
        position_embeddings: The `(cos, sin)` tables of every position the cache holds, the chunk's last.
        attention_mask: None - the read is causal and unpadded.
        past_key_values: The transformers cache, or None for a document in one pass.
        **kwargs: Whatever else the decoder layer passes, ignored.

    Returns:
        Attention output [B, S, hidden_size], and None for the weights.
    """
    unpadded(attention_mask)
    # Called as a module so a hook on the indexer - the native selection an extraction keeps - sees it
    selected = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
    batch_size, query_count = hidden_states.shape[:2]
    head_dim = self.head_dim
    cos, sin = (table[:, -query_count:] for table in position_embeddings)

    query, gate = torch.chunk(
        self.q_proj(hidden_states).view(batch_size, query_count, -1, head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(batch_size, query_count, -1)
    query = self.q_norm(query).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(batch_size, query_count, -1, head_dim)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(batch_size, query_count, -1, head_dim).transpose(1, 2)
    query, key = rotary(self)(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, self.layer_idx)

    # [B, T, Hkv, dh] - a position's heads together, so one gather reads them all
    keys, values = key.transpose(1, 2), value.transpose(1, 2)
    kv_heads, groups = keys.shape[2], query.shape[1] // keys.shape[2]
    width = selected.shape[-1]
    batch_index = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
    rows = max(1, Qwen4Exp.GATHER_CELLS // (width * kv_heads * head_dim))
    outputs = []
    for start in range(0, query_count, rows):
        stop = min(start + rows, query_count)
        ids = selected[:, start:stop].long()
        readable = ids >= 0
        ids = ids.clamp(min=0)
        # [B, kv, group, rows, dh] - query head h reads key head h // group, as `repeat_kv` lays them out
        block_query = query[:, :, start:stop].reshape(batch_size, kv_heads, groups, stop - start, head_dim)
        scores = torch.einsum("bkgrd,brwkd->bkgrw", block_query.float(), keys[batch_index, ids].float())
        scores = (scores * self.scaling).masked_fill(~readable[:, None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1).nan_to_num()
        attended = torch.einsum("bkgrw,brwkd->bkgrd", weights, values[batch_index, ids].float())
        outputs.append(attended.reshape(batch_size, -1, stop - start, head_dim).to(hidden_states.dtype))

    output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch_size, query_count, -1)
    return self.o_proj(output * torch.sigmoid(gate)), None


def unreachable_attention(*args: Any, **kwargs: Any) -> None:
    """Stand in for an attention kernel no switched layer calls.

    Raises:
        RuntimeError: Always - a layer got here without its read through the selection bound.
    """
    raise RuntimeError(
        f"{Qwen4Exp.SELECTION_ATTENTION} is a name for 'no mask', not a kernel - every QSA layer reads through "
        "`attend_through_selection`, so a call here is a layer that was not bound"
    )


def bind_selection(model: Any, text_config: Any) -> int:
    """Bind the blocked selection and the read through it onto every QSA layer, and build no mask for them.

    Bound per instance, so a second model loaded in the same process keeps the forward transformers shipped. The
    decoder stack is switched to an attention implementation whose mask is None for an unpadded batch - flash
    attention's - which is all the bound read takes.

    Args:
        model: The loaded model, modified in place.
        text_config: The decoder stack's config, switched in place.

    Returns:
        How many attentions were rebound.
    """
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask

    bound = 0
    for module in model.modules():
        name = type(module).__name__
        if name == Qwen4Exp.INDEXER_CLASS:
            module.forward = MethodType(select_blocks, module)
        elif name == Qwen4Exp.ATTENTION_CLASS:
            module.forward = MethodType(attend_through_selection, module)
            bound += 1
    AttentionInterface.register(Qwen4Exp.SELECTION_ATTENTION, unreachable_attention)
    AttentionMaskInterface.register(Qwen4Exp.SELECTION_ATTENTION, flash_attention_mask)
    text_config._attn_implementation = Qwen4Exp.SELECTION_ATTENTION
    return bound


class Qwen4ExpSparseAttention(Qwen35SparseAttention):
    """A Qwen3.8-Flash-Next global layer, read for its index vectors - Qwen3.5's projections and output gate."""


class Qwen4ExpAdapter(ModelAdapter):
    """Qwen3.8-Flash-Next - a multimodal hybrid checkpoint whose global layers select through QSA."""

    attention_class = Qwen4ExpSparseAttention
    layer_types = Qwen4Exp.LAYER_TYPES

    def model_class(self) -> Any:
        """Return the auto class the checkpoint registers its generation head under.

        `qwen4_exp` is a vision-language checkpoint, `Qwen4ExpForConditionalGeneration` over a text stack and a vision
        tower, so its head sits in the image-text-to-text mapping.

        Returns:
            The auto class to call `from_pretrained` on.
        """
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText

    def load(
        self,
        model_config: ModelConfig,
        device: str,
        attention: str = "eager",
        span: tuple[int, int] | None = None,
    ) -> tuple[Any, Any]:
        """Load the checkpoint, or one span of its stack, with the vision tower dropped and QSA read through its selection.

        Every caller that runs the model's own attention needs the blocked selection - upstream's loop does not finish
        a long document - so it is bound here rather than in `prepare_long_prefill`.

        Args:
            model_config: Which checkpoint, and in what dtype.
            device: A transformers device map.
            attention: The attention implementation to load with, replaced by the selection once loaded.
            span: Decoder layers `[start, stop)` to load, or None for the whole model.

        Returns:
            model: The loaded model in eval mode, its QSA layers reading through their selection.
            tokenizer: Its tokenizer.
        """
        model, tokenizer = super().load(model_config, device, attention, span)
        inner = self.inner_model(model)
        if getattr(inner, "visual", None) is not None:
            inner.visual = None
        bind_selection(model, self.text_model(model).config)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model, tokenizer

    def native_selector(self, attention: Any) -> Any | None:
        """Return the layer's QSA indexer, which once loaded gives the selected token ids `[B, S, 2051]`.

        Args:
            attention: The layer's `Qwen4ExpTextAttention`.

        Returns:
            Its indexer.
        """
        return attention.indexer

    def stack_input(self, text_model: Any, ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids into the hyper-connection streams the first decoder layer reads.

        Args:
            text_model: The decoder stack.
            ids: Token ids [1, S] int64.

        Returns:
            The embedding repeated into every stream [1, S, hc_count * hidden_size], as the stack's own forward does.
        """
        return text_model.embed_tokens(ids).repeat(1, 1, text_model.config.hc_count)

    def forward_span(
        self,
        text_model: Any,
        span: tuple[int, int],
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache: Any,
        ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run decoder layers `[start, stop)` over one chunk, as `Qwen4ExpTextModel.forward` runs them on text.

        Text positions are the same in all three rotary sections, and the tables cover every position up to the chunk's
        last, which is what the stack hands each layer. No mask is built, as for the whole stack once loaded.

        Args:
            text_model: The decoder stack.
            span: Decoder layers `[start, stop)`.
            hidden: The streams layer `start` reads [1, S, hc_count * hidden_size].
            positions: Absolute positions of the chunk [1, S] int64, which follow the cache's.
            cache: The transformers cache the chunks share, or None for a document in one pass.
            ids: The chunk's token ids [1, S] int64, which the PLE layer reads.

        Returns:
            The streams layer `stop` reads [1, S, hc_count * hidden_size].

        Raises:
            ValueError: The span holds the PLE layer and no ids were given.
        """
        stop = int(positions[0, -1]) + 1
        every = torch.arange(stop, device=hidden.device).view(1, 1, -1).expand(3, hidden.shape[0], -1)
        tables = text_model.rotary_emb(hidden, every)
        for index in range(*span):
            layer = text_model.layers[index]
            if layer.ple is not None and ids is None:
                raise ValueError(f"layer {index} embeds the token n-grams, so its span needs the token ids")
            hidden = layer(
                hidden,
                position_embeddings=tables,
                attention_mask=None,
                conv_mask=None,
                past_key_values=cache,
                ple_input_ids=ids,
            )
        return hidden
