"""The Qwen3.8-Flash-Next adapter, against the real transformers module rather than a fake of it.

The checkpoint is 350 GB, but its stack is ordinary python: the same `Qwen4ExpTextModel` is built here at a size that
fits in a test - QSA on two of six layers, two hyper-connection streams, an n-gram layer - with random weights. The
bound selection and read are checked against upstream's own indexer and attention, and the span against the stack's
own forward.

QSA's scores are rectified, so blocks every head scores below zero tie at zero and upstream's `topk` breaks those ties
however it does. Where a test compares against upstream, the indexer is given a direction every query and key share,
which makes every score positive and the ranking strict.

Skipped without the evaluation extra, which is what CI installs.
"""

import copy

import pytest
import torch

pytest.importorskip("transformers")

from lsa import constants as const
from lsa.core import RotaryLayout
from lsa.evaluation.adapter import adapter_for
from lsa.evaluation.adapter.qwen4_exp import (
    Qwen4ExpAdapter,
    Qwen4ExpSparseAttention,
    attend_through_selection,
    bind_selection,
    select_blocks,
    top_blocks,
)
from lsa.evaluation.config import ModelConfig
from lsa.sglang.settings.design import IndexerConfig
from lsa.training.distillation.data.teacher_capture import capture_layer_inputs, hook_native_selection

TOKENS = 80
HIDDEN = 64
# 4 blocks of 4 a query - most rows here see more, so the selection is a real one
BUDGET = 16
RATIO = 4
INDEXER_DIM = 16
# A quarter of a 32-dim head, as the checkpoint rotates a quarter of its 256
ROTATED = 8
QSA_LAYERS = [2, 5]
SPEC = ModelConfig(name="Qwen/Qwen3.8-Flash-Next", dtype="float32")
INDEXER = IndexerConfig(window_size=8, sink_size=2, top_k=TOKENS, bucket_size=4, bucket_cap=8, pool_size=None)


def build():
    """A Qwen3.8-Flash-Next-shaped text stack small enough to run on CPU, every weight nudged off its init.

    Six layers, the third and sixth QSA, and the n-gram layer on the second as the checkpoint has it.
    """
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    torch.manual_seed(0)
    config = Qwen4ExpTextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_hidden_layers=6,
        full_attention_interval=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[2],
        ple_embed_dim=16,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=INDEXER_DIM,
        indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO,
        max_position_embeddings=4096,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        attn_implementation="eager",
    )
    model = Qwen4ExpTextModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    return model


def bound(model):
    """The same stack with the selection bound, as `Qwen4ExpAdapter.load` leaves it."""
    copied = copy.deepcopy(model)
    bind_selection(copied, copied.config)
    return copied


def shared_direction(model):
    """Give every QSA query and key a large component along one direction of the input, past the rotated dims.

    Returns:
        The direction [hidden_size], which an input carries to make every indexer score positive.
    """
    direction = torch.nn.functional.normalize(torch.randn(HIDDEN), dim=0)
    with torch.no_grad():
        for index in QSA_LAYERS:
            weight = model.layers[index].self_attn.indexer.index_qk_proj.weight
            weight.view(-1, INDEXER_DIM, HIDDEN)[:, ROTATED:] += 2.0 * direction
    return direction


def tables(model, length):
    """The `(cos, sin)` tables of positions `[0, length)`, text positions in all three sections."""
    positions = torch.arange(length).view(1, 1, -1).expand(3, 1, -1)
    return model.rotary_emb(torch.zeros(1, length, HIDDEN), positions)


def causal_mask(model, length):
    from transformers.masking_utils import create_causal_mask

    hidden = torch.zeros(1, length, HIDDEN)
    return create_causal_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=torch.arange(length)[None],
        allow_is_causal_skip=False,
    )


def as_mask(selected, length):
    """Selected ids [1, S, W] as the [S, length] bool mask upstream's indexer returns."""
    mask = torch.zeros(selected.shape[1], length + 1, dtype=torch.bool)
    ids = torch.where(selected[0] >= 0, selected[0].long(), length)
    return mask.scatter(1, ids, True)[:, :length]


def tie_free_input(model):
    direction = shared_direction(model)
    return torch.randn(1, TOKENS, HIDDEN) * 0.5 + 5.0 * direction


@torch.no_grad()
def test_the_blocked_selection_is_qsas_own():
    model = build()
    hidden = tie_free_input(model)
    indexer = model.layers[2].self_attn.indexer
    native = indexer(hidden, tables(model, TOKENS), causal_mask(model, TOKENS), None)

    selected = select_blocks(indexer, hidden, tables(model, TOKENS), None, None)

    assert selected.shape == (1, TOKENS, BUDGET + RATIO - 1)
    assert torch.equal(as_mask(selected, TOKENS), native[0, 0] == 0)


@torch.no_grad()
def test_the_selection_a_chunk_at_a_time_is_the_selection_of_the_whole():
    from transformers.cache_utils import DynamicCache

    model = build()
    hidden = tie_free_input(model)
    indexer = model.layers[2].self_attn.indexer
    whole = select_blocks(indexer, hidden, tables(model, TOKENS), None, None)

    cache = DynamicCache(config=model.config)
    pieces = [
        select_blocks(
            indexer, hidden[:, start : start + 24], tables(model, min(start + 24, TOKENS)), None, cache
        )
        for start in range(0, TOKENS, 24)
    ]

    assert torch.equal(torch.cat(pieces, dim=1), whole)


@torch.no_grad()
def test_the_read_through_the_selection_is_qsas_attention():
    model = build()
    hidden = tie_free_input(model)
    native = model.layers[2].self_attn
    expected, _ = native(hidden, tables(model, TOKENS), causal_mask(model, TOKENS))

    read = bound(model).layers[2].self_attn
    actual, weights = attend_through_selection(read, hidden, tables(model, TOKENS), None)

    assert weights is None
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_a_tie_goes_to_the_most_recent_block():
    scores = torch.tensor([[[3.0, 0.0, 0.0, 1.0, 0.0, float("-inf")]]])

    picked = top_blocks(scores, 3)

    assert sorted(picked[0, 0].tolist()) == [0, 3, 4]


def test_a_row_that_sees_fewer_blocks_than_it_keeps_marks_the_rest_unread():
    model = build()
    indexer = model.layers[2].self_attn.indexer
    with torch.no_grad():
        selected = select_blocks(indexer, torch.randn(1, 9, HIDDEN), tables(model, 9), None, None)

    # Position 8 sees 9 tokens - two closed blocks and one open token
    assert sorted(selected[0, 8][selected[0, 8] >= 0].tolist()) == list(range(9))
    # Position 2 sees no closed block, only its own open one
    assert sorted(selected[0, 2][selected[0, 2] >= 0].tolist()) == [0, 1, 2]


def test_a_mask_is_refused():
    model = build()
    with pytest.raises(ValueError, match="causal and unpadded"):
        select_blocks(
            model.layers[2].self_attn.indexer,
            torch.randn(1, 8, HIDDEN),
            tables(model, 8),
            causal_mask(model, 8),
            None,
        )


@torch.no_grad()
def test_a_document_read_in_chunks_is_the_document_read_whole():
    from transformers.cache_utils import DynamicCache

    model = bound(build())
    ids = torch.randint(2, 128, (1, TOKENS))
    whole = model(input_ids=ids).last_hidden_state

    cache = DynamicCache(config=model.config)
    pieces = [
        model(
            input_ids=ids[:, start : start + 24],
            position_ids=torch.arange(start, min(start + 24, TOKENS))[None],
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        for start in range(0, TOKENS, 24)
    ]

    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, atol=1e-5, rtol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("chunk", [TOKENS, 24])
def test_the_span_runs_the_stack_as_its_own_forward_does(chunk):
    from transformers.cache_utils import DynamicCache

    model = bound(build())
    ids = torch.randint(2, 128, (1, TOKENS))
    streams = []
    hook = model.hyper_connection_mixer.register_forward_pre_hook(
        lambda module, args: streams.append(args[0])
    )
    model(input_ids=ids)
    hook.remove()

    adapter = Qwen4ExpAdapter()
    cache = DynamicCache(config=model.config) if chunk < TOKENS else None
    pieces = []
    for start in range(0, TOKENS, chunk):
        stop = min(start + chunk, TOKENS)
        hidden = adapter.stack_input(model, ids[:, start:stop])
        positions = torch.arange(start, stop)[None]
        pieces.append(adapter.forward_span(model, (0, 6), hidden, positions, cache, ids[:, start:stop]))

    torch.testing.assert_close(torch.cat(pieces, dim=1), streams[0], atol=1e-5, rtol=1e-5)


def test_a_span_holding_the_ngram_layer_needs_the_token_ids():
    model = bound(build())
    adapter = Qwen4ExpAdapter()
    hidden = adapter.stack_input(model, torch.randint(2, 128, (1, 8)))
    with pytest.raises(ValueError, match="token ids"):
        adapter.forward_span(model, (0, 2), hidden, torch.arange(8)[None], None)


def test_the_qsa_layers_are_the_ones_swapped():
    model = build()
    assert Qwen4ExpAdapter().full_attention_layers(model.config, SPEC) == QSA_LAYERS


def test_the_rotation_is_measured_as_a_prefix_of_a_quarter_of_the_head():
    model = build()
    assert Qwen4ExpAdapter().rotary_layout(model, SPEC, QSA_LAYERS) == RotaryLayout(
        const.Rotary.PREFIX, ROTATED
    )


def test_the_checkpoint_type_resolves_to_this_adapter(monkeypatch):
    from lsa.evaluation import adapter

    monkeypatch.setattr(adapter, "model_type_of", lambda name: "qwen4_exp")
    assert isinstance(adapter_for(SPEC), Qwen4ExpAdapter)


@torch.no_grad()
def test_the_projection_is_the_attention_the_model_runs():
    model = build()
    hidden = torch.randn(1, TOKENS, HIDDEN)
    native = model.layers[2].self_attn
    reader = Qwen4ExpSparseAttention(native, INDEXER, RotaryLayout(const.Rotary.PREFIX, ROTATED))

    projection = reader.project(hidden)
    q, k = reader.rotated(projection, tables(model, TOKENS))
    # Upstream's attention over the same input with a mask that keeps every causal key reads the same heads
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k.repeat_interleave(2, dim=1)) * native.scaling
    causal = torch.ones(TOKENS, TOKENS, dtype=torch.bool).tril()
    weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    values = torch.einsum("bhqk,bhkd->bqhd", weights, projection.v.repeat_interleave(2, dim=1)).flatten(-2)
    expected = native.o_proj(values * torch.sigmoid(projection.gate))

    dense = copy.deepcopy(model)
    dense.layers[2].self_attn.indexer.token_budget = TOKENS
    dense.layers[2].self_attn.indexer.block_topk = TOKENS // RATIO
    actual, _ = dense.layers[2].self_attn(hidden, tables(model, TOKENS), causal_mask(model, TOKENS))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_a_chunked_capture_keeps_one_table_a_position():
    model = bound(build())
    ids = torch.randint(2, 128, (1, TOKENS))

    whole = capture_layer_inputs(model, QSA_LAYERS, ids)
    chunked = capture_layer_inputs(model, QSA_LAYERS, ids, prefill=24)

    for index in QSA_LAYERS:
        hidden, (cos, sin) = chunked[index]
        assert hidden.shape == (1, TOKENS, HIDDEN)
        assert cos.shape[1] == sin.shape[1] == TOKENS
        torch.testing.assert_close(hidden, whole[index][0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cos, whole[index][1][0])


@torch.no_grad()
def test_the_native_selection_is_kept_at_the_sampled_positions():
    model = bound(build())
    ids = torch.randint(2, 128, (1, TOKENS))
    positions = torch.tensor([3, 40, 79])

    selection, hooks = hook_native_selection(Qwen4ExpAdapter(), model, QSA_LAYERS, positions)
    capture_layer_inputs(model, QSA_LAYERS, ids, prefill=24)
    for hook in hooks:
        hook.remove()

    picked = selection.of(2, TOKENS)
    assert picked.shape == (3, BUDGET + RATIO - 1)
    # The last query keeps four closed blocks of the 20 it sees, and has no open block
    assert (picked[2] >= 0).sum() == BUDGET
    assert sorted(picked[0][picked[0] >= 0].tolist()) == [0, 1, 2, 3]


def test_yarn_past_the_native_context_keeps_the_rotated_prefix():
    # The card serves the model under YaRN to 2M, which scales the tables but must not move the rotated dims
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

    config = build().config
    config.rope_parameters = {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "factor": 9.0,
        "original_max_position_embeddings": 512,
    }
    model = Qwen4ExpTextModel(config).eval()

    assert model.rotary_emb.rope_type == "yarn"
    assert Qwen4ExpAdapter().rotary_layout(model, SPEC, QSA_LAYERS) == RotaryLayout(
        const.Rotary.PREFIX, ROTATED
    )
