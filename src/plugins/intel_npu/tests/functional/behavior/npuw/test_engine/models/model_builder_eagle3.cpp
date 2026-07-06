// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Eagle3 speculative decoding model support.
//
// Mirrors the models NPUW receives after GenAI's Eagle3 preprocessing
// (openvino.genai eagle3_model_transforms.cpp):
//   - Target: a regular decoder LLM whose selected layer outputs are captured,
//     concatenated ("eagle3_hidden_states_concat"), projected by the draft's fc
//     ("eagle3_hidden_state_fc") and exposed as an extra "last_hidden_state"
//     output marked with "manually_added_output" rt_info.
//   - Draft: a single "midlayer" decoder that combines token embeddings with a
//     "hidden_states" input (dual RMSNorm + Concat feeding 2*hidden-wide Q/K/V
//     projections), outputs draft-vocab "logits" plus the midlayer residual as
//     "last_hidden_state". The raw export's fc has already been rehomed to the
//     target model, so "hidden_states" is consumed directly.
//

#include "model_builder_eagle3.hpp"

#include <cstdint>
#include <vector>

#include "model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

std::shared_ptr<ov::op::v0::Result> make_manually_added_result(const ov::Output<ov::Node>& value,
                                                               const std::string& name) {
    auto result = std::make_shared<ov::op::v0::Result>(value);
    result->set_friendly_name(name);
    result->output(0).set_names({name});
    result->get_rt_info()[kManuallyAddedOutput] = true;
    return result;
}

ov::Output<ov::Node> make_eagle3_hidden_capture(const ov::OutputVector& captured_layers,
                                                size_t hidden_size,
                                                ov::element::Type precision,
                                                const WeightFn& fc_weight) {
    OPENVINO_ASSERT(!captured_layers.empty(), "Eagle3 hidden capture requires at least one layer output");

    ov::Output<ov::Node> hidden = captured_layers.front();
    if (captured_layers.size() > 1) {
        auto concat = std::make_shared<ov::opset11::Concat>(captured_layers, -1);
        concat->set_friendly_name("eagle3_hidden_states_concat");
        hidden = concat->output(0);
    }

    if (fc_weight) {
        hidden = make_linear(hidden,
                             captured_layers.size() * hidden_size,
                             hidden_size,
                             "eagle3_hidden_state_fc",
                             precision,
                             fc_weight);
    }
    return hidden;
}

ov::Output<ov::Node> make_eagle3_d2t(size_t draft_vocab_size, size_t target_vocab_size) {
    OPENVINO_ASSERT(draft_vocab_size > 0 && draft_vocab_size <= target_vocab_size,
                    "Eagle3 d2t requires 0 < draft_vocab_size (",
                    draft_vocab_size,
                    ") <= target_vocab_size (",
                    target_vocab_size,
                    ")");
    // Monotonic spread of the draft vocab over the target vocab, stored as
    // offsets: target_id = draft_id + d2t[draft_id].
    std::vector<int64_t> offsets(draft_vocab_size);
    for (size_t i = 0; i < draft_vocab_size; ++i) {
        offsets[i] = static_cast<int64_t>(i * target_vocab_size / draft_vocab_size) - static_cast<int64_t>(i);
    }
    auto d2t = ov::opset11::Constant::create(ov::element::i64, ov::Shape{draft_vocab_size}, offsets);
    d2t->set_friendly_name("model.d2t");
    return d2t->output(0);
}

std::shared_ptr<ov::Model> ModelBuilder::build_eagle3_draft(const Eagle3DraftConfig& config_in) {
    clear();

    Eagle3DraftConfig config = config_in;
    if (!config.norm)
        config.norm = RMSNorm(config.hidden_size, config.precision);
    if (!config.ffn)
        config.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.precision, config.weight);

    const auto prec = config.precision;
    const auto hs = config.hidden_size;
    const auto kv_heads = config.get_kv_heads();
    const std::string layer = "model.midlayer.";

    OPENVINO_ASSERT(config.num_captured_layers >= 1, "Eagle3 draft requires num_captured_layers >= 1");
    OPENVINO_ASSERT(config.lm_head_weight,
                    "Eagle3 draft requires lm_head_weight: draft-vocab \"logits\" are its primary output");
    OPENVINO_ASSERT(!config.dynamic_hidden_states || config.num_captured_layers == 1,
                    "dynamic_hidden_states relies on the get_static_input fallback (hidden_size width), "
                    "so it is only valid with num_captured_layers == 1");
    const size_t hidden_width = config.num_captured_layers * hs;

    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    // Feature dim is num_captured_layers * hidden_size. Kept static like the real
    // NPU export, unless dynamic_hidden_states forces the ReshapeToStatic fallback
    // to recover it from the last_hidden_state output.
    auto hidden_feat = config.dynamic_hidden_states ? ov::Dimension(-1) : ov::Dimension(hidden_width);
    auto hidden_states = parameter(ov::element::f32, ov::PartialShape{-1, -1, hidden_feat}, "hidden_states");
    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");
    auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    setup_position_ids(config, input_ids->output(0));

    ov::Output<ov::Node> hidden_in = hidden_states->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(hidden_in, prec);
        cvt->set_friendly_name("model.hidden_states_convert");
        hidden_in = cvt->output(0);
    }

    // fc projects the concatenated multi-layer hidden state back to hidden_size.
    // GenAI keeps this in the NPU draft (it rehomes fc into the target only for
    // the continuous-batching pipeline). A single captured layer needs no fc.
    ov::Output<ov::Node> hidden_proj = hidden_in;
    if (hidden_width != hs) {
        hidden_proj = make_linear(hidden_in, hidden_width, hs, "model.fc", prec, config.weight);
    }

    // Dual-stream front: normalized token embedding and normalized hidden state
    // are concatenated into a 2*hidden-wide attention input.
    auto embed = make_embedding(input_ids->output(0), config.vocab_size, hs, "model.embed_tokens", prec);
    auto embed_normed = config.norm(embed, layer + "input_layernorm");
    auto hidden_normed = config.norm(hidden_proj, layer + "hidden_norm");
    auto cat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{embed_normed, hidden_normed}, -1);
    cat->set_friendly_name(layer + "concat");

    ov::Output<ov::Node> sdpa_mask = make_causal_mask(input_ids->output(0), attention_mask->output(0), prec);
    if (config.use_tree_mask) {
        // Additive tree mask for the topk pipeline. NPUW reshapes it to
        // {1,1,1,1} (prefill) / {1,1,input,kvcache} (generate), both of which
        // broadcast onto the causal mask.
        auto tree_mask = parameter(prec, ov::PartialShape{-1, 1, -1, -1}, "eagle_tree_mask");
        auto masked = std::make_shared<ov::opset11::Add>(sdpa_mask, tree_mask);
        masked->set_friendly_name("model.tree_mask_add");
        sdpa_mask = masked->output(0);
    }

    Attention attn = config.make_attention();
    attn.sdpa_mask = sdpa_mask;
    attn.kv_cache_fn =
        make_decoder_kv_cache_fn(input_ids->output(0), beam_idx->output(0), kv_heads, config.head_dim, prec);

    // Q/K/V project from the 2*hidden concat, so the layer projects explicitly
    // and uses Attention's pre-projected entry point.
    const size_t cat_dim = 2 * hs;
    const size_t kv_dim = kv_heads * config.head_dim;
    auto q = make_linear(cat, cat_dim, hs, layer + "self_attn.q_proj", prec, config.weight, config.attn_bias);
    auto k = make_linear(cat, cat_dim, kv_dim, layer + "self_attn.k_proj", prec, config.weight, config.attn_bias);
    auto v = make_linear(cat, cat_dim, kv_dim, layer + "self_attn.v_proj", prec, config.weight, config.attn_bias);
    auto attn_out = attn(q, k, v, layer, 0);

    // Residual anchors on the fc-projected hidden state (the "midlayer" add).
    auto residual1 = std::make_shared<ov::opset11::Add>(hidden_proj, attn_out);
    residual1->set_friendly_name(layer + "attn_residual");

    auto normed2 = config.norm(residual1->output(0), layer + "post_attention_layernorm");
    auto ffn_out = config.ffn(normed2, layer + "mlp");
    auto residual2 = std::make_shared<ov::opset11::Add>(residual1, ffn_out);
    residual2->set_friendly_name(layer + "ffn_residual");

    // GenAI taps the midlayer residual as the draft's last_hidden_state.
    add_output(make_manually_added_result(residual2->output(0), "last_hidden_state"));

    if (config.with_d2t) {
        auto d2t = make_eagle3_d2t(config.draft_vocab_size, config.vocab_size);
        auto d2t_result = std::make_shared<ov::op::v0::Result>(d2t);
        d2t_result->set_friendly_name("d2t");
        d2t_result->output(0).set_names({"d2t"});
        add_output(d2t_result);
    }

    auto final_norm = config.norm(residual2->output(0), "model.norm");
    auto logits = make_lm_head(final_norm, hs, config.draft_vocab_size, "lm_head", prec, config.lm_head_weight);

    return make_model(logits, "logits", "synthetic_eagle3_draft");
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
