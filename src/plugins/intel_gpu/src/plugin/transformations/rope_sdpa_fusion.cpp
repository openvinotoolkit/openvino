// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_sdpa_fusion.hpp"

#include <vector>

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_gpu {

// The kernel reads the table densely as (batch, tokens, head_size) halves: row stride head_size,
// batch stride tokens*head_size. So head_size has to be innermost and the dimensions carrying
// batch and tokens have to appear in that order with nothing but unit dims around them. Matching
// on the product instead would admit [1, batch*tokens, 1, head_size], which is the same element
// count laid out so that the reference RoPE kernel and this one read different elements. Unit
// dimensions do not move anything in memory, so they are dropped rather than rejected.
static bool is_flat_cos_sin(const ov::Output<ov::Node>& out, int64_t batch, int64_t tokens, int64_t head_size) {
    const auto& pshape = out.get_partial_shape();
    if (pshape.is_dynamic() || pshape.size() < 2)
        return false;
    if (pshape[pshape.size() - 1].get_length() != head_size)
        return false;

    // The token axis is the one the kernel strides by head_size, so it has to be the innermost
    // leading dimension that carries anything. Matching it by value anywhere in the shape would
    // accept a table whose batch and token axes are swapped whenever one of the two happens to
    // be 1 -- same element count, different arrangement, and the kernel reads the wrong rows.
    size_t lead = pshape.size() - 1;
    if (tokens != 1) {
        while (lead > 1 && pshape[lead - 1].get_length() == 1)
            lead--;
    }
    if (lead == 0 || pshape[lead - 1].get_length() != tokens)
        return false;

    std::vector<int64_t> outer;
    for (size_t i = 0; i + 1 < lead; i++) {
        const int64_t dim = pshape[i].get_length();
        if (dim != 1)
            outer.push_back(dim);
    }

    std::vector<int64_t> expected;
    if (batch != 1)
        expected.push_back(batch);
    return outer == expected;
}

RoPESDPAFusion::RoPESDPAFusion() {
    using namespace ov::pass::pattern;

    // Only the plugin's own plain SDPA can absorb the rotation: IndirectSDPA reaches the primitive
    // through a different creator, a compressed KV has no room for two more inputs, and an SDPA
    // that already carries a rotated Q must not be handed a second table.
    auto plain_sdpa = ov::pass::pattern::op::Predicate(
        [](const ov::Output<ov::Node>& out) -> bool {
            auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(out.get_node_shared_ptr());
            return sdpa && !ov::as_type_ptr<ov::intel_gpu::op::IndirectSDPA>(sdpa) && !sdpa->get_kv_compressed() &&
                   !sdpa->get_rope_q();
        },
        "plain_sdpa()");

    // Q side of an interleaved RoPE: exactly three inputs, f16 in and out because that is all the
    // fused rotation in the micro-kernel reads, and a rank-4 [batch, tokens, heads, head_size] Q
    // whose three indexed dimensions the callback then requires to be constants.
    auto x_m = any_input(shape_matches("[batch, tokens, ?, head_size]"));
    auto cos_m = any_input(type_matches(ov::element::f16));
    auto sin_m = any_input(type_matches(ov::element::f16));
    auto rope_m = wrap_type<ov::op::internal::RoPE>({x_m, cos_m, sin_m},
                                                    consumers_count(1) && type_matches(ov::element::f16));

    // SDPA carries three to five inputs; spelling each arity out is what pins the RoPE to Q at
    // input 0 inside the pattern, since argument matching requires an exact input count. A sixth
    // input (the attention sink) is deliberately absent: the fusion appends cos/sin after the
    // existing inputs, and the generator reads them as the last two, which a sink would displace.
    auto k_m = any_input();
    auto v_m = any_input();
    auto sdpa_qkv_m = wrap_type<ov::intel_gpu::op::SDPA>({rope_m, k_m, v_m}, plain_sdpa);
    auto sdpa_mask_m = wrap_type<ov::intel_gpu::op::SDPA>({rope_m, k_m, v_m, any_input()}, plain_sdpa);
    auto sdpa_scale_m = wrap_type<ov::intel_gpu::op::SDPA>({rope_m, k_m, v_m, any_input(), any_input()}, plain_sdpa);
    auto sdpa_m =
        std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{sdpa_qkv_m, sdpa_mask_m, sdpa_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(m.get_match_root());
        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(
            m.get_pattern_value_map().at(rope_m).get_node_shared_ptr());

        const auto& cfg = rope->get_config();
        if (!cfg.is_interleaved || cfg.input_trans0213 || cfg.output_trans0213 || cfg.is_chatglm ||
            cfg.is_qwen || cfg.support_2d_rope || cfg.support_3d_rope || cfg.is_ltx_video ||
            cfg.use_rope_cache || cfg.gather_position_arg_id != 0 || cfg.slice_start != cfg.slice_stop)
            return false;

        // The kernel indexes the cos/sin table with the sequence length and head size it reads
        // through input_q_transpose_order, while the pattern binds them from Q's physical shape.
        // Those coincide only for the canonical [batch, tokens, heads, head_size] -> BHLS order,
        // and any other permutation makes the kernel walk the table along the wrong axis.
        static const std::vector<int64_t> bhls_from_btns{0, 2, 1, 3};
        if (sdpa->get_input0_transpose_order() != bhls_from_btns)
            return false;

        // An i8 key selects the integer K^T*Q contraction, whose contract is that Q already holds
        // exact integer codes so the micro-kernel's rounding into SLM is lossless. A rotation
        // mixes two codes per output element, so the rotated value is neither integral nor
        // bounded by the original code range -- it saturates. The two cannot both apply.
        if (sdpa->get_input_element_type(1) == ov::element::i8)
            return false;

        // shape_matches also binds a dynamic dimension that carries a symbol; the three the kernel
        // indexes with have to be constants.
        auto& symbols = m.get_symbols();
        const auto& batch_sym = symbols["batch"];
        const auto& tokens_sym = symbols["tokens"];
        const auto& head_size_sym = symbols["head_size"];
        if (!batch_sym.is_integer() || !tokens_sym.is_integer() || !head_size_sym.is_integer())
            return false;
        const int64_t batch = batch_sym.i();
        const int64_t tokens = tokens_sym.i();
        const int64_t head_size = head_size_sym.i();
        if (cfg.rotary_ndims != static_cast<size_t>(head_size) || head_size % 2 != 0)
            return false;

        for (size_t i = 1; i < 3; i++) {
            if (!is_flat_cos_sin(rope->input_value(i), batch, tokens, head_size))
                return false;
        }

        if (transformation_callback(sdpa))
            return false;

        // cos/sin go last so the existing mask/scale slots keep their indices.
        ov::OutputVector inputs = sdpa->input_values();
        inputs[0] = rope->input_value(0);
        inputs.push_back(rope->input_value(1));
        inputs.push_back(rope->input_value(2));

        auto new_sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs,
                                                                 sdpa->get_causal(),
                                                                 sdpa->get_input0_transpose_order(),
                                                                 sdpa->get_input1_transpose_order(),
                                                                 sdpa->get_input2_transpose_order(),
                                                                 sdpa->get_output_transpose_order(),
                                                                 sdpa->get_output_type(),
                                                                 true,
                                                                 sdpa->get_causal_mask_alignment());
        new_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{rope, sdpa}, new_sdpa);
        ov::replace_node(sdpa, new_sdpa);
        return true;
    };

    this->register_matcher(std::make_shared<Matcher>(sdpa_m, "RoPESDPAFusion"), callback);
}

}  // namespace ov::intel_gpu
