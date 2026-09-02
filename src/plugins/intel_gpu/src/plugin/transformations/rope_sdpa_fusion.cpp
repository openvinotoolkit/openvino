// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_sdpa_fusion.hpp"

#include <cstdlib>
#include <string>

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_gpu {

// The kernel indexes the table as (batch, token, head_size) halves, so the leading dims have to
// collapse to batch*tokens with head_size innermost and nothing else in between.
static bool is_flat_cos_sin(const ov::Output<ov::Node>& out, int64_t batch, int64_t tokens, int64_t head_size) {
    const auto& pshape = out.get_partial_shape();
    if (pshape.is_dynamic() || pshape.size() < 2)
        return false;
    if (pshape[pshape.size() - 1].get_length() != head_size)
        return false;
    int64_t lead = 1;
    for (size_t i = 0; i + 1 < pshape.size(); i++)
        lead *= pshape[i].get_length();
    return lead == batch * tokens;
}

RoPESDPAFusion::RoPESDPAFusion() {
    using namespace ov::pass::pattern;

    auto rope_m = wrap_type<ov::op::internal::RoPE>(consumers_count(1));

    const char* disable = std::getenv("OV_ROPE_SDPA");
    const bool enabled = !disable || std::string(disable) != "0";

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        if (!enabled)
            return false;
        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(m.get_match_root());
        if (!rope || rope->get_input_size() != 3)
            return false;
        const auto& cfg = rope->get_config();
        if (!cfg.is_interleaved || cfg.input_trans0213 || cfg.output_trans0213 || cfg.is_chatglm ||
            cfg.is_qwen || cfg.support_2d_rope || cfg.support_3d_rope || cfg.is_ltx_video ||
            cfg.use_rope_cache || cfg.gather_position_arg_id != 0 || cfg.slice_start != cfg.slice_stop)
            return false;

        const auto& x_shape = rope->get_input_partial_shape(0);
        if (x_shape.is_dynamic() || x_shape.size() != 4)
            return false;
        const int64_t batch = x_shape[0].get_length();
        const int64_t tokens = x_shape[1].get_length();
        const int64_t head_size = x_shape[3].get_length();
        if (cfg.rotary_ndims != static_cast<size_t>(head_size) || head_size % 2 != 0)
            return false;

        if (rope->get_output_element_type(0) != ov::element::f16)
            return false;
        for (size_t i = 1; i < 3; i++) {
            if (rope->get_input_element_type(i) != ov::element::f16)
                return false;
            if (!is_flat_cos_sin(rope->input_value(i), batch, tokens, head_size))
                return false;
        }

        const auto& targets = rope->output(0).get_target_inputs();
        if (targets.size() != 1)
            return false;
        const auto& target = *targets.begin();
        if (target.get_index() != 0)
            return false;

        auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(target.get_node()->shared_from_this());
        // IndirectSDPA derives from SDPA but reaches the primitive through a different creator.
        if (!sdpa || ov::as_type_ptr<ov::intel_gpu::op::IndirectSDPA>(sdpa))
            return false;
        const auto sdpa_inputs = sdpa->get_input_size();
        if (sdpa->get_kv_compressed() || sdpa->get_rope_q() || sdpa_inputs < 3 || sdpa_inputs > 5)
            return false;
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
                                                                 true);
        new_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(ov::NodeVector{rope, sdpa}, new_sdpa);
        ov::replace_node(sdpa, new_sdpa);
        return true;
    };

    this->register_matcher(std::make_shared<Matcher>(rope_m, "RoPESDPAFusion"), callback);
}

}  // namespace ov::intel_gpu
