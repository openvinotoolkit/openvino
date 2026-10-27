// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_rms_rope.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_gpu {

bool FuseRMSRoPE::run_on_model(const std::shared_ptr<ov::Model>& model) {
    size_t fused = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto rope = ov::as_type_ptr<ov::op::internal::RoPE>(node);
        if (!rope)
            continue;

        const auto& config = rope->get_config();

        if (rope->get_input_size() != 3 || config.gather_position_arg_id != 0 || config.is_qwen || config.is_chatglm || config.is_ltx_video ||
            config.is_interleaved || config.input_interleaved || config.output_trans0213 || config.slice_start != 0 || config.slice_stop != 0 ||
            rope->get_input_element_type(0) != ov::element::f16) {
            continue;
        }

        // Full-head RotateHalf only: the fused kernel normalizes exactly the rotated elements.
        const auto& rope_shape = rope->get_input_partial_shape(0);
        if (rope_shape.rank().is_dynamic() || rope_shape.rank().get_length() != 4 || !rope_shape[3].is_static())
            continue;
        const auto head_size = static_cast<size_t>(rope_shape[3].get_length());
        // rope_opt reduces the head only on its 16-wide f16 path: f16 cos/sin and head_size % 32 == 0.
        if (head_size != config.rotary_ndims || head_size % 32 != 0 || rope->get_input_element_type(1) != ov::element::f16 ||
            rope->get_input_element_type(2) != ov::element::f16)
            continue;

        // The RMSNorm either feeds RoPE directly (transpose already absorbed) or through a single view.
        const auto producer = rope->get_input_node_shared_ptr(0);
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(producer);
        std::shared_ptr<ov::Node> view;
        if (!rms) {
            if ((!ov::is_type<ov::op::v1::Reshape>(producer) && !ov::is_type<ov::op::v1::Transpose>(producer)) ||
                producer->output(0).get_target_inputs().size() != 1) {
                continue;
            }
            view = producer;
            rms = ov::as_type_ptr<ov::op::internal::RMS>(view->get_input_node_shared_ptr(0));
        }

        if (!rms || !rms->get_elementwise_affine() || rms->get_input_size() != 2 || rms->output(0).get_target_inputs().size() != 1 ||
            rms->get_output_element_type(0) != ov::element::f16 || rms->get_input_element_type(1) != ov::element::f16) {
            continue;
        }

        const auto gamma_shape = rms->get_input_partial_shape(1);
        if (!gamma_shape.is_static() || ov::shape_size(gamma_shape.to_shape()) != head_size)
            continue;

        const ov::Output<ov::Node> rope_data = view ? view->output(0) : rms->input_value(0);
        ov::OutputVector inputs{rope_data, rope->input_value(1), rope->input_value(2), rms->input_value(1)};
        auto fused_rope = std::make_shared<ov::op::internal::RoPE>(inputs, config);
        fused_rope->set_friendly_name(rope->get_friendly_name());
        if (view) {
            ov::copy_runtime_info({rms, view, rope}, fused_rope);
            view->input(0).replace_source_output(rms->input_value(0));
        } else {
            ov::copy_runtime_info({rms, rope}, fused_rope);
        }
        fused_rope->get_rt_info()[fuse_rms_rope_epsilon_key] = static_cast<float>(rms->get_epsilon());

        ov::replace_node(rope, fused_rope);
        ++fused;
        GPU_DEBUG_TRACE_DETAIL << "[rms_rope] fused into " << fused_rope->get_friendly_name() << std::endl;
    }

    GPU_DEBUG_TRACE << "[rms_rope] fused " << fused << " RMSNorm + RoPE pairs" << std::endl;
    return fused > 0;
}

}  // namespace ov::intel_gpu
