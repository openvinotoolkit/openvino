// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include <intel_gpu/primitives/gated_delta_net.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov {
namespace op {
namespace internal {
using GatedDeltaNet = ov::op::internal::GatedDeltaNet;
using GatedDeltaNetWithVariable = ov::op::internal::GatedDeltaNetWithVariable;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateGatedDeltaNetOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatedDeltaNet>& op) {
    validate_inputs_count(op, {6});

    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    cldnn::gated_delta_net gated_delta_net_prim(layerName, inputs, op->get_fuse_qk_l2norm(), op->get_q_l2_norm_eps(), op->get_k_l2_norm_eps());

    const auto query_ps = op->get_input_partial_shape(0);
    const auto key_ps = op->get_input_partial_shape(1);
    const auto value_ps = op->get_input_partial_shape(2);

    if (query_ps.rank().is_static() && key_ps.rank().is_static() && value_ps.rank().is_static()) {
        const auto query_rank = query_ps.rank().get_length();
        const auto key_rank = key_ps.rank().get_length();
        const auto value_rank = value_ps.rank().get_length();

        if (query_rank >= 2 && key_rank >= 1 && value_rank >= 2) {
            const auto k_head_size_dim = key_ps[key_rank - 1];
            const auto v_head_size_dim = value_ps[value_rank - 1];
            const auto k_heads_num_dim = query_ps[query_rank - 2];
            const auto v_heads_num_dim = value_ps[value_rank - 2];

            if (k_head_size_dim.is_static())
                gated_delta_net_prim.k_head_size = k_head_size_dim.get_length();
            if (v_head_size_dim.is_static())
                gated_delta_net_prim.v_head_size = v_head_size_dim.get_length();
            if (k_heads_num_dim.is_static())
                gated_delta_net_prim.k_heads_num = k_heads_num_dim.get_length();
            if (v_heads_num_dim.is_static())
                gated_delta_net_prim.v_heads_num = v_heads_num_dim.get_length();
        }
    }

    gated_delta_net_prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, gated_delta_net_prim);
}

static void CreateGatedDeltaNetWithVariableOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatedDeltaNetWithVariable>& op) {
    validate_inputs_count(op, {6});

    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    cldnn::gated_delta_net gated_delta_net_prim(layerName,
                                                inputs,
                                                op->get_fuse_qk_l2norm(),
                                                op->get_q_l2_norm_eps(),
                                                op->get_k_l2_norm_eps(),
                                                op->get_variable()->get_info());

    const auto query_ps = op->get_input_partial_shape(0);
    const auto key_ps = op->get_input_partial_shape(1);
    const auto value_ps = op->get_input_partial_shape(2);

    if (query_ps.rank().is_static() && key_ps.rank().is_static() && value_ps.rank().is_static()) {
        const auto query_rank = query_ps.rank().get_length();
        const auto key_rank = key_ps.rank().get_length();
        const auto value_rank = value_ps.rank().get_length();

        if (query_rank >= 2 && key_rank >= 1 && value_rank >= 2) {
            const auto k_head_size_dim = key_ps[key_rank - 1];
            const auto v_head_size_dim = value_ps[value_rank - 1];
            const auto k_heads_num_dim = query_ps[query_rank - 2];
            const auto v_heads_num_dim = value_ps[value_rank - 2];

            if (k_head_size_dim.is_static())
                gated_delta_net_prim.k_head_size = k_head_size_dim.get_length();
            if (v_head_size_dim.is_static())
                gated_delta_net_prim.v_head_size = v_head_size_dim.get_length();
            if (k_heads_num_dim.is_static())
                gated_delta_net_prim.k_heads_num = k_heads_num_dim.get_length();
            if (v_heads_num_dim.is_static())
                gated_delta_net_prim.v_heads_num = v_heads_num_dim.get_length();
        }
    }

    gated_delta_net_prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, gated_delta_net_prim);
}

REGISTER_FACTORY_IMPL(internal, GatedDeltaNet);
REGISTER_FACTORY_IMPL(internal, GatedDeltaNetWithVariable);

}  // namespace ov::intel_gpu