// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <intel_gpu/primitives/paged_gated_delta_net.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov {
namespace op {
namespace internal {
using PagedGatedDeltaNet = ov::op::internal::PagedGatedDeltaNet;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreatePagedGatedDeltaNetOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedGatedDeltaNet>& op) {
    validate_inputs_count(op, {11});

    auto inputs = p.GetInputInfo(op);
    const std::string layer_name = layer_type_name_ID(op);
    cldnn::paged_gated_delta_net prim(layer_name, inputs, op->get_use_qk_l2norm(), op->get_q_l2_norm_eps(), op->get_k_l2_norm_eps());

    const auto query_ps = op->get_input_partial_shape(0);
    const auto key_ps = op->get_input_partial_shape(1);
    const auto value_ps = op->get_input_partial_shape(2);

    if (query_ps.rank().is_static() && key_ps.rank().is_static() && value_ps.rank().is_static()) {
        const auto query_rank = query_ps.rank().get_length();
        const auto key_rank = key_ps.rank().get_length();
        const auto value_rank = value_ps.rank().get_length();

        if (query_rank >= 2 && key_rank >= 2 && value_rank >= 2) {
            const auto k_head_size_dim = key_ps[key_rank - 1];
            const auto v_head_size_dim = value_ps[value_rank - 1];
            const auto k_heads_num_dim = query_ps[query_rank - 2];
            const auto v_heads_num_dim = value_ps[value_rank - 2];

            if (k_head_size_dim.is_static())
                prim.k_head_size = k_head_size_dim.get_length();
            if (v_head_size_dim.is_static())
                prim.v_head_size = v_head_size_dim.get_length();
            if (k_heads_num_dim.is_static())
                prim.k_heads_num = k_heads_num_dim.get_length();
            if (v_heads_num_dim.is_static())
                prim.v_heads_num = v_heads_num_dim.get_length();
        }
        OPENVINO_ASSERT(prim.k_head_size > 0, "PagedGatedDeltaNet must have k_head_size > 0");
        OPENVINO_ASSERT(prim.v_head_size > 0, "PagedGatedDeltaNet must have v_head_size > 0");
        OPENVINO_ASSERT(prim.k_heads_num > 0, "PagedGatedDeltaNet must have k_heads_num > 0");
        OPENVINO_ASSERT(prim.v_heads_num > 0, "PagedGatedDeltaNet must have v_heads_num > 0");
    }

    prim.num_outputs = 1;
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedGatedDeltaNet);

}  // namespace ov::intel_gpu
