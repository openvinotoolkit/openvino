// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/gather_matmul.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateGatherMatmulCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatherMatmulCompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    // GatherMatmulCompressed inputs:
    //   0: A             - [n_activated_experts, batch*seq, hidden_size]
    //   1: B             - [n_all_experts, N, K] (weights, transposed)
    //   2: indices        - [batch*seq, top_k]
    //   3: bias           - [n_all_experts, 1, N] or scalar 0
    //   4: weight_scales  - [n_all_experts, N, groups]
    //   5: weight_zero_points - [n_all_experts, N, groups] or empty
    validate_inputs_count(op, {6});

    const auto& a_shape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 1 && a_shape[0].is_static(),
                    "GatherMatmulCompressed requires static n_activated_experts (input A dim[0]), got shape ",
                    a_shape);
    const int32_t n_activated_experts = static_cast<int32_t>(a_shape[0].get_length());

    // Only constant decompression is supported, so shapes are always static.
    // Placeholder inputs have shape_size <= 1 (scalar 0 or empty).
    const bool has_bias = ov::shape_size(op->get_input_shape(3)) > 1;
    const bool has_zp = ov::shape_size(op->get_input_shape(5)) > 1;

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gather_matmul bgm(layerName, inputs, has_bias, has_zp, n_activated_experts);

    p.add_primitive(*op, bgm);
}

REGISTER_FACTORY_IMPL(internal, GatherMatmulCompressed);

}  // namespace ov::intel_gpu
