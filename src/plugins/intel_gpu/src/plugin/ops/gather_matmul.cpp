// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/gather_matmul.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

namespace ov {
namespace op {
namespace internal {
using GatherMatmulCompressed = ov::op::internal::GatherMatmulCompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov

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
    int32_t n_activated_experts = a_shape[0].is_static() ? static_cast<int32_t>(a_shape[0].get_length()) : 0;

    // Detect if zero points are present by checking if input 5 has actual data
    // (not a scalar constant 0)
    bool has_zp = false;
    if (op->get_input_size() > 5) {
        const auto& zp_shape = op->get_input_partial_shape(5);
        // If ZP shape has more than 1 element, it's real ZP data
        if (zp_shape.rank().is_static() && zp_shape.rank().get_length() > 0) {
            bool is_scalar = true;
            for (int64_t i = 0; i < zp_shape.rank().get_length(); i++) {
                if (zp_shape[i].is_dynamic() || zp_shape[i].get_length() > 1) {
                    is_scalar = false;
                    break;
                }
            }
            has_zp = !is_scalar;
        }
    }

    // Check if bias is present (not a scalar constant 0)
    bool has_bias = false;
    if (op->get_input_size() > 3) {
        const auto& bias_shape = op->get_input_partial_shape(3);
        if (bias_shape.rank().is_static() && bias_shape.rank().get_length() > 0) {
            bool is_scalar = true;
            for (int64_t i = 0; i < bias_shape.rank().get_length(); i++) {
                if (bias_shape[i].is_dynamic() || bias_shape[i].get_length() > 1) {
                    is_scalar = false;
                    break;
                }
            }
            has_bias = !is_scalar;
        }
    }

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gather_matmul bgm(layerName, inputs, has_bias, has_zp, n_activated_experts);

    p.add_primitive(*op, bgm);
}

REGISTER_FACTORY_IMPL(internal, GatherMatmulCompressed);

}  // namespace ov::intel_gpu
