// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/gather_matmul.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateGatherMatmulOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatherMatmul>& op) {
    auto inputs = p.GetInputInfo(op);
    // Inputs: A, B (transposed), indices, [bias]
    validate_inputs_count(op, {3, 4});

    const auto& a_shape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 1,
                    "GatherMatmul requires static input A rank, got shape ", a_shape);
    // n_activated_experts is overridden at runtime from the actual input shape
    // (see GatherMatmulOCLImpl::update_rt_params); pass 0 when dim[0] is dynamic.
    const int32_t n_activated_experts =
        a_shape[0].is_static() ? static_cast<int32_t>(a_shape[0].get_length()) : 0;

    // Placeholder inputs have shape_size <= 1.
    const bool has_bias = op->get_input_size() == 4 && ov::shape_size(op->get_input_shape(3)) > 1;

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gather_matmul bgm(layerName, inputs, has_bias, /*has_zp=*/false, n_activated_experts);

    p.add_primitive(*op, bgm);
}

static void CreateGatherMatmulCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatherMatmulCompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    // Inputs: A, B, indices, bias, scales, zp (placeholder when absent).
    validate_inputs_count(op, {6});

    const auto& a_shape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 1 && a_shape[0].is_static(),
                    "GatherMatmulCompressed requires static n_activated_experts (input A dim[0]), got shape ",
                    a_shape);
    const int32_t n_activated_experts = static_cast<int32_t>(a_shape[0].get_length());

    // Placeholder inputs (absent bias/zp) are empty constants with shape_size == 0.
    const bool has_bias = ov::shape_size(op->get_input_shape(3)) > 1;
    const size_t zp_elems = ov::shape_size(op->get_input_shape(5));
    // A scalar (per-tensor) zp has exactly 1 element and is a real zero point, not a
    // placeholder. Only the u2 reference path consumes scalar zps today; for other
    // weight dtypes keep the previous "treat as absent" behavior so their per-group
    // zp indexing is not disturbed.
    const bool weights_u2 = op->get_input_element_type(1) == ov::element::u2;
    const bool has_zp = zp_elems > 1 || (weights_u2 && zp_elems == 1);

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::gather_matmul bgm(layerName, inputs, has_bias, has_zp, n_activated_experts);

    p.add_primitive(*op, bgm);
}

REGISTER_FACTORY_IMPL(internal, GatherMatmul);
REGISTER_FACTORY_IMPL(internal, GatherMatmulCompressed);

}  // namespace ov::intel_gpu
