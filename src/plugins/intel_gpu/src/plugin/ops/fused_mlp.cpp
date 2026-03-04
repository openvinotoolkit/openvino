// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/fused_mlp.hpp"
#include "intel_gpu/primitives/fused_mlp.hpp"

using FusedMLP = ov::op::internal::FusedMLP;

namespace ov::intel_gpu {

static void CreateFusedMLPOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::FusedMLP>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layer_name = layer_type_name_ID(op);

    cldnn::fused_mlp prim(layer_name, inputs);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, FusedMLP);

}  // namespace ov::intel_gpu

