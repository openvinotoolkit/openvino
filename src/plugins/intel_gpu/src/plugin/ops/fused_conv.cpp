// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/fused_conv.hpp>
#include "openvino/op/fused_conv.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace op {
namespace internal {
using FusedConv = ov::op::FusedConv;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateFusedConvOp(ProgramBuilder& p, const std::shared_ptr<ov::op::FusedConv>& op) {
    validate_inputs_count(op, {4});

    auto inputs = p.GetInputInfo(op);

    const std::string layerName = layer_type_name_ID(op);
    cldnn::fused_conv fused_conv_prim(layerName, inputs, op->get_variable()->get_info());
    fused_conv_prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, fused_conv_prim);
}

REGISTER_FACTORY_IMPL(internal, FusedConv);

}  // namespace ov::intel_gpu
