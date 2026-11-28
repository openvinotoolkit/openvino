// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/op/moe_scatter_reduce.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_scatter_reduction.hpp"

namespace ov {
namespace op {
namespace internal {
using MoEScatterReduce = ov::intel_gpu::op::MoEScatterReduce;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateMoEScatterReduceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MoEScatterReduce>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    const auto& weights_shape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(weights_shape.rank().is_static() && weights_shape[1].is_static(),
                    "[GPU] MoEScatterReduce: top_k dimension must be static, got ",
                    weights_shape);
    const auto top_k = static_cast<int32_t>(weights_shape[1].get_length());
    p.add_primitive(*op, cldnn::moe_scatter_reduction(layer_type_name_ID(op), inputs[0], inputs[1], inputs[2], top_k));
}

REGISTER_FACTORY_IMPL(internal, MoEScatterReduce);

}  // namespace ov::intel_gpu
