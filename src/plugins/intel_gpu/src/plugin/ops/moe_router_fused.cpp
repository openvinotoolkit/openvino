// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/op/moe_router_fused.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_router_fused.hpp"

namespace ov {
namespace op {
namespace internal {
using MoERouterFused = ov::intel_gpu::op::MoERouterFused;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateMoERouterFusedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MoERouterFused>& op) {
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    const size_t expected_inputs = (config.routing_type == ov::op::internal::MoERouterFused::RoutingType::SIGMOID_BIAS) ? 3 : 1;
    validate_inputs_count(op, {expected_inputs});
    const std::string layerName = layer_type_name_ID(op);
    p.add_primitive(*op, cldnn::moe_router_fused(layerName, inputs, config));
}

REGISTER_FACTORY_IMPL(internal, MoERouterFused);

}  // namespace ov::intel_gpu
