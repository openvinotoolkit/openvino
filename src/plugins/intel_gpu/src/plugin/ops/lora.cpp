// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "intel_gpu/op/lora_subgraph_fused.hpp"
#include "ov_ops/lora_subgraph.hpp"

using Lora = ov::op::internal::LoraSubgraph;

namespace ov::op::internal {
using LoraSubgraphFused = ov::intel_gpu::op::LoraSubgraphFused;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void LoraSubgraphImpl(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op) {
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto lora = cldnn::lora(primitive_name, inputs);

    p.add_primitive(*op, lora);
}

static void CreateLoraSubgraphOp(ProgramBuilder& p, const std::shared_ptr<Lora>& op) {
    validate_inputs_count(op, {5});
    LoraSubgraphImpl(p, op);
}

static void CreateLoraSubgraphFusedOp(ProgramBuilder& p, const std::shared_ptr<op::LoraSubgraphFused>& op) {
    validate_inputs_count(op, {8, 11});
    LoraSubgraphImpl(p, op);
}

REGISTER_FACTORY_IMPL(internal, LoraSubgraph);
REGISTER_FACTORY_IMPL(internal, LoraSubgraphFused);

}  // namespace ov::intel_gpu
