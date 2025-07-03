// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "intel_gpu/op/lora_subgraph_fused.hpp"
#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/lora_subgraph.hpp"

using Lora = ov::op::internal::LoraSubgraph;

namespace ov::op::internal {
using LoraSubgraphFused = ov::intel_gpu::op::LoraSubgraphFused;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void LoraSubgraphImpl(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, bool transposed_states) {
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto lora = cldnn::lora(primitive_name, inputs, transposed_states);

    p.add_primitive(*op, lora);
}

static void CreateLoraSubgraphOp(ProgramBuilder& p, const std::shared_ptr<Lora>& op) {
    validate_inputs_count(op, {5});

    bool transposed_states = true;
    const auto& subgraph_ops = op->get_function()->get_ops();
    for (const auto& op : subgraph_ops) {
        if (ov::is_type<const ov::intel_gpu::op::Gemm>(op.get())) {
            const auto& gemm = ov::as_type<const ov::intel_gpu::op::Gemm>(op.get());
            // Assumption that all states are simultaneously transposed or not transposed
            const auto& input1_order = gemm->get_input1_transpose_order();
            const auto& default_order = op::Gemm::default_order(input1_order.size());
            transposed_states = !std::equal(default_order.begin(), default_order.end(), input1_order.begin());
            break;
        }
    }

    LoraSubgraphImpl(p, op, transposed_states);
}

static void CreateLoraSubgraphFusedOp(ProgramBuilder& p, const std::shared_ptr<op::LoraSubgraphFused>& op) {
    validate_inputs_count(op, {8, 11});
    LoraSubgraphImpl(p, op, op->is_transposed_states());
}

REGISTER_FACTORY_IMPL(internal, LoraSubgraph);
REGISTER_FACTORY_IMPL(internal, LoraSubgraphFused);

}  // namespace ov::intel_gpu
