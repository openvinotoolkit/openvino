// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
/**
 * @interface LoraSubgraph
 * @brief LoraSubgraph operation, which is used for LoRA subgraphs fusion.
 * It always has only 1 output, and the following inputs, whose order is fixed:
 * 1. main_flow_input: input from original model.
 * 2. LoRA_input: input to which the Low-Rank adaptation is applied.
 *    The adapted input is combined with `main_flow_input`.
 * 3. LoRA_matrices: 3 Low-Rank adaptation matrices applied to `LoRA_input`.
 * The fused subgraph can be optimized in runtime based on LoRA semantic.
 * For instance, `main_flow_input` can be fast-forwarded to output in case of empty `LoRA_matrices`.
 */
class TRANSFORMATIONS_API LoraSubgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("LoraSubgraph", "ie_internal_opset", ov::op::util::SubGraphOp);

    LoraSubgraph() = default;
    LoraSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
