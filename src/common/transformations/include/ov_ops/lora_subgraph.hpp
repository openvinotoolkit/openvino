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
class TRANSFORMATIONS_API LoraSubgraph : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("LoraSubgraph", "ie_internal_opset");

    LoraSubgraph() = default;
    LoraSubgraph(const OutputVector& args, const std::shared_ptr<ov::Model>& body);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
