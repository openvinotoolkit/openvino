// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
class TRANSFORMATIONS_API MSDA : public ov::op::Op {
public:
    OPENVINO_OP("MSDA", "ie_internal_opset", ov::op::Op);

    MSDA() = default;

    MSDA(const OutputVector& inputs);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov