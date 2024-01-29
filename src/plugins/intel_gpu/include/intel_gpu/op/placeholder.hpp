// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class Placeholder : public ov::op::Op {
public:
    OPENVINO_OP("Placeholder", "gpu_opset");

    Placeholder();

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
